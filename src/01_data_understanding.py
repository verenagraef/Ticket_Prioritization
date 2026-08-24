"""CRISP-DM Data Understanding for the multilingual IT support ticket dataset."""

# %% 00 - Load packages

from pathlib import Path
import hashlib
import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.stats import chi2_contingency
from langdetect import DetectorFactory, detect_langs
from sklearn.neighbors import NearestNeighbors


# %% 01 - Configure analysis

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables" / "01_tickets_data_understanding"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "01_tickets_data_understanding"
CACHE_DIR = PROJECT_ROOT / "cache" / "01_tickets_data_understanding"
LANGUAGE_CACHE_PATH = CACHE_DIR / "language_detection_detailed.parquet"
EMBEDDING_CACHE_PATH = CACHE_DIR / "multilingual_embeddings.npy"
EMBEDDING_IDS_PATH = CACHE_DIR / "multilingual_embedding_source_rows.npy"
SEMANTIC_REVIEW_TEMPLATE_PATH = TABLES_DIR / "semantic_pair_review_sample.csv"
LANGUAGE_REVIEW_TEMPLATE_PATH = TABLES_DIR / "language_label_manual_review_sample.csv"
PRIORITY_REVIEW_TEMPLATE_PATH = TABLES_DIR / "priority_label_plausibility_review_sample.csv"
MANIFEST_PATH = TABLES_DIR / "01_data_understanding_manifest.json"

EXPECTED_RAW_RECORDS = 28_587
EXPECTED_RAW_VARIABLES = 16
EXPECTED_RAW_SHA256 = "f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0"
EXPECTED_REVIEW_COUNTS = {"semantic": 60, "language": 60, "priority": 30}
EXPECTED_SEMANTIC_REVIEW_PAIR_SHA256 = "493ec5afb1f9a19febc20c985d8c8d4f82aa70372504578333258792a86ed6b8"

ANSWER_COVERAGE_THRESHOLD = 0.70
ANSWER_COVERAGE_SENSITIVITY = [0.50, 0.60, 0.70, 0.80, 0.90]
VERY_SHORT_MAX_CHARACTERS = 20
SHORT_TEXT_MAX_CHARACTERS = 40
MISSING_WHITESPACE_MAX_WORDS = 2
LANGUAGE_CONFIDENCE_THRESHOLD = 0.80
NON_TARGET_LANGUAGE_CONFIDENCE_THRESHOLD = 0.99
SEMANTIC_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
SEMANTIC_THRESHOLDS = [0.90, 0.95, 0.98]
SEMANTIC_GROUP_THRESHOLD = 0.95
TARGET_LANGUAGES = {"de", "en"}
RANDOM_STATE = 42
DetectorFactory.seed = RANDOM_STATE

for folder in [TABLES_DIR, FIGURES_DIR, CACHE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 40)
pd.set_option("display.width", 180)


# %% 02 - Define helper functions

def fmt_int(value):
    return f"{int(value):,}".replace(",", ".")


def save_table(dataframe, filename):
    dataframe.to_csv(TABLES_DIR / filename, index=False, encoding="utf-8-sig")


def sha256_file(path, block_size=1_048_576):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def save_json(payload, path):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def semantic_review_pair_fingerprint(review_template):
    """Hash only the stable identity of the 60 semantic review pairs."""
    canonical = review_template[["pair_id", "text_a", "text_b"]].copy()
    canonical["pair_id"] = pd.to_numeric(
        canonical["pair_id"],
        errors="raise",
    ).astype("int64")
    payload = canonical.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def distribution_table(series, label, order=None):
    counts = series.fillna("<Missing>").value_counts()
    if order:
        remaining = [value for value in counts.index if value not in order]
        counts = counts.reindex(order + remaining, fill_value=0)
    table = counts.rename_axis(label).reset_index(name="records")
    table["share"] = table["records"] / table["records"].sum()
    return table


def print_distribution(title, table, label):
    total = pd.DataFrame([{label: "Total", "records": table["records"].sum(), "share": 1.0}])
    output = pd.concat([table, total], ignore_index=True)
    print(f"\n{title}")
    print(output.to_string(
        index=False,
        formatters={"records": fmt_int, "share": lambda value: f"{value:.4f}"},
    ))


def normalize_comparison_text(series):
    return (
        series.fillna("").astype(str).str.lower()
        .str.replace(r"\\[nrt]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def count_words(series):
    return (
        series.str.replace(r"\\[nrt]", " ", regex=True)
        .str.findall(r"(?u)\b\w+\b").str.len()
    )


def detect_ticket_language(value):
    try:
        result = detect_langs(str(value))[0]
    except Exception:
        return "uncertain", 0.0
    return str(result.lang), float(result.prob)


def duplicate_groups(values, name, target):
    return (
        pd.DataFrame({name: values, "priority": target})
        .query(f"{name} != ''")
        .groupby(name)
        .agg(records=("priority", "size"), distinct_priorities=("priority", "nunique"))
        .query("records > 1")
        .reset_index()
    )


def cramers_v_bias_corrected(table):
    chi2, p_value, dof, expected = chi2_contingency(table)
    n, (rows, columns) = table.to_numpy().sum(), table.shape
    if n <= 1 or rows <= 1 or columns <= 1:
        return chi2, p_value, dof, np.nan, expected

    phi2 = max(0.0, chi2 / n - ((columns - 1) * (rows - 1)) / (n - 1))
    rows_corr = rows - ((rows - 1) ** 2) / (n - 1)
    cols_corr = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(rows_corr - 1, cols_corr - 1)
    value = math.sqrt(phi2 / denominator) if denominator > 0 else np.nan
    return chi2, p_value, dof, value, expected


def holm_adjusted_pvalues(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0

    for rank, index in enumerate(order):
        corrected = (len(p_values) - rank) * p_values[index]
        running_max = max(running_max, corrected)
        adjusted[index] = min(running_max, 1.0)

    return adjusted



def save_distribution_plot(table, category, title, filename, horizontal=False):
    figure, axis = plt.subplots(figsize=(8.0, 5.5) if horizontal else (6.5, 4.2))
    if horizontal:
        plot_data = table.sort_values("records")
        axis.barh(plot_data[category], plot_data["records"], color="0.80", edgecolor="black")
        axis.set(xlabel="Records", ylabel=category.replace("_", " ").title(), title=title)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: fmt_int(value)))
        axis.grid(axis="x", linestyle=":", alpha=0.4)
    else:
        axis.bar(table[category], table["records"], color="0.75", edgecolor="black")
        axis.set(xlabel=category.replace("_", " ").title(), ylabel="Records", title=title)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: fmt_int(value)))
        axis.grid(axis="y", linestyle=":", alpha=0.4)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_priority_composition_plot(dataframe, category, title, filename):
    table = pd.crosstab(
        dataframe[category].fillna("<Missing>"),
        dataframe["priority"],
        normalize="index",
    ).reindex(columns=priority_order, fill_value=0).sort_values("high")

    figure, axis = plt.subplots(figsize=(8.0, max(4.2, 0.42 * len(table))))
    table.plot(
        kind="barh",
        stacked=True,
        color=["0.85", "0.60", "0.35"],
        edgecolor="black",
        linewidth=0.5,
        ax=axis,
    )
    axis.set(title=title, xlabel="Share within category", ylabel=category.replace("_", " ").title())
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="x", linestyle=":", alpha=0.4)
    axis.set_axisbelow(True)
    axis.legend(title="Priority", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


# %% 03 - Collect and describe the dataset

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)
records, variables = df.shape
source_sha256 = sha256_file(DATA_PATH)

required = ["subject", "body", "answer", "priority", "queue", "type", "language", "version"]
expected_columns = required + [f"tag_{index}" for index in range(1, 9)]
missing_required = sorted(set(expected_columns) - set(df.columns))
unexpected_columns = sorted(set(df.columns) - set(expected_columns))

if missing_required or unexpected_columns:
    raise KeyError(
        "Raw source schema differs from the validated dataset. "
        f"Missing: {missing_required}; unexpected: {unexpected_columns}."
    )
if records != EXPECTED_RAW_RECORDS or variables != EXPECTED_RAW_VARIABLES:
    raise ValueError(
        f"Unexpected raw dimensions: {records} x {variables}."
    )
if source_sha256 != EXPECTED_RAW_SHA256:
    raise ValueError("Raw source file differs from the validated dataset.")

tag_columns = [column for column in df.columns if re.fullmatch(r"tag_\d+", str(column))]
variable_roles = {
    "subject": "early_text_predictor",
    "body": "early_text_predictor",
    "priority": "target",
    "queue": "early_context_predictor",
    "type": "early_context_predictor",
    "language": "early_context_predictor",
    "version": "dataset_context_excluded",
    "answer": "post_ticket_information_excluded",
    **{column: "auxiliary_tag_excluded" for column in tag_columns},
}

variable_profile = pd.DataFrame({
    "variable": df.columns,
    "role": [variable_roles.get(column, "unclassified") for column in df.columns],
    "dtype": [str(df[column].dtype) for column in df.columns],
    "unique_values": [df[column].nunique(dropna=True) for column in df.columns],
    "missing_records": [df[column].isna().sum() for column in df.columns],
    "blank_string_records": [
        df[column].astype("string").str.strip().eq("").fillna(False).sum()
        for column in df.columns
    ],
})
variable_profile["missing_share"] = variable_profile["missing_records"] / records

save_table(variable_profile, "variable_profile.csv")

print("\nData understanding started")
print("--------------------------")
print(f"Dataset                   : {DATA_PATH.name}")
print(f"Records                   : {fmt_int(records)}")
print(f"Variables                 : {variables}")
print(f"File size                 : {DATA_PATH.stat().st_size / (1024 ** 2):.2f} MB")
print("Source identity           : validated")

print("\nVariable profile")
print("----------------")
print(variable_profile.to_string(
    index=False,
    formatters={
        "unique_values": fmt_int,
        "missing_records": fmt_int,
        "blank_string_records": fmt_int,
        "missing_share": lambda value: f"{value:.4f}",
    },
))


# %% 04 - Create temporary text views for data-quality analysis

text_series = {
    column: df[column].fillna("").astype(str).str.strip()
    for column in ["subject", "body", "answer"]
}
text_series["combined"] = (text_series["subject"] + " " + text_series["body"]).str.strip()

combined_text = text_series["combined"]
combined_words = count_words(combined_text)
combined_characters = combined_text.str.len()
normalized_text = normalize_comparison_text(combined_text)


# %% 05 - Explore priority target

canonical_priority_order = ["low", "medium", "high"]

observed_priorities = (
    df["priority"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

if {value.lower() for value in observed_priorities} == set(canonical_priority_order):
    casing_lookup = {value.lower(): value for value in observed_priorities}
    priority_order = [casing_lookup[label] for label in canonical_priority_order]
else:
    priority_order = sorted(observed_priorities)

invalid_priorities = (
    set(
        df["priority"]
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
    )
    - set(canonical_priority_order)
)

priority_distribution = distribution_table(
    df["priority"],
    "priority",
    priority_order,
)

class_counts = priority_distribution.loc[
    priority_distribution["records"].gt(0),
    "records",
]

print("\nPriority target")
print("---------------")
print_distribution(
    "Priority distribution",
    priority_distribution,
    "priority",
)

print(
    f"Majority-to-minority ratio : "
    f"{class_counts.max() / class_counts.min():.4f}"
)
print(
    f"Unexpected priority labels : "
    f"{fmt_int(len(invalid_priorities))}"
)

# %% 06 - Explore language variable

language_distribution = distribution_table(df["language"], "language")

invalid_languages = sorted(
    set(
        df["language"]
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
    )
    - {"de", "en"}
)

print("\nLanguage variable")
print("-----------------")
print_distribution("Language distribution", language_distribution, "language")
print(f"Missing language labels    : {fmt_int(df['language'].isna().sum())}")
print(f"Unexpected language values : {fmt_int(len(invalid_languages))}")


# %% 07 - Explore structured variables, tags, and queue breadth

queue_distribution = distribution_table(df["queue"], "queue")
type_distribution = distribution_table(df["type"], "type")
version_distribution = distribution_table(df["version"], "version")

categorical_distributions = pd.concat([
    priority_distribution.rename(columns={"priority": "category"}).assign(variable="priority"),
    queue_distribution.rename(columns={"queue": "category"}).assign(variable="queue"),
    type_distribution.rename(columns={"type": "category"}).assign(variable="type"),
    language_distribution.rename(columns={"language": "category"}).assign(variable="language"),
    version_distribution.rename(columns={"version": "category"}).assign(variable="version"),
], ignore_index=True)[["variable", "category", "records", "share"]]

save_table(categorical_distributions, "categorical_distributions.csv")

print("\nStructured variables")
print("--------------------")
print_distribution("Queue distribution", queue_distribution, "queue")
print_distribution("Ticket type distribution", type_distribution, "type")
print_distribution("Observed values of variable 'version'", version_distribution, "version")

tag_matrix = (
    df[tag_columns].astype("string")
    .apply(lambda column: column.str.strip())
    .replace("", pd.NA)
)
tag_values = tag_matrix.stack()
tags_per_record = tag_matrix.notna().sum(axis=1)
tag_frequency = tag_values.value_counts().rename_axis("tag").reset_index(name="frequency")

print("\nTag fields")
print("----------")
print(f"Tag columns                : {len(tag_columns)}")
print("Modeling decision          : excluded from predictive feature sets")


# %% 08 - Explore text characteristics and technical artifacts

text_summary_rows = []
for field, values in text_series.items():
    characters = values.str.len()
    words = count_words(values)
    text_summary_rows.append({
        "field": field,
        "empty_records": int(characters.eq(0).sum()),
        "empty_share": characters.eq(0).mean(),
        "median_chars": characters.median(),
        "p95_chars": characters.quantile(0.95),
        "max_chars": characters.max(),
        "median_words": words.median(),
        "p95_words": words.quantile(0.95),
        "max_words": words.max(),
    })
text_summary = pd.DataFrame(text_summary_rows)
save_table(text_summary, "text_summary.csv")

placeholder_pattern = (
    r"^(?:n/?a|none|null|nan|test|todo|placeholder|"
    r"no description|not available|[-_.]+)$"
)
stacktrace_pattern = r"traceback|exception|\bat\s+[\w.$]+\([^)]*:\d+\)"
placeholder_mask = combined_text.str.fullmatch(placeholder_pattern, case=False, na=False)

very_short_mask = (
    combined_characters.between(1, VERY_SHORT_MAX_CHARACTERS) & ~placeholder_mask
)
short_text_mask = (
    combined_characters.between(VERY_SHORT_MAX_CHARACTERS + 1, SHORT_TEXT_MAX_CHARACTERS)
    & ~placeholder_mask
)
suspected_missing_whitespace = (
    combined_words.le(MISSING_WHITESPACE_MAX_WORDS)
    & combined_characters.gt(SHORT_TEXT_MAX_CHARACTERS)
    & ~placeholder_mask
)

reviewed_uninformative_short = normalized_text.isin({
    "hello support team",
    "hallo support",
    "support benötigt",
})

text_quality = pd.DataFrame({
    "indicator": [
        "contains_url",
        "contains_html_or_markup",
        "placeholder_like_text",
        "very_short_text_up_to_20_characters",
        "short_text_21_to_40_characters",
        "suspected_missing_whitespace",
        "reviewed_uninformative_short_text",
        "contains_stacktrace_signal",
        "contains_literal_escape_sequence",
    ],
    "records": [
        combined_text.str.contains(r"https?://|www\.", case=False, regex=True, na=False).sum(),
        combined_text.str.contains(r"<[^>]+>|\{code\}|\{noformat\}", case=False, regex=True, na=False).sum(),
        placeholder_mask.sum(),
        very_short_mask.sum(),
        short_text_mask.sum(),
        suspected_missing_whitespace.sum(),
        reviewed_uninformative_short.sum(),
        combined_text.str.contains(stacktrace_pattern, case=False, regex=True, na=False).sum(),
        combined_text.str.contains(r"\\[nrt]", regex=True, na=False).sum(),
    ],
})
text_quality["share"] = text_quality["records"] / records
save_table(text_quality, "text_quality_indicators.csv")

short_text_screening = pd.DataFrame({
    "record_index": df.index,
    "priority": df["priority"],
    "language": df["language"],
    "type": df["type"],
    "queue": df["queue"],
    "subject": text_series["subject"],
    "body": text_series["body"],
    "combined_words": combined_words,
    "combined_characters": combined_characters,
    "suspected_missing_whitespace": suspected_missing_whitespace,
    "reviewed_uninformative": reviewed_uninformative_short,
    "combined_text": combined_text,
}).loc[very_short_mask | short_text_mask | suspected_missing_whitespace].copy()

short_text_screening["decision"] = np.select(
    [
        short_text_screening["reviewed_uninformative"],
        short_text_screening["suspected_missing_whitespace"],
    ],
    [
        "exclude_after_manual_content_review",
        "retain_missing_whitespace_artifact",
    ],
    default="retain",
)

short_text_screening["decision_reason"] = np.select(
    [
        short_text_screening["reviewed_uninformative"],
        short_text_screening["suspected_missing_whitespace"],
    ],
    [
        "no_issue_specific_content",
        "missing_whitespace_artifact_retained",
    ],
    default="issue_specific_content_retained",
)
short_text_screening = short_text_screening.sort_values(
    ["combined_characters", "record_index"]
).reset_index(drop=True)
save_table(short_text_screening, "short_text_screening_candidates.csv")

print("\nText characteristics")
print("--------------------")
print("Text basis: subject and body are analyzed separately and as one combined ticket text.")
print(text_summary.to_string(
    index=False,
    formatters={
        "empty_records": fmt_int,
        "empty_share": lambda value: f"{value:.4f}",
        "median_chars": lambda value: f"{value:.1f}",
        "p95_chars": lambda value: f"{value:.1f}",
        "max_chars": fmt_int,
        "median_words": lambda value: f"{value:.1f}",
        "p95_words": lambda value: f"{value:.1f}",
        "max_words": fmt_int,
    },
))

print("\nText quality indicators")
print("-----------------------")
print(text_quality.to_string(
    index=False,
    formatters={"records": fmt_int, "share": lambda value: f"{value:.4f}"},
))

print("\nShort-text screening result")
print("---------------------------")
print(f"Very short texts (≤{VERY_SHORT_MAX_CHARACTERS} characters)       : {fmt_int(very_short_mask.sum())}")
print(f"Short texts ({VERY_SHORT_MAX_CHARACTERS + 1}–{SHORT_TEXT_MAX_CHARACTERS} characters)          : {fmt_int(short_text_mask.sum())}")
print(f"Possible whitespace artifacts (> {SHORT_TEXT_MAX_CHARACTERS} chars, ≤{MISSING_WHITESPACE_MAX_WORDS} words): {fmt_int(suspected_missing_whitespace.sum())}")
print(f"Reviewed as uninformative                         : {fmt_int(reviewed_uninformative_short.sum())}")

angle_pattern = re.compile(r"<[a-zA-Z_]{1,20}>")
discovered_tokens = sorted({
    token.lower() for text in combined_text for token in angle_pattern.findall(text)
})
angle_bracket_rows = []
for token in discovered_tokens:
    escaped = re.escape(token)
    contains = combined_text.str.contains(escaped, case=False, regex=True, na=False)
    occurrences = combined_text.str.count(escaped, flags=re.IGNORECASE)
    angle_bracket_rows.append({
        "token": token,
        "records": int(contains.sum()),
        "share_of_records": contains.mean(),
        "total_occurrences": int(occurrences.sum()),
    })
angle_bracket_tokens = pd.DataFrame(angle_bracket_rows)

print("\nObserved angle-bracket tokens")
print("-----------------------------")
if angle_bracket_tokens.empty:
    print("No angle-bracket tokens were found.")
else:
    print(angle_bracket_tokens.sort_values("records", ascending=False).to_string(
        index=False,
        formatters={
            "records": fmt_int,
            "share_of_records": lambda value: f"{value:.4f}",
            "total_occurrences": fmt_int,
        },
    ))


# %% 09 - Verify duplicate texts and body-answer contamination

exact_text_duplicates = duplicate_groups(combined_text, "exact_text", df["priority"])
normalized_text_duplicates = duplicate_groups(
    normalized_text,
    "normalized_text",
    df["priority"],
)

body_normalized = normalize_comparison_text(df["body"])
answer_normalized = normalize_comparison_text(df["answer"])
exact_body_answer = body_normalized.ne("") & body_normalized.eq(answer_normalized)

answer_in_body = pd.Series([
    bool(answer and len(answer) >= 20 and answer in body)
    for body, answer in zip(body_normalized, answer_normalized)
], index=df.index)

body_in_answer = pd.Series([
    bool(body and len(body) >= 20 and body in answer)
    for body, answer in zip(body_normalized, answer_normalized)
], index=df.index)

body_quoted_in_answer = body_in_answer & ~exact_body_answer
answer_coverage = (
    answer_normalized.str.len()
    .div(body_normalized.str.len().replace(0, np.nan))
    .fillna(0.0)
)
likely_answer_leakage = (
    exact_body_answer
    | (answer_in_body & answer_coverage.ge(ANSWER_COVERAGE_THRESHOLD))
)
body_answer_screening_mask = exact_body_answer | answer_in_body

leakage_sensitivity_rows = []
for threshold in ANSWER_COVERAGE_SENSITIVITY:
    additional_cases = (
        answer_in_body
        & ~exact_body_answer
        & answer_coverage.ge(threshold)
    )
    threshold_leakage = exact_body_answer | additional_cases
    threshold_exclusion = (
        placeholder_mask
        | reviewed_uninformative_short
        | threshold_leakage
        | normalized_text.eq("")
    )
    leakage_sensitivity_rows.append({
        "answer_coverage_threshold": threshold,
        "primary_threshold": bool(np.isclose(
            threshold,
            ANSWER_COVERAGE_THRESHOLD,
        )),
        "exact_body_answer_matches": int(exact_body_answer.sum()),
        "additional_answer_in_body_cases": int(additional_cases.sum()),
        "total_likely_leakage_cases": int(threshold_leakage.sum()),
        "total_exclusion_records": int(threshold_exclusion.sum()),
        "expected_prepared_records": int(records - threshold_exclusion.sum()),
    })

leakage_threshold_sensitivity = pd.DataFrame(leakage_sensitivity_rows)
save_table(
    leakage_threshold_sensitivity,
    "body_answer_threshold_sensitivity.csv",
)


quality_checks = pd.DataFrame({
    "check": [
        "missing_priority_records",
        "records_without_subject_and_body",
        "exact_duplicate_full_rows",
        "exact_duplicate_text_groups",
        "duplicate_normalized_text_groups",
        "records_in_duplicate_text_groups",
        "conflicting_priority_text_groups",
        "exact_body_answer_matches",
        "answer_contained_in_body",
        "body_quoted_in_later_answer",
        "likely_answer_leakage_into_body",
        "unexpected_priority_labels",
        "unexpected_language_labels",
    ],
    "value": [
        df["priority"].isna().sum(),
        normalized_text.eq("").sum(),
        df.duplicated(keep=False).sum(),
        len(exact_text_duplicates),
        len(normalized_text_duplicates),
        normalized_text_duplicates["records"].sum(),
        normalized_text_duplicates["distinct_priorities"].gt(1).sum(),
        exact_body_answer.sum(),
        answer_in_body.sum(),
        body_quoted_in_answer.sum(),
        likely_answer_leakage.sum(),
        len(invalid_priorities),
        len(invalid_languages),
    ],
})
save_table(quality_checks, "data_quality_summary.csv")

console_quality_checks = quality_checks.loc[
    quality_checks["check"].isin([
        "missing_priority_records",
        "records_without_subject_and_body",
        "exact_duplicate_full_rows",
        "unexpected_priority_labels",
        "unexpected_language_labels",
    ])
]

print("\nBasic data-quality verification")
print("-------------------------------")
print(
    console_quality_checks.to_string(
        index=False,
        formatters={"value": fmt_int},
    )
)

print("\nDuplicate-text screening result")
print("-------------------------------")
print(f"Exact duplicate text groups      : {fmt_int(len(exact_text_duplicates))}")
print(f"Normalized duplicate text groups : {fmt_int(len(normalized_text_duplicates))}")
print(
    f"Conflicting priority groups       : "
    f"{fmt_int(normalized_text_duplicates['distinct_priorities'].gt(1).sum())}"
)

low_coverage_answer_overlaps = body_answer_screening_mask & ~likely_answer_leakage
print("\nBody-answer contamination screening")
print("-----------------------------------")
print(f"Exact body-answer matches                     : {fmt_int(exact_body_answer.sum())}")
print(
    f"High-coverage answer-in-body cases (≥{ANSWER_COVERAGE_THRESHOLD:.2f}) : "
    f"{fmt_int((likely_answer_leakage & ~exact_body_answer).sum())}"
)
print(f"Likely leakage cases                          : {fmt_int(likely_answer_leakage.sum())}")
print(f"Low-coverage overlap cases retained           : {fmt_int(low_coverage_answer_overlaps.sum())}")
print(f"Body quoted in later answer                   : {fmt_int(body_quoted_in_answer.sum())}")
print("Detailed candidates were saved; no full ticket texts are printed.")

print("\nBody-answer threshold sensitivity")
print("---------------------------------")
print(leakage_threshold_sensitivity.to_string(
    index=False,
    formatters={
        "answer_coverage_threshold": lambda value: f"{value:.2f}",
        "exact_body_answer_matches": fmt_int,
        "additional_answer_in_body_cases": fmt_int,
        "total_likely_leakage_cases": fmt_int,
        "total_exclusion_records": fmt_int,
        "expected_prepared_records": fmt_int,
    },
))


# %% 10 - Explore categorical relationships with priority

association_data = df.loc[df["priority"].isin(priority_order)]
association_results = []

for variable in ["queue", "type", "language", "version"]:
    table = pd.crosstab(
        association_data[variable].fillna("<Missing>"),
        association_data["priority"],
    ).reindex(columns=priority_order, fill_value=0)

    if variable == "queue":
        high_share = table.div(table.sum(axis=1), axis=0)[priority_order[-1]]
        table = table.loc[high_share.sort_values().index]

    chi2, p_value, dof, value, expected = cramers_v_bias_corrected(table)
    association_results.append({
        "variable": variable,
        "categories": table.shape[0],
        "chi2": chi2,
        "degrees_of_freedom": dof,
        "p_value": p_value,
        "cramers_v": value,
        "minimum_expected_count": expected.min(),
        "assumption_met": expected.min() >= 1 and (expected < 5).mean() <= 0.20,
    })

associations = pd.DataFrame(association_results).sort_values(
    "cramers_v", ascending=False
).reset_index(drop=True)
associations["p_value_holm"] = holm_adjusted_pvalues(
    associations["p_value"].to_numpy()
)
associations["significant_holm_0_05"] = associations["p_value_holm"].lt(0.05)
save_table(associations, "priority_associations.csv")

console_associations = associations[[
    "variable",
    "categories",
    "p_value",
    "p_value_holm",
    "cramers_v",
    "minimum_expected_count",
    "assumption_met",
]].copy()

for column in ["p_value", "p_value_holm"]:
    console_associations[column] = console_associations[column].map(
        lambda value: "<0.001" if value < 0.001 else f"{value:.3f}"
    )

console_associations["cramers_v"] = console_associations["cramers_v"].map(
    lambda value: f"{value:.4f}"
)
console_associations["minimum_expected_count"] = console_associations[
    "minimum_expected_count"
].map(lambda value: f"{value:.1f}")

print("\nCategorical relationships with priority")
print("---------------------------------------")
print("Raw and Holm-adjusted p-values are reported.")
print(console_associations.to_string(index=False))


# %% 11 - Define preliminary text-quality exclusions

preliminary_exclusion_mask = (
    placeholder_mask
    | reviewed_uninformative_short
    | likely_answer_leakage
    | normalized_text.eq("")
)


# %% 12 - Recalculate language audit, semantic embeddings, and review samples

audit_data = pd.DataFrame({
    "source_row_id": df.index + 1,
    "priority": df["priority"].astype(str).str.lower().str.strip(),
    "type": df["type"].fillna("<Missing>").astype(str).str.strip(),
    "queue": df["queue"].fillna("<Missing>").astype(str).str.strip(),
    "version": df["version"].fillna("<Missing>").astype(str).str.strip(),
    "declared_language": df["language"].astype(str).str.lower().str.strip(),
    "text": combined_text,
}).loc[~preliminary_exclusion_mask].reset_index(drop=True)

values = [
    detect_ticket_language(value)
    for value in audit_data["text"]
]
language_audit = pd.DataFrame({
    "source_row_id": audit_data["source_row_id"],
    "detected_language": [value[0] for value in values],
    "language_confidence": [value[1] for value in values],
})
language_audit.to_parquet(
    LANGUAGE_CACHE_PATH,
    index=False,
)
audit_data = audit_data.merge(
    language_audit,
    on="source_row_id",
    how="left",
    validate="one_to_one",
)

confident_language = audit_data["language_confidence"].ge(
    LANGUAGE_CONFIDENCE_THRESHOLD
)
high_confidence_non_target = audit_data["language_confidence"].ge(
    NON_TARGET_LANGUAGE_CONFIDENCE_THRESHOLD
)
audit_data["language_mismatch"] = (
    confident_language
    & audit_data["declared_language"].isin(TARGET_LANGUAGES)
    & audit_data["detected_language"].isin(TARGET_LANGUAGES)
    & audit_data["declared_language"].ne(
        audit_data["detected_language"]
    )
)
audit_data["non_target_language_candidate"] = (
    confident_language
    & ~audit_data["detected_language"].isin(TARGET_LANGUAGES)
)
audit_data["non_target_language"] = (
    high_confidence_non_target
    & ~audit_data["detected_language"].isin(TARGET_LANGUAGES)
)
audit_data["uncertain_language"] = ~confident_language
audit_data["detected_language_review"] = np.select(
    [
        audit_data["uncertain_language"],
        confident_language & ~audit_data["detected_language"].isin(TARGET_LANGUAGES),
    ],
    [
        "uncertain",
        "other",
    ],
    default=audit_data["detected_language"],
)

language_summary = (
    audit_data.assign(
        detected_language_report=np.where(
            confident_language,
            audit_data["detected_language_review"],
            "uncertain",
        )
    )
    .groupby(
        ["declared_language", "detected_language_report"],
        dropna=False,
    )
    .size()
    .rename("records")
    .reset_index()
)
language_summary["share"] = language_summary["records"] / len(audit_data)
save_table(language_summary, "language_label_audit_summary.csv")


source_rows = audit_data["source_row_id"].to_numpy(dtype=np.int64)

try:
    from sentence_transformers import SentenceTransformer
except (ImportError, OSError) as exc:
    raise RuntimeError(
        "sentence-transformers/PyTorch is required because Data Understanding "
        "recomputes semantic embeddings from scratch."
    ) from exc

model = SentenceTransformer(
    SEMANTIC_MODEL_ID,
    revision=SEMANTIC_MODEL_REVISION,
    device="cpu",
)
embeddings = model.encode(
    audit_data["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)

np.save(EMBEDDING_CACHE_PATH, embeddings)
np.save(EMBEDDING_IDS_PATH, source_rows)
del model

semantic_data = audit_data.reset_index(drop=True)
semantic_embeddings = embeddings

distances, indices = NearestNeighbors(
    n_neighbors=3,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1,
).fit(semantic_embeddings).kneighbors(semantic_embeddings)

nearest_rows = []
for row_index, (row_distances, row_indices) in enumerate(
    zip(distances, indices)
):
    position = next(
        i for i, neighbour in enumerate(row_indices)
        if neighbour != row_index
    )
    nearest_rows.append({
        "row_a": row_index,
        "row_b": int(row_indices[position]),
        "cosine_similarity": float(1.0 - row_distances[position]),
    })

nearest = pd.DataFrame(nearest_rows)
for suffix, index_column in [("a", "row_a"), ("b", "row_b")]:
    for column in [
        "source_row_id", "priority", "type", "queue", "version",
        "declared_language", "detected_language", "text",
    ]:
        nearest[f"{column}_{suffix}"] = nearest[index_column].map(
            semantic_data[column]
        )

for variable in ["priority", "type", "queue", "version"]:
    nearest[f"{variable}_conflict"] = nearest[f"{variable}_a"].ne(
        nearest[f"{variable}_b"]
    )
nearest["cross_language"] = nearest["detected_language_a"].ne(
    nearest["detected_language_b"]
)

semantic_rows = []
for threshold in SEMANTIC_THRESHOLDS:
    selected = nearest["cosine_similarity"].ge(threshold)
    semantic_rows.append({
        "similarity_threshold": threshold,
        "records": int(selected.sum()),
        "share": float(selected.mean()),
        "priority_conflicts": int(
            (selected & nearest["priority_conflict"]).sum()
        ),
        "cross_language_records": int(
            (selected & nearest["cross_language"]).sum()
        ),
        "type_conflicts": int(
            (selected & nearest["type_conflict"]).sum()
        ),
        "queue_conflicts": int(
            (selected & nearest["queue_conflict"]).sum()
        ),
        "version_conflicts": int(
            (selected & nearest["version_conflict"]).sum()
        ),
    })

semantic_summary = pd.DataFrame(semantic_rows)
save_table(
    semantic_summary,
    "semantic_redundancy_summary.csv",
)

print("\nLanguage-label consistency")
print("--------------------------")
print(
    f"Confident DE/EN mismatches : "
    f"{fmt_int(audit_data['language_mismatch'].sum())}"
)
print(
    f"Confident non-DE/EN texts  : "
    f"{fmt_int(audit_data['non_target_language'].sum())}"
)
print(
    f"Uncertain language cases   : "
    f"{fmt_int(audit_data['uncertain_language'].sum())}"
)

print("\nSemantic redundancy")
print("-------------------")
print(semantic_summary.to_string(
    index=False,
    formatters={
        "similarity_threshold": "{:.2f}".format,
        "records": fmt_int,
        "share": "{:.4f}".format,
        "priority_conflicts": fmt_int,
        "cross_language_records": fmt_int,
        "type_conflicts": fmt_int,
        "queue_conflicts": fmt_int,
        "version_conflicts": fmt_int,
    },
))


# %% 13 - Generate deterministic manual-review templates

# Semantic review: 20 unique nearest-neighbour pairs from each similarity band.
semantic_pairs = nearest.assign(
    pair_key=nearest.apply(
        lambda row: tuple(sorted((int(row["row_a"]), int(row["row_b"])))),
        axis=1,
    )
).drop_duplicates("pair_key")

semantic_review_parts = []
for lower, upper in [
    (0.90, 0.95),
    (0.95, 0.98),
    (0.98, 1.01),
]:
    candidates = semantic_pairs.loc[
        semantic_pairs["cosine_similarity"].ge(lower)
        & semantic_pairs["cosine_similarity"].lt(upper)
    ]
    if not candidates.empty:
        semantic_review_parts.append(
            candidates.sample(
                n=min(20, len(candidates)),
                random_state=RANDOM_STATE,
            )
        )

semantic_review_template = pd.concat(
    semantic_review_parts,
    ignore_index=True,
)[["cosine_similarity", "text_a", "text_b"]].copy()
semantic_review_template.insert(
    0,
    "pair_id",
    np.arange(1, len(semantic_review_template) + 1),
)

semantic_pair_sha256 = semantic_review_pair_fingerprint(
    semantic_review_template
)
if semantic_pair_sha256 != EXPECTED_SEMANTIC_REVIEW_PAIR_SHA256:
    raise ValueError(
        "The generated semantic review sample does not match the frozen "
        "60-pair sample that was manually reviewed. Do not create a new "
        "manual review. Check the semantic-review population/model setup."
    )

semantic_review_template["review_decision"] = ""
semantic_review_template["review_comment"] = ""
save_table(
    semantic_review_template,
    SEMANTIC_REVIEW_TEMPLATE_PATH.name,
)

# Language review: 20 confident mismatches, 20 confident matches, and
# 20 uncertain/non-target automatic detections.
audit_data["language_review_group"] = np.select(
    [
        audit_data["language_mismatch"],
        (
            confident_language
            & audit_data["declared_language"].isin(TARGET_LANGUAGES)
            & audit_data["detected_language"].isin(TARGET_LANGUAGES)
            & audit_data["declared_language"].eq(audit_data["detected_language"])
        ),
        audit_data["detected_language_review"].isin(["uncertain", "other"]),
    ],
    [
        "confident_mismatch",
        "confident_match",
        "uncertain_or_other",
    ],
    default="not_reviewed",
)

language_review_parts = []
for review_group in [
    "confident_mismatch",
    "confident_match",
    "uncertain_or_other",
]:
    candidates = audit_data.loc[
        audit_data["language_review_group"].eq(review_group)
    ]
    if not candidates.empty:
        language_review_parts.append(
            candidates.sample(
                n=min(20, len(candidates)),
                random_state=RANDOM_STATE,
            )
        )

language_review_template = pd.concat(
    language_review_parts,
    ignore_index=True,
)[
    [
        "language_review_group",
        "source_row_id",
        "version",
        "declared_language",
        "detected_language_review",
        "language_confidence",
        "text",
    ]
].rename(
    columns={"detected_language_review": "detected_language"}
).copy()
language_review_template.insert(
    0,
    "review_id",
    np.arange(1, len(language_review_template) + 1),
)
language_review_template["manual_language"] = ""
language_review_template["manual_assessment"] = ""
language_review_template["review_comment"] = ""
save_table(
    language_review_template,
    LANGUAGE_REVIEW_TEMPLATE_PATH.name,
)

# Priority review: five records for every priority x declared-language stratum.
priority_review_template = (
    audit_data.loc[
        audit_data["priority"].isin(["low", "medium", "high"])
        & audit_data["declared_language"].isin(TARGET_LANGUAGES)
    ]
    .groupby(
        ["priority", "declared_language"],
        group_keys=False,
    )
    .sample(
        n=5,
        random_state=RANDOM_STATE,
    )
    .reset_index(drop=True)
)
priority_review_template = priority_review_template[
    [
        "source_row_id",
        "priority",
        "declared_language",
        "version",
        "type",
        "queue",
        "text",
    ]
].copy()
priority_review_template.insert(
    0,
    "review_id",
    np.arange(1, len(priority_review_template) + 1),
)
priority_review_template["priority_assessment"] = ""
priority_review_template["missing_context_required"] = ""
priority_review_template["review_comment"] = ""
save_table(
    priority_review_template,
    PRIORITY_REVIEW_TEMPLATE_PATH.name,
)

review_counts = {
    "semantic": len(semantic_review_template),
    "language": len(language_review_template),
    "priority": len(priority_review_template),
}
if review_counts != EXPECTED_REVIEW_COUNTS:
    raise ValueError(
        "Generated manual-review sample sizes differ from the frozen design: "
        f"{review_counts}"
    )

print("\nManual-review templates")
print("-----------------------")
print(f"Semantic pairs : {fmt_int(len(semantic_review_template))}")
print(f"Language cases : {fmt_int(len(language_review_template))}")
print(f"Priority cases : {fmt_int(len(priority_review_template))}")

# The three templates are outputs of Data Understanding.
# They are completed manually outside this script and validated in script 01a.


# %% 14 - Create final figures


plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

save_distribution_plot(
    priority_distribution, "priority",
    "Ticket Priority Distribution", "priority_distribution.png"
)
save_priority_composition_plot(
    association_data, "queue",
    "Priority Composition within Support Queues", "priority_within_queue.png"
)

association_plot = associations.sort_values("cramers_v").copy()
association_plot["label"] = association_plot["variable"].str.replace(
    "_", " ", regex=False
).str.title()

figure, axis = plt.subplots(figsize=(7.5, 4.5))
axis.barh(
    association_plot["label"],
    association_plot["cramers_v"],
    color="0.70",
    edgecolor="black",
)
axis.set(
    title="Categorical Associations with Ticket Priority",
    xlabel="Bias-corrected Cramér's V",
    ylabel="Variable",
)
axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
axis.set_xlim(left=0)
axis.grid(axis="x", linestyle=":", alpha=0.4)
axis.set_axisbelow(True)
figure.tight_layout()
figure.savefig(
    FIGURES_DIR / "cramers_v_priority_associations.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)


manifest_payload = {
    "script": Path(__file__).name,
    "raw_data": DATA_PATH.name,
    "raw_sha256": source_sha256,
    "raw_records": records,
    "raw_variables": variables,
    "random_state": RANDOM_STATE,
    "language_detection": {
        "method": "langdetect",
        "confidence_threshold": LANGUAGE_CONFIDENCE_THRESHOLD,
        "non_target_confidence_threshold": NON_TARGET_LANGUAGE_CONFIDENCE_THRESHOLD,
        "artifact": str(LANGUAGE_CACHE_PATH.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(LANGUAGE_CACHE_PATH),
    },
    "semantic_embeddings": {
        "model_id": SEMANTIC_MODEL_ID,
        "revision": SEMANTIC_MODEL_REVISION,
        "normalized_embeddings": True,
        "artifact": str(EMBEDDING_CACHE_PATH.relative_to(PROJECT_ROOT)),
        "source_row_ids_artifact": str(EMBEDDING_IDS_PATH.relative_to(PROJECT_ROOT)),
        "embeddings_sha256": sha256_file(EMBEDDING_CACHE_PATH),
        "source_row_ids_sha256": sha256_file(EMBEDDING_IDS_PATH),
    },
    "review_templates": {
        "semantic": {
            "records": len(semantic_review_template),
            "path": str(SEMANTIC_REVIEW_TEMPLATE_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(SEMANTIC_REVIEW_TEMPLATE_PATH),
            "pair_identity_sha256": semantic_pair_sha256,
        },
        "language": {
            "records": len(language_review_template),
            "path": str(LANGUAGE_REVIEW_TEMPLATE_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(LANGUAGE_REVIEW_TEMPLATE_PATH),
        },
        "priority": {
            "records": len(priority_review_template),
            "path": str(PRIORITY_REVIEW_TEMPLATE_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(PRIORITY_REVIEW_TEMPLATE_PATH),
        },
    },
}
save_json(manifest_payload, MANIFEST_PATH)

print("\nData Understanding completed")
print("----------------------------")
print(f"Raw records               : {fmt_int(records)}")
print(f"Review templates          : {sum(EXPECTED_REVIEW_COUNTS.values())} cases across 3 files")
print(f"Manifest                  : {MANIFEST_PATH.name}")
