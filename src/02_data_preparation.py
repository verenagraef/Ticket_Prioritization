"""CRISP-DM Data Preparation for the multilingual IT support ticket dataset."""

# %% 00 - Load packages

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


# %% 01 - Configure data preparation

SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name.lower() != "src":
    raise RuntimeError("Place this script in <project>/src before running it.")
PROJECT_ROOT = SCRIPT_PATH.parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
PREPARED_PATH = PROJECT_ROOT / "data" / "processed" / "customer_it_support_prepared.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "02_tickets_data_preparation"
DU_TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "01_tickets_data_understanding"
REVIEW_DIR = PROJECT_ROOT / "reports" / "tables" / "01a_manual_review_validation"
DU_CACHE_DIR = PROJECT_ROOT / "cache" / "01_tickets_data_understanding"

DU_MANIFEST_PATH = DU_TABLE_DIR / "01_data_understanding_manifest.json"
REVIEW_MANIFEST_PATH = REVIEW_DIR / "01a_manual_review_manifest.json"
VALIDATED_LANGUAGE_PATH = REVIEW_DIR / "01a_validated_language_decisions.csv"
LANGUAGE_PATH = DU_CACHE_DIR / "language_detection_detailed.parquet"
EMBEDDING_PATH = DU_CACHE_DIR / "multilingual_embeddings.npy"
EMBEDDING_IDS_PATH = DU_CACHE_DIR / "multilingual_embedding_source_rows.npy"

for directory in [PREPARED_PATH.parent, TABLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

EXPECTED_RAW_SHA256 = "f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0"
EXPECTED_RAW_RECORDS = 28_587
EXPECTED_RAW_VARIABLES = 16
EXPECTED_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_EMBEDDING_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"

ANSWER_COVERAGE_THRESHOLD = 0.70
LANGUAGE_CONFIDENCE_THRESHOLD = 0.80
NON_TARGET_LANGUAGE_CONFIDENCE_THRESHOLD = 0.99
SEMANTIC_SIMILARITY_THRESHOLD = 0.95
SEMANTIC_DISTANCE_THRESHOLD = 1.0 - SEMANTIC_SIMILARITY_THRESHOLD
SEMANTIC_SIMILARITY_TOLERANCE = 1e-6
MINIMUM_ALLOWED_SEMANTIC_SIMILARITY = SEMANTIC_SIMILARITY_THRESHOLD - SEMANTIC_SIMILARITY_TOLERANCE
SEMANTIC_GROUPING_METHOD = "complete_linkage_within_radius_components"

TARGET_ORDER = ["low", "medium", "high"]
TARGET_CLASSES = set(TARGET_ORDER)
TARGET_LANGUAGES = {"de", "en"}
REQUIRED_RAW_COLUMNS = ["subject", "body", "answer", "type", "queue", "priority", "language", "version"]
EXPECTED_RAW_COLUMNS = REQUIRED_RAW_COLUMNS + [f"tag_{i}" for i in range(1, 9)]
FINAL_COLUMNS = [
    "record_id", "source_row_id", "text", "text_clean", "text_group_id", "text_group_size",
    "semantic_group_id", "semantic_group_size", "type", "queue", "language", "priority",
    "text_char_count", "text_word_count",
]

KNOWN_PLACEHOLDERS = ["<tel_num>", "<acc_num>", "<name>", "<email>"]
REPLACEMENT_TOKENS = ["<url>", "<email_address>"]
MISSING_CATEGORY_TOKEN = "<missing>"
MANUALLY_REVIEWED_UNINFORMATIVE_TEXTS = {
    "hello support team", "hallo support", "support benötigt",
}


# %% 02 - Define compact helper functions

LITERAL_ESCAPE_PATTERN = re.compile(r"\\[nrt]")
BR_PATTERN = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
ANGLE_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]{}]+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF]")
WORD_PATTERN = re.compile(r"(?u)\b\w+\b")
PLACEHOLDER_ONLY_PATTERN = re.compile(
    r"^(?:n/?a|none|null|nan|test|todo|placeholder|no description|not available|[-_.]+)$",
    flags=re.IGNORECASE,
)
PLACEHOLDER_PROTECTION = {
    placeholder: f"PLACEHOLDER_{i}_TOKEN"
    for i, placeholder in enumerate(sorted(KNOWN_PLACEHOLDERS), start=1)
}


def fmt_int(value):
    return f"{int(value):,}".replace(",", ".")


def fmt_share(value):
    return f"{float(value):.4f}"


def print_section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def print_table(title, dataframe):
    print_section(title)
    print("<empty>" if dataframe.empty else dataframe.to_string(index=False))


def sha256_file(path, block_size=1_048_576):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha1_text(value):
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_table(frame, filename):
    frame.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")


def clean_category(value, lowercase=False):
    if value is None or pd.isna(value):
        return pd.NA
    value = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()
    if not value:
        return pd.NA
    return value.lower() if lowercase else value


def normalize_comparison_text(series):
    return (
        series.fillna("").astype(str).str.lower()
        .str.replace(r"\\[nrt]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def normalize_ticket_text(value):
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = LITERAL_ESCAPE_PATTERN.sub(" ", text)
    for placeholder, token in PLACEHOLDER_PROTECTION.items():
        text = re.sub(re.escape(placeholder), f" {token} ", text, flags=re.IGNORECASE)
    text = BR_PATTERN.sub(" ", text)
    text = ANGLE_TAG_PATTERN.sub(" ", text)
    for placeholder, token in PLACEHOLDER_PROTECTION.items():
        text = text.replace(token, placeholder)
    text = URL_PATTERN.sub(" <url> ", text)
    text = EMAIL_PATTERN.sub(" <email_address> ", text)
    text = CONTROL_PATTERN.sub(" ", text)
    text = ZERO_WIDTH_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def text_artifact_counts(series):
    text = series.fillna("").astype(str)
    known = "|".join(re.escape(v) for v in KNOWN_PLACEHOLDERS)
    allowed = "|".join(re.escape(v) for v in KNOWN_PLACEHOLDERS + REPLACEMENT_TOKENS)
    without_allowed = text.str.replace(allowed, " ", case=False, regex=True)
    without_allowed_or_br = without_allowed.str.replace(BR_PATTERN, " ", regex=True)
    return {
        "literal_escape_sequences": int(text.str.contains(LITERAL_ESCAPE_PATTERN, na=False).sum()),
        "br_markup": int(text.str.contains(BR_PATTERN, na=False).sum()),
        "other_angle_markup": int(without_allowed_or_br.str.contains(ANGLE_TAG_PATTERN, na=False).sum()),
        "known_anonymization_placeholders": int(text.str.contains(known, case=False, regex=True, na=False).sum()),
        "unmasked_email_addresses": int(text.str.contains(EMAIL_PATTERN, na=False).sum()),
        "urls": int(text.str.contains(URL_PATTERN, na=False).sum()),
        "email_address_tokens": int(text.str.contains(re.escape("<email_address>"), case=False, regex=True, na=False).sum()),
        "url_tokens": int(text.str.contains(re.escape("<url>"), case=False, regex=True, na=False).sum()),
    }


def require_hash(path, expected, label):
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"{label} does not match the upstream manifest.")


def load_upstream_inputs():
    required = [
        RAW_PATH, DU_MANIFEST_PATH, REVIEW_MANIFEST_PATH, VALIDATED_LANGUAGE_PATH,
        LANGUAGE_PATH, EMBEDDING_PATH, EMBEDDING_IDS_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Data Preparation inputs:\n" + "\n".join(map(str, missing)))

    du_manifest = load_json(DU_MANIFEST_PATH)
    review_manifest = load_json(REVIEW_MANIFEST_PATH)
    raw_sha = sha256_file(RAW_PATH)

    if raw_sha != EXPECTED_RAW_SHA256 or du_manifest.get("raw_sha256") != raw_sha:
        raise ValueError("Raw dataset and 01 manifest are not aligned.")
    if review_manifest.get("data_understanding_manifest_sha256") != sha256_file(DU_MANIFEST_PATH):
        raise ValueError("01a was not validated against the current 01 manifest.")
    if not review_manifest.get("all_reviews_validated", False):
        raise ValueError("01a does not confirm successful manual-review validation.")

    language_meta = du_manifest.get("language_detection", {})
    embedding_meta = du_manifest.get("semantic_embeddings", {})
    if language_meta.get("confidence_threshold") != LANGUAGE_CONFIDENCE_THRESHOLD:
        raise ValueError("01 language confidence threshold differs from Data Preparation.")
    if language_meta.get("non_target_confidence_threshold") != NON_TARGET_LANGUAGE_CONFIDENCE_THRESHOLD:
        raise ValueError("01 non-target language threshold differs from Data Preparation.")
    if embedding_meta.get("model_id") != EXPECTED_EMBEDDING_MODEL:
        raise ValueError("01 embedding model differs from the frozen setup.")
    embedding_revision = embedding_meta.get("revision")
    if (
        embedding_revision is not None
        and embedding_revision != EXPECTED_EMBEDDING_MODEL_REVISION
    ):
        raise ValueError("01 embedding model revision differs from the frozen setup.")
    if not embedding_meta.get("normalized_embeddings", False):
        raise ValueError("01 manifest does not confirm normalized embeddings.")

    require_hash(LANGUAGE_PATH, language_meta.get("sha256"), "Language artifact")
    require_hash(EMBEDDING_PATH, embedding_meta.get("embeddings_sha256"), "Embedding artifact")
    require_hash(EMBEDDING_IDS_PATH, embedding_meta.get("source_row_ids_sha256"), "Embedding ID artifact")
    require_hash(
        VALIDATED_LANGUAGE_PATH,
        review_manifest.get("output_hashes", {}).get("validated_language_decisions"),
        "Validated language decisions",
    )

    semantic_share = float(review_manifest.get("semantic_equivalent_share", np.nan))
    priority_not_observable = int(review_manifest.get("priority_not_observable", -1))
    priority_implausible = int(review_manifest.get("priority_implausible", -1))

    manual_language = pd.read_csv(VALIDATED_LANGUAGE_PATH, encoding="utf-8-sig")
    required_columns = {"source_row_id", "manual_language"}
    language_review_count = int(
        review_manifest.get("review_counts", {}).get("language", -1)
    )
    if (
        not required_columns.issubset(manual_language.columns)
        or language_review_count < 1
        or len(manual_language) != language_review_count
    ):
        raise ValueError("Validated language decisions have an incompatible schema or size.")
    manual_language["source_row_id"] = pd.to_numeric(manual_language["source_row_id"], errors="raise").astype(np.int64)
    manual_language["manual_language"] = manual_language["manual_language"].astype(str).str.strip().str.lower()
    if manual_language["source_row_id"].duplicated().any():
        raise ValueError("Validated language decisions contain duplicate source rows.")

    return raw_sha, semantic_share, priority_not_observable, priority_implausible, manual_language


def load_language_artifact(audit_data):
    audit = pd.read_parquet(LANGUAGE_PATH)
    required = ["source_row_id", "detected_language", "language_confidence"]
    if not set(required).issubset(audit.columns):
        raise ValueError("Language artifact has an incompatible schema.")
    audit = audit[required].copy()
    audit["source_row_id"] = pd.to_numeric(audit["source_row_id"], errors="raise").astype(np.int64)
    audit["detected_language"] = audit["detected_language"].astype(str).str.strip().str.lower()
    audit["language_confidence"] = pd.to_numeric(audit["language_confidence"], errors="raise")
    expected_ids = audit_data["source_row_id"].to_numpy(dtype=np.int64)
    if len(audit) != len(audit_data) or not np.array_equal(audit["source_row_id"].to_numpy(), expected_ids):
        raise ValueError("Language artifact is not aligned with the current audit population.")
    if audit[["detected_language", "language_confidence"]].isna().any().any():
        raise ValueError("Language artifact contains missing values.")
    return audit


def load_embedding_artifact(audit_data):
    embeddings = np.load(EMBEDDING_PATH)
    source_ids = np.asarray(np.load(EMBEDDING_IDS_PATH), dtype=np.int64)
    expected_ids = audit_data["source_row_id"].to_numpy(dtype=np.int64)
    if embeddings.ndim != 2 or len(embeddings) != len(source_ids):
        raise ValueError("Embedding artifacts have incompatible dimensions.")
    if not np.array_equal(source_ids, expected_ids):
        raise ValueError("Embedding artifacts are not aligned with the current audit population.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
        raise ValueError("Embedding artifact is not normalized.")
    return embeddings.astype(np.float32, copy=False), source_ids


def create_semantic_groups(embeddings, source_row_ids):
    radius_model = NearestNeighbors(
        radius=SEMANTIC_DISTANCE_THRESHOLD + SEMANTIC_SIMILARITY_TOLERANCE,
        metric="cosine", algorithm="brute", n_jobs=-1,
    ).fit(embeddings)
    graph = radius_model.radius_neighbors_graph(embeddings, mode="connectivity")
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph = graph.maximum(graph.T).tocsr()
    component_count, component_labels = connected_components(graph, directed=False, return_labels=True)

    final_labels = np.full(len(source_row_ids), -1, dtype=np.int64)
    next_label = 0
    for component_label in range(component_count):
        positions = np.flatnonzero(component_labels == component_label)
        if len(positions) == 1:
            local_labels = np.zeros(1, dtype=np.int64)
        else:
            local_labels = AgglomerativeClustering(
                n_clusters=None, metric="cosine", linkage="complete",
                distance_threshold=SEMANTIC_DISTANCE_THRESHOLD, compute_full_tree=True,
            ).fit_predict(embeddings[positions])
        for local_label in sorted(np.unique(local_labels)):
            final_labels[positions[local_labels == local_label]] = next_label
            next_label += 1

    if (final_labels < 0).any():
        raise RuntimeError("Semantic grouping left records without a final group.")

    membership = pd.DataFrame({
        "source_row_id": np.asarray(source_row_ids, dtype=np.int64),
        "group_label": final_labels,
    })
    group_keys = membership.groupby("group_label", sort=True)["source_row_id"].apply(
        lambda values: "|".join(map(str, sorted(map(int, values))))
    )
    group_ids = group_keys.map(lambda value: sha1_text(f"semantic_complete_linkage|{value}"))
    membership["semantic_group_id"] = membership["group_label"].map(group_ids)
    membership["semantic_group_size"] = membership.groupby("group_label")["group_label"].transform("size").astype(np.int64)

    cohesion_rows = []
    for group_label, group in membership.groupby("group_label", sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        if len(positions) < 2:
            continue
        similarity = embeddings[positions] @ embeddings[positions].T
        values = similarity[np.triu_indices(len(positions), k=1)]
        cohesion_rows.append({
            "semantic_group_id": group["semantic_group_id"].iloc[0],
            "records": len(positions),
            "pair_count": len(values),
            "pair_similarity_min": float(values.min()),
            "pair_similarity_p05": float(np.quantile(values, 0.05)),
            "pair_similarity_median": float(np.median(values)),
            "pair_similarity_mean": float(values.mean()),
            "share_pairs_at_or_ab_threshold": float(np.mean(values >= MINIMUM_ALLOWED_SEMANTIC_SIMILARITY)),
        })
    cohesion = pd.DataFrame(cohesion_rows)

    minimum_similarity = 1.0 if cohesion.empty else float(cohesion["pair_similarity_min"].min())
    minimum_pair_share = 1.0 if cohesion.empty else float(cohesion["share_pairs_at_or_ab_threshold"].min())
    groups_below = 0 if cohesion.empty else int(cohesion["pair_similarity_min"].lt(MINIMUM_ALLOWED_SEMANTIC_SIMILARITY).sum())
    sizes = membership.groupby("semantic_group_id").size()
    summary = {
        "semantic_grouping_method": SEMANTIC_GROUPING_METHOD,
        "candidate_radius_components": int(component_count),
        "candidate_similarity_edges": int(graph.nnz // 2),
        "semantic_groups": int(len(sizes)),
        "multi_record_semantic_groups": int(sizes.gt(1).sum()),
        "records_in_semantic_groups": int(sizes[sizes.gt(1)].sum()),
        "largest_semantic_group": int(sizes.max()),
        "minimum_within_group_similarity": minimum_similarity,
        "minimum_pair_share_at_or_ab_threshold": minimum_pair_share,
        "groups_below_similarity_threshold": groups_below,
    }
    return membership.drop(columns="group_label"), summary, cohesion


# %% 03 - Load and validate upstream inputs

raw_sha, semantic_equivalent_share, priority_not_observable, priority_implausible, manual_language = load_upstream_inputs()
raw_data = pd.read_csv(RAW_PATH, low_memory=False)
raw_records, raw_variables = raw_data.shape

missing_columns = sorted(set(EXPECTED_RAW_COLUMNS) - set(raw_data.columns))
unexpected_columns = sorted(set(raw_data.columns) - set(EXPECTED_RAW_COLUMNS))
if missing_columns or unexpected_columns:
    raise ValueError(f"Unexpected raw schema. Missing: {missing_columns}; unexpected: {unexpected_columns}.")
if (raw_records, raw_variables) != (EXPECTED_RAW_RECORDS, EXPECTED_RAW_VARIABLES):
    raise ValueError(f"Unexpected raw dimensions: {raw_records} x {raw_variables}.")

priority_raw = raw_data["priority"].map(lambda value: clean_category(value, lowercase=True))
if priority_raw.isna().any() or set(priority_raw.unique()) != TARGET_CLASSES:
    raise ValueError("Priority labels differ from the frozen source.")

print_section("CRISP-DM Data Preparation started")
print(f"Dataset                   : {RAW_PATH.name}")
print(f"Raw records               : {fmt_int(raw_records)}")
print(f"Raw variables             : {raw_variables}")
print("Source identity           : validated")
print("01 artifacts              : validated")
print("01a manual reviews        : validated")
print(f"Semantic threshold        : {SEMANTIC_SIMILARITY_THRESHOLD:.2f}")
print(f"Prepared output           : {PREPARED_PATH.name}")


# %% 04 - Reproduce exclusions and corrected language labels

raw_text = {column: raw_data[column].fillna("").astype(str).str.strip() for column in ["subject", "body", "answer"]}
raw_combined_text = (raw_text["subject"] + " " + raw_text["body"]).str.strip()
comparison_text = normalize_comparison_text(raw_combined_text)
placeholder_only = raw_combined_text.str.fullmatch(PLACEHOLDER_ONLY_PATTERN, na=False)
empty_text = comparison_text.eq("")
reviewed_uninformative = comparison_text.isin(MANUALLY_REVIEWED_UNINFORMATIVE_TEXTS)

body_normalized = normalize_comparison_text(raw_data["body"])
answer_normalized = normalize_comparison_text(raw_data["answer"])
exact_body_answer = body_normalized.ne("") & body_normalized.eq(answer_normalized)
answer_in_body = pd.Series(
    [bool(answer and len(answer) >= 20 and answer in body) for body, answer in zip(body_normalized, answer_normalized)],
    index=raw_data.index,
)
answer_coverage = answer_normalized.str.len().div(body_normalized.str.len().replace(0, np.nan)).fillna(0.0)
likely_leakage = exact_body_answer | (answer_in_body & answer_coverage.ge(ANSWER_COVERAGE_THRESHOLD))
preliminary_exclusion = placeholder_only | empty_text | reviewed_uninformative | likely_leakage

# This population must match the one used by 01 for language detection and embeddings.
audit_data = pd.DataFrame({
    "source_row_id": raw_data.index + 1,
    "declared_language": raw_data["language"].map(lambda value: clean_category(value, lowercase=True)),
    "text": raw_combined_text,
}).loc[~preliminary_exclusion].reset_index(drop=True)

audit_data = audit_data.merge(load_language_artifact(audit_data), on="source_row_id", how="left", validate="one_to_one")
confident = audit_data["language_confidence"].ge(LANGUAGE_CONFIDENCE_THRESHOLD)
high_confidence_non_target = audit_data["language_confidence"].ge(NON_TARGET_LANGUAGE_CONFIDENCE_THRESHOLD)
audit_data["language_mismatch"] = (
    confident
    & audit_data["declared_language"].isin(TARGET_LANGUAGES)
    & audit_data["detected_language"].isin(TARGET_LANGUAGES)
    & audit_data["declared_language"].ne(audit_data["detected_language"])
)
audit_data["non_target_language_candidate"] = confident & ~audit_data["detected_language"].isin(TARGET_LANGUAGES)
audit_data["non_target_language"] = high_confidence_non_target & ~audit_data["detected_language"].isin(TARGET_LANGUAGES)
audit_data["uncertain_language"] = ~confident

audit_data["manual_language"] = audit_data["source_row_id"].map(
    manual_language.set_index("source_row_id")["manual_language"]
)
manual_target = audit_data["manual_language"].isin(TARGET_LANGUAGES)
manual_non_target = audit_data["manual_language"].notna() & ~audit_data["manual_language"].isin(TARGET_LANGUAGES | {"uncertain"})
audit_data["final_non_target_language"] = np.where(
    audit_data["manual_language"].notna(), manual_non_target, audit_data["non_target_language"]
).astype(bool)

audit_data["final_language"] = audit_data["declared_language"]
automatic_target = audit_data["manual_language"].isna() & confident & audit_data["detected_language"].isin(TARGET_LANGUAGES)
audit_data.loc[manual_target, "final_language"] = audit_data.loc[manual_target, "manual_language"]
audit_data.loc[automatic_target, "final_language"] = audit_data.loc[automatic_target, "detected_language"]
audit_data["language_label_corrected"] = (
    audit_data["final_language"].ne(audit_data["declared_language"])
    & ~audit_data["final_non_target_language"]
)

audit_data["language_decision_source"] = np.select(
    [
        manual_target,
        automatic_target,
        audit_data["final_non_target_language"],
        audit_data["uncertain_language"],
    ],
    [
        "manual_review",
        "automatic_detection",
        "excluded_non_target_language",
        "declared_label_fallback_uncertain_detection",
    ],
    default="declared_label_retained",
)

audit_data["language_scope_status"] = np.select(
    [manual_target, manual_non_target, audit_data["manual_language"].eq("uncertain"), audit_data["final_non_target_language"], audit_data["uncertain_language"], audit_data["language_mismatch"]],
    ["target_language_confirmed_manually", "non_target_language_confirmed_manually", "language_uncertain_after_manual_review", "non_target_language_detected_automatically", "language_detection_uncertain", "de_en_label_mismatch_retained"],
    default="target_language_retained",
)

non_target_rows = set(audit_data.loc[audit_data["final_non_target_language"], "source_row_id"].astype(int))
non_target_mask = pd.Series((raw_data.index + 1).isin(non_target_rows), index=raw_data.index)
exclusion_flags = pd.DataFrame({
    "exclude_no_usable_text": placeholder_only | empty_text,
    "exclude_uninformative_short": reviewed_uninformative,
    "exclude_body_answer_leakage": likely_leakage,
    "exclude_non_target_language": non_target_mask,
}, index=raw_data.index)
exclusion_flags["exclude_record"] = exclusion_flags.any(axis=1)
reason_labels = {
    "exclude_no_usable_text": "no_usable_text",
    "exclude_uninformative_short": "reviewed_uninformative_short_text",
    "exclude_body_answer_leakage": "likely_body_answer_leakage",
    "exclude_non_target_language": "conservative_language_scope_exclusion",
}
exclusion_flags["exclusion_reason"] = exclusion_flags.apply(
    lambda row: " | ".join(label for column, label in reason_labels.items() if bool(row[column])), axis=1
)

decision_counts = {
    "no_usable_text": int(exclusion_flags["exclude_no_usable_text"].sum()),
    "uninformative_short": int(exclusion_flags["exclude_uninformative_short"].sum()),
    "body_answer_leakage": int(exclusion_flags["exclude_body_answer_leakage"].sum()),
    "non_target_language": int(exclusion_flags["exclude_non_target_language"].sum()),
    "total_exclusions": int(exclusion_flags["exclude_record"].sum()),
    "prepared_records": int((~exclusion_flags["exclude_record"]).sum()),
    "language_mismatches": int(audit_data["language_mismatch"].sum()),
    "uncertain_languages": int(audit_data["uncertain_language"].sum()),
    "lower_confidence_non_target": int((audit_data["non_target_language_candidate"] & ~audit_data["non_target_language"]).sum()),
}
excluded_record_audit = pd.DataFrame({
    "source_row_id": raw_data.index + 1,
    "priority": priority_raw,
    "type": raw_data["type"],
    "queue": raw_data["queue"],
    "declared_language": raw_data["language"],
    "combined_characters": raw_combined_text.str.len(),
    "combined_words": raw_combined_text.str.findall(r"(?u)\b\w+\b").str.len(),
    "exact_body_answer": exact_body_answer,
    "answer_in_body": answer_in_body,
    "answer_coverage": answer_coverage,
    "exclusion_reason": exclusion_flags["exclusion_reason"],
}).loc[exclusion_flags["exclude_record"]].reset_index(drop=True)
excluded_record_audit = excluded_record_audit.merge(
    audit_data[["source_row_id", "detected_language", "language_confidence", "manual_language", "language_scope_status"]],
    on="source_row_id", how="left", validate="one_to_one",
)
save_table(excluded_record_audit, "excluded_record_audit.csv")

language_scope_summary = (
    audit_data["language_scope_status"]
    .value_counts()
    .rename_axis("language_scope_status")
    .reset_index(name="records")
)
language_scope_summary["share"] = language_scope_summary["records"] / len(audit_data)
save_table(language_scope_summary, "language_scope_summary.csv")

language_correction_audit = audit_data.loc[
    audit_data["language_label_corrected"]
    | audit_data["final_non_target_language"]
    | audit_data["uncertain_language"],
    [
        "source_row_id",
        "declared_language",
        "detected_language",
        "language_confidence",
        "manual_language",
        "final_language",
        "language_label_corrected",
        "final_non_target_language",
        "language_decision_source",
        "language_scope_status",
    ],
].sort_values("source_row_id")
save_table(language_correction_audit, "language_correction_audit.csv")

language_correction_summary = pd.DataFrame([
    ["corrected_de_en_labels", int(audit_data["language_label_corrected"].sum())],
    ["manual_target_language_assignments", int(manual_target.sum())],
    ["automatic_target_language_assignments", int(automatic_target.sum())],
    ["uncertain_detection_fallbacks", int((audit_data["uncertain_language"] & ~audit_data["final_non_target_language"]).sum())],
    ["non_target_language_exclusions", int(audit_data["final_non_target_language"].sum())],
], columns=["indicator", "records"])
save_table(language_correction_summary, "language_correction_summary.csv")

decision_console = pd.DataFrame(
    [
        ["No usable subject/body text", decision_counts["no_usable_text"]],
        ["Reviewed uninformative short text", decision_counts["uninformative_short"]],
        ["Likely body-answer leakage", decision_counts["body_answer_leakage"]],
        ["Conservative language-scope exclusions", decision_counts["non_target_language"]],
        ["Unique records excluded", decision_counts["total_exclusions"]],
    ],
    columns=["decision", "records"],
)
decision_console["share_of_raw"] = decision_console["records"] / raw_records
decision_console["records"] = decision_console["records"].map(fmt_int)
decision_console["share_of_raw"] = decision_console["share_of_raw"].map(fmt_share)
print_table("Applied Data Understanding decisions", decision_console)
print(f"Corrected DE/EN labels    : {fmt_int(audit_data['language_label_corrected'].sum())}")


# %% 05 - Construct normalized modeling variables

retained_index = raw_data.index[~exclusion_flags["exclude_record"]]
working = pd.DataFrame(index=retained_index)
working["source_row_id"] = retained_index + 1
working["priority"] = priority_raw.loc[retained_index]
working["type"] = raw_data.loc[retained_index, "type"].map(clean_category)
working["queue"] = raw_data.loc[retained_index, "queue"].map(clean_category)
working["language"] = working["source_row_id"].map(audit_data.set_index("source_row_id")["final_language"])
working["subject"] = raw_data.loc[retained_index, "subject"].map(normalize_ticket_text)
working["body"] = raw_data.loc[retained_index, "body"].map(normalize_ticket_text)
working["text"] = (working["subject"] + " " + working["body"]).str.replace(r"\s+", " ", regex=True).str.strip()
working["text_clean"] = working["text"].str.lower()
working["text_char_count"] = working["text_clean"].str.len().astype(np.int64)
working["text_word_count"] = working["text_clean"].map(lambda value: len(WORD_PATTERN.findall(value))).astype(np.int64)
for column in ["type", "queue", "language"]:
    working[column] = working[column].fillna(MISSING_CATEGORY_TOKEN)

if working["text_clean"].eq("").any():
    raise ValueError("Text normalization created empty modeling text.")
if set(working["language"].unique()) != TARGET_LANGUAGES:
    raise ValueError("Final modeling language must contain exactly de and en.")

working["text_group_id"] = working["text_clean"].map(sha1_text)
working["text_group_size"] = working.groupby("text_group_id")["text_group_id"].transform("size").astype(np.int64)


# %% 06 - Create complete-linkage semantic groups

audit_embeddings, embedding_source_ids = load_embedding_artifact(audit_data)
embedding_position = pd.Series(np.arange(len(embedding_source_ids), dtype=np.int64), index=embedding_source_ids)
positions = working["source_row_id"].map(embedding_position)
if positions.isna().any():
    raise ValueError("A retained source row is missing from the 01 embedding artifact.")
final_embeddings = audit_embeddings[positions.to_numpy(dtype=np.int64)]

semantic_membership, semantic_summary, semantic_cohesion = create_semantic_groups(
    final_embeddings, working["source_row_id"].to_numpy(dtype=np.int64)
)
if semantic_summary["groups_below_similarity_threshold"] != 0:
    raise ValueError("Complete-linkage grouping produced a group below the 0.95 threshold.")
save_table(semantic_cohesion, "semantic_group_cohesion.csv")

working = working.reset_index(drop=True).merge(
    semantic_membership, on="source_row_id", how="left", validate="one_to_one"
)
text_group_audit = working.groupby("text_group_id").agg(
    records=("source_row_id", "size"),
    distinct_priorities=("priority", "nunique"),
    semantic_groups=("semantic_group_id", "nunique"),
)
duplicate_text_groups = text_group_audit.loc[text_group_audit["records"].gt(1)]
if duplicate_text_groups["distinct_priorities"].gt(1).any():
    raise ValueError("Conflicting priorities occur within an exact text group.")
if text_group_audit["semantic_groups"].gt(1).any():
    raise ValueError("An exact text group was split across semantic groups.")

semantic_group_audit = working.groupby("semantic_group_id").agg(
    records=("source_row_id", "size"),
    distinct_priorities=("priority", "nunique"),
    priority_labels=("priority", lambda values: " | ".join(sorted(set(values)))),
    distinct_types=("type", "nunique"),
    distinct_queues=("queue", "nunique"),
    distinct_languages=("language", "nunique"),
)
semantic_conflict_groups = semantic_group_audit.loc[
    semantic_group_audit["distinct_priorities"].gt(1)
].reset_index()
semantic_priority_conflicts = len(semantic_conflict_groups)

if not semantic_conflict_groups.empty:
    save_table(
        semantic_conflict_groups,
        "semantic_priority_conflict_groups.csv",
    )

semantic_summary.update({
    "semantic_priority_conflict_groups": semantic_priority_conflicts,
    "semantic_similarity_threshold": SEMANTIC_SIMILARITY_THRESHOLD,
    "manual_equivalent_share_at_threshold": semantic_equivalent_share,
    "target_used_for_grouping": False,
})
save_table(
    pd.DataFrame([{"indicator": key, "value": value} for key, value in semantic_summary.items()]),
    "semantic_group_summary.csv",
)

working = working.sort_values("source_row_id").reset_index(drop=True)
working.insert(0, "record_id", np.arange(1, len(working) + 1, dtype=np.int64))
records_in_duplicate_groups = int(working["text_group_size"].gt(1).sum())

print_section("Group construction")
print(f"Exact duplicate groups retained : {fmt_int(len(duplicate_text_groups))}")
print(f"Records in exact groups         : {fmt_int(records_in_duplicate_groups)}")
print(f"Semantic groups                 : {fmt_int(semantic_summary['semantic_groups'])}")
print(
    f"Multi-record semantic groups    : "
    f"{fmt_int(semantic_summary['multi_record_semantic_groups'])}"
)
print(
    f"Records in semantic groups      : "
    f"{fmt_int(semantic_summary['records_in_semantic_groups'])}"
)
print(f"Largest semantic group          : {fmt_int(semantic_summary['largest_semantic_group'])}")
print(
    f"Minimum within-group similarity : "
    f"{semantic_summary['minimum_within_group_similarity']:.6f}"
)
print(
    f"Groups below threshold          : "
    f"{fmt_int(semantic_summary['groups_below_similarity_threshold'])}"
)
print(f"Priority-conflict groups        : {fmt_int(semantic_priority_conflicts)}")
print("Split control column            : semantic_group_id")


# %% 07 - Save compact preparation audits

artifacts_before = text_artifact_counts(raw_combined_text.loc[retained_index])
artifacts_after = text_artifact_counts(working["text"])
normalization_actions = {
    "literal_escape_sequences": "replace with whitespace",
    "br_markup": "remove markup token",
    "other_angle_markup": "remove markup syntax",
    "known_anonymization_placeholders": "retain",
    "unmasked_email_addresses": "replace with <email_address>",
    "urls": "replace with <url>",
    "email_address_tokens": "created replacement token",
    "url_tokens": "created replacement token",
}
normalization_audit = pd.DataFrame([
    {"indicator": key, "records_before": artifacts_before[key], "records_after": artifacts_after[key], "action": normalization_actions[key]}
    for key in artifacts_before
])
save_table(normalization_audit, "text_normalization_audit.csv")

priority_before = priority_raw.value_counts().reindex(TARGET_ORDER, fill_value=0)
priority_after = working["priority"].value_counts().reindex(TARGET_ORDER, fill_value=0)
priority_comparison = pd.DataFrame({
    "priority": TARGET_ORDER,
    "records_before": [int(priority_before[label]) for label in TARGET_ORDER],
    "share_before": [float(priority_before[label] / len(priority_raw)) for label in TARGET_ORDER],
    "records_after": [int(priority_after[label]) for label in TARGET_ORDER],
    "share_after": [float(priority_after[label] / len(working)) for label in TARGET_ORDER],
})
priority_comparison["removed_records"] = priority_comparison["records_before"] - priority_comparison["records_after"]
save_table(priority_comparison, "final_priority_distribution.csv")

normalization_console = normalization_audit.copy()
normalization_console["records_before"] = normalization_console["records_before"].map(fmt_int)
normalization_console["records_after"] = normalization_console["records_after"].map(fmt_int)
print_table("Text normalization audit", normalization_console)

priority_console = priority_comparison.copy()
for column in ["records_before", "records_after", "removed_records"]:
    priority_console[column] = priority_console[column].map(fmt_int)
for column in ["share_before", "share_after"]:
    priority_console[column] = priority_console[column].map(fmt_share)
print_table("Priority distribution before and after preparation", priority_console)


# %% 08 - Validate and export the prepared modeling dataset

prepared = working.loc[:, FINAL_COLUMNS].copy()
required_modeling_columns = ["type", "queue", "language", "priority"]
if prepared[required_modeling_columns].isna().any().any():
    raise ValueError("Prepared dataset contains missing modeling values.")
if prepared[required_modeling_columns].astype("string").apply(lambda column: column.str.strip().eq("")).any().any():
    raise ValueError("Prepared dataset contains empty modeling values.")
if prepared["record_id"].duplicated().any() or prepared["source_row_id"].duplicated().any():
    raise ValueError("Prepared dataset contains duplicate IDs.")
if prepared.groupby("text_group_id")["priority"].nunique().gt(1).any():
    raise ValueError("Conflicting priorities remain within exact text groups.")
if prepared.groupby("text_group_id")["semantic_group_id"].nunique().gt(1).any():
    raise ValueError("An exact text group is split across semantic groups.")
if set(prepared["source_row_id"]).intersection(excluded_record_audit["source_row_id"]):
    raise ValueError("An excluded source row remains in the prepared dataset.")

semantic_sizes = prepared.groupby("semantic_group_id").size()
observed_final = {
    "exact_text_groups": int(prepared["text_group_id"].nunique()),
    "duplicate_text_groups": int(len(duplicate_text_groups)),
    "records_in_duplicate_text_groups": records_in_duplicate_groups,
    "semantic_groups": int(prepared["semantic_group_id"].nunique()),
    "multi_record_semantic_groups": int(semantic_sizes.gt(1).sum()),
    "records_in_semantic_groups": int(semantic_sizes[semantic_sizes.gt(1)].sum()),
    "largest_semantic_group": int(semantic_sizes.max()),
    "semantic_priority_conflict_groups": semantic_priority_conflicts,
    "corrected_de_en_labels": int(audit_data["language_label_corrected"].sum()),
}
missing_required = prepared[required_modeling_columns].isna().any(axis=1)
empty_required = (
    prepared[required_modeling_columns]
    .astype("string")
    .apply(lambda column: column.str.strip().eq(""))
    .any(axis=1)
)

final_validation = pd.DataFrame(
    [
        ["Missing modeling texts", int(prepared["text_clean"].eq("").sum())],
        ["Missing priority labels", int(prepared["priority"].isna().sum())],
        [
            "Priority class deviations",
            int(len(set(prepared["priority"].unique()) ^ TARGET_CLASSES)),
        ],
        [
            "Conflicting exact text groups",
            int(prepared.groupby("text_group_id")["priority"].nunique().gt(1).sum()),
        ],
        [
            "Exact groups split across semantic groups",
            int(
                prepared.groupby("text_group_id")["semantic_group_id"]
                .nunique()
                .gt(1)
                .sum()
            ),
        ],
        [
            "Semantic groups below similarity threshold",
            int(semantic_summary["groups_below_similarity_threshold"]),
        ],
        ["Duplicate record IDs", int(prepared["record_id"].duplicated().sum())],
        ["Missing required values", int(missing_required.sum())],
        ["Empty required values", int(empty_required.sum())],
        [
            "Explicit missing category values",
            int(
                prepared[["type", "queue", "language"]]
                .eq(MISSING_CATEGORY_TOKEN)
                .sum()
                .sum()
            ),
        ],
    ],
    columns=["check", "value"],
)
print_table("Final dataset validation", final_validation)

summary_values = {
    "source_file": RAW_PATH.name,
    "source_sha256": raw_sha,
    "source_identity_validated": True,
    "raw_records": raw_records,
    "raw_variables": raw_variables,
    "records_without_usable_text_removed": decision_counts["no_usable_text"],
    "uninformative_short_records_removed": decision_counts["uninformative_short"],
    "body_answer_leakage_records_removed": decision_counts["body_answer_leakage"],
    "non_target_language_records_removed": decision_counts["non_target_language"],
    "unique_records_removed": decision_counts["total_exclusions"],
    "final_prepared_records": len(prepared),
    "final_variables": len(FINAL_COLUMNS),
    "language_label_mismatches_retained": decision_counts["language_mismatches"],
    "uncertain_language_cases_retained": decision_counts["uncertain_languages"],
    "lower_confidence_non_target_cases_retained": decision_counts["lower_confidence_non_target"],
    "corrected_de_en_labels": observed_final["corrected_de_en_labels"],
    "unique_text_groups": observed_final["exact_text_groups"],
    "duplicate_text_groups_retained": observed_final["duplicate_text_groups"],
    "records_in_duplicate_text_groups": observed_final["records_in_duplicate_text_groups"],
    "semantic_group_threshold": SEMANTIC_SIMILARITY_THRESHOLD,
    "semantic_grouping_method": SEMANTIC_GROUPING_METHOD,
    "semantic_candidate_radius_components": semantic_summary["candidate_radius_components"],
    "semantic_candidate_similarity_edges": semantic_summary["candidate_similarity_edges"],
    "semantic_groups": observed_final["semantic_groups"],
    "multi_record_semantic_groups": observed_final["multi_record_semantic_groups"],
    "records_in_semantic_groups": observed_final["records_in_semantic_groups"],
    "largest_semantic_group": observed_final["largest_semantic_group"],
    "minimum_within_group_similarity": semantic_summary["minimum_within_group_similarity"],
    "minimum_pair_share_at_or_ab_threshold": semantic_summary["minimum_pair_share_at_or_ab_threshold"],
    "semantic_groups_below_similarity_threshold": semantic_summary["groups_below_similarity_threshold"],
    "semantic_priority_conflict_groups": observed_final["semantic_priority_conflict_groups"],
    "semantic_grouping_target_blind": True,
    "manual_semantic_equivalent_share": semantic_equivalent_share,
    "priority_not_observable_in_review": priority_not_observable,
    "priority_implausible_in_review": priority_implausible,
    "unique_ticket_types": int(prepared["type"].nunique()),
    "unique_queues": int(prepared["queue"].nunique()),
    "primary_modeling_features": "text_clean, type, language, queue",
    "split_control_column": "semantic_group_id",
    "exact_duplicate_control_column": "text_group_id",
    "excluded_raw_columns": "answer, version, tag_1 to tag_8; raw declared language replaced by corrected language",
    "deferred_to_modeling": "split, TF-IDF, stopwords, one-hot encoding, resampling, feature selection, tuning",
}
summary = pd.DataFrame([{"indicator": key, "value": value} for key, value in summary_values.items()])
save_table(summary, "data_preparation_summary.csv")

summary_indicators = [
    "raw_records",
    "unique_records_removed",
    "final_prepared_records",
    "corrected_de_en_labels",
    "unique_text_groups",
    "duplicate_text_groups_retained",
    "records_in_duplicate_text_groups",
    "semantic_groups",
    "multi_record_semantic_groups",
    "records_in_semantic_groups",
    "semantic_priority_conflict_groups",
    "largest_semantic_group",
    "semantic_groups_below_similarity_threshold",
    "unique_ticket_types",
    "unique_queues",
]
summary_console = summary.loc[summary["indicator"].isin(summary_indicators)].copy()
summary_console["value"] = summary_console["value"].map(
    lambda value: fmt_int(value) if isinstance(value, (int, np.integer)) else value
)
print_table("Prepared dataset summary", summary_console)

prepared.to_csv(PREPARED_PATH, index=False, encoding="utf-8-sig", lineterminator="\n")

print_section("Prepared dataset exported")
print(f"Prepared records          : {fmt_int(len(prepared))}")
print(f"Prepared variables        : {len(FINAL_COLUMNS)}")
print(f"Exact text groups         : {fmt_int(observed_final['exact_text_groups'])}")
print(f"Semantic groups           : {fmt_int(observed_final['semantic_groups'])}")
print("Primary predictors        : text_clean, type, language, queue")
print("Split control             : semantic_group_id")
print(f"Grouping method           : {SEMANTIC_GROUPING_METHOD}")
print(
    f"Minimum group similarity : "
    f"{semantic_summary['minimum_within_group_similarity']:.6f}"
)
print(f"Corrected language labels : {fmt_int(observed_final['corrected_de_en_labels'])}")
print("Language feature          : retained after correction")
