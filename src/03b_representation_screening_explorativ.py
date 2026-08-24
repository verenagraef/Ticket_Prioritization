"""03b - Training-only Word vs Character TF-IDF representation screening."""

# %% 00 - Packages

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import hashlib
import json

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

# %% 01 - Configuration

SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name.lower() != "src":
    raise RuntimeError("Place this file in <project>/src before running it.")

PROJECT_ROOT = SCRIPT_PATH.parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_it_support_prepared.csv"
POOL_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "03a_algorithm_screening"
    / "03a_algorithm_screening_pool.csv"
)
SPLIT_PATH = (
    PROJECT_ROOT / "cache" / "03_tickets_modeling"
    / "03_split_assignments.parquet"
)

TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "03b_representation_screening"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "03b_representation_screening"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BUILD = "2026-08-11_word_char_representation_screening_v1"
RANDOM_STATE = 42

TARGET = "priority"
GROUP = "semantic_group_id"
EXACT_GROUP = "text_group_id"
TEXT = "text_clean"
CLASS_ORDER = ["low", "medium", "high"]
CLASS_TO_INT = {label: i for i, label in enumerate(CLASS_ORDER)}

WORD_MAX_FEATURES = 30_000
CHAR_MAX_FEATURES = 30_000
SELECT_K = 15_000
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)

MODELS = {
    "logistic_regression": LogisticRegression(
        C=1.0, solver="saga", max_iter=3000, random_state=RANDOM_STATE
    ),
    "linear_svm": LinearSVC(
        C=1.0, max_iter=10000, random_state=RANDOM_STATE
    ),
    "complement_naive_bayes": ComplementNB(alpha=0.5),
}

OUT_RESULTS = TABLE_DIR / "03b_representation_screening_results.csv"
OUT_FOLDS = TABLE_DIR / "03b_representation_screening_fold_results.csv"
OUT_CLASSES = TABLE_DIR / "03b_representation_screening_class_results.csv"
OUT_MANIFEST = TABLE_DIR / "03b_representation_screening_manifest.json"
OUT_FIGURE = FIGURE_DIR / "03b_representation_macro_f1.png"


# %% 02 - Helpers

def section(title):
    print(f"\n## {title}\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stop_words():
    try:
        return sorted(set(stopwords.words("english")) | set(stopwords.words("german")))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return sorted(set(stopwords.words("english")) | set(stopwords.words("german")))


def calc_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(3), zero_division=0
    )
    overall = {
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
    }
    classes = pd.DataFrame({
        "priority": CLASS_ORDER,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    return overall, classes


def word_tfidf(stops):
    return TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=WORD_NGRAM_RANGE,
        min_df=2,
        max_features=WORD_MAX_FEATURES,
        stop_words=stops,
        sublinear_tf=True,
        dtype=np.float32,
    )


def char_tfidf():
    return TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=2,
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )


def representation(name, stops):
    if name == "word":
        return word_tfidf(stops)
    if name == "char":
        return char_tfidf()
    if name == "word_char":
        return FeatureUnion([
            ("word", word_tfidf(stops)),
            ("char", char_tfidf()),
        ])
    raise KeyError(name)


def pipeline(name, estimator, stops):
    return Pipeline([
        ("representation", representation(name, stops)),
        ("selector", SelectKBest(chi2, k=SELECT_K)),
        ("classifier", clone(estimator)),
    ])


# %% 03 - Reconstruct exact 03a screening sample

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Prepared data not found: {DATA_PATH}")
if not POOL_PATH.exists():
    raise FileNotFoundError(
        "03a screening pool missing. Run "
        "03a_algorithm_screening_explorativ.py first."
    )
if not SPLIT_PATH.exists():
    raise FileNotFoundError(
        "Frozen split assignments are missing. Run "
        "03a_algorithm_screening_explorativ.py first; "
        "03a creates the deterministic split when needed."
    )

data = pd.read_csv(DATA_PATH, low_memory=False)
data["modeling_row_id"] = np.arange(len(data), dtype=np.int64)
data[TARGET] = data[TARGET].astype(str).str.lower().str.strip()
data[TEXT] = data[TEXT].fillna("").astype(str)
data[GROUP] = data[GROUP].fillna("").astype(str)
data[EXACT_GROUP] = data[EXACT_GROUP].fillna("").astype(str)
data["priority_encoded"] = data[TARGET].map(CLASS_TO_INT).astype(np.int8)

split = (
    pd.read_parquet(SPLIT_PATH)
    .sort_values("modeling_row_id")
    .reset_index(drop=True)
)
data = data.sort_values("modeling_row_id").reset_index(drop=True)

required_split_columns = {
    "modeling_row_id",
    GROUP,
    EXACT_GROUP,
    TARGET,
    "split",
}
if not required_split_columns.issubset(split.columns):
    raise ValueError("Frozen split artifact has an incompatible schema.")
if len(split) != len(data):
    raise ValueError("Frozen split assignments do not match the prepared data.")
if split["modeling_row_id"].duplicated().any():
    raise ValueError("Frozen split artifact contains duplicate modeling_row_id values.")
if not np.array_equal(
    split["modeling_row_id"].to_numpy(dtype=np.int64),
    data["modeling_row_id"].to_numpy(dtype=np.int64),
):
    raise ValueError("Frozen split row order does not match the prepared data.")

for column in [GROUP, EXACT_GROUP, TARGET]:
    if not np.array_equal(
        split[column].astype(str).to_numpy(),
        data[column].astype(str).to_numpy(),
    ):
        raise ValueError(
            f"Frozen split artifact disagrees with prepared data on {column}."
        )

if set(split["split"]) != {"training", "holdout"}:
    raise ValueError("Unexpected frozen split labels.")

data["split"] = split["split"].to_numpy()

pool = pd.read_csv(POOL_PATH)
required_pool_columns = {
    "modeling_row_id",
    GROUP,
    EXACT_GROUP,
    TARGET,
    "screen_outer_fold",
    "screen_cv_fold",
}
if not required_pool_columns.issubset(pool.columns):
    raise ValueError("03a screening pool has an incompatible schema.")
if pool["modeling_row_id"].duplicated().any():
    raise ValueError("03a screening pool contains duplicate modeling_row_id values.")

screen = pool[
    ["modeling_row_id", "screen_cv_fold"]
].merge(
    data[[
        "modeling_row_id",
        TARGET,
        "priority_encoded",
        TEXT,
        GROUP,
        EXACT_GROUP,
        "split",
    ]],
    on="modeling_row_id",
    how="left",
    validate="one_to_one",
)

if screen[[TARGET, TEXT, GROUP, EXACT_GROUP, "split"]].isna().any().any():
    raise ValueError("03a screening pool contains IDs not found in prepared data.")

if not screen["split"].eq("training").all():
    raise ValueError("A non-training record entered 03b.")
if set(screen["screen_cv_fold"].unique()) != {0, 1}:
    raise ValueError("Expected the same two 03a screening folds.")

pool_check = pool[
    ["modeling_row_id", TARGET, GROUP, EXACT_GROUP]
].merge(
    data[["modeling_row_id", TARGET, GROUP, EXACT_GROUP]],
    on="modeling_row_id",
    how="left",
    validate="one_to_one",
    suffixes=("_pool", "_data"),
)
for column in [TARGET, GROUP, EXACT_GROUP]:
    if not pool_check[f"{column}_pool"].astype(str).equals(
        pool_check[f"{column}_data"].astype(str)
    ):
        raise ValueError(
            f"03a screening pool disagrees with prepared data on {column}."
        )

for fold in [0, 1]:
    tr = screen.loc[screen["screen_cv_fold"].ne(fold)]
    va = screen.loc[screen["screen_cv_fold"].eq(fold)]
    if set(tr[GROUP]) & set(va[GROUP]):
        raise ValueError(f"Semantic-group overlap in fold {fold}.")
    if set(tr[EXACT_GROUP]) & set(va[EXACT_GROUP]):
        raise ValueError(f"Exact-text overlap in fold {fold}.")

section("Representation screening population")
print(f"Screening records         : {len(screen):,}".replace(",", "."))
print(f"Semantic groups           : {screen[GROUP].nunique():,}".replace(",", "."))
print("Shared CV folds           : 2")
print("Changed factor            : text representation only")


# %% 04 - Run representation ablation

REPRESENTATIONS = {
    "word": "Word TF-IDF (1-2 grams)",
    "char": "Character TF-IDF (char_wb 3-5 grams)",
    "word_char": "Word + Character TF-IDF",
}

stops = stop_words()
y_all = screen["priority_encoded"].to_numpy(np.int8)
fold_array = screen["screen_cv_fold"].to_numpy(np.int8)

splits = [
    (
        np.flatnonzero(fold_array != validation_fold),
        np.flatnonzero(fold_array == validation_fold),
    )
    for validation_fold in [0, 1]
]

summary_rows = []
fold_rows = []
class_rows = []

for model_name, estimator in MODELS.items():
    for repr_name, repr_label in REPRESENTATIONS.items():

        local = []
        fit_times = []

        for fold, (train_idx, valid_idx) in enumerate(splits):
            model = pipeline(repr_name, estimator, stops)

            start = perf_counter()
            model.fit(
                screen.iloc[train_idx][TEXT],
                y_all[train_idx],
            )
            fit_seconds = perf_counter() - start

            start = perf_counter()
            pred = np.asarray(
                model.predict(screen.iloc[valid_idx][TEXT]),
                dtype=np.int8,
            )
            predict_seconds = perf_counter() - start

            overall, classes = calc_metrics(y_all[valid_idx], pred)
            local.append(overall)
            fit_times.append(fit_seconds)

            fold_rows.append({
                "model": model_name,
                "representation": repr_name,
                "fold": fold,
                **overall,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            })

            for row in classes.itertuples(index=False):
                class_rows.append({
                    "model": model_name,
                    "representation": repr_name,
                    "fold": fold,
                    "priority": row.priority,
                    "precision": float(row.precision),
                    "recall": float(row.recall),
                    "f1": float(row.f1),
                    "support": int(row.support),
                })


        frame = pd.DataFrame(local)
        summary_rows.append({
            "model": model_name,
            "representation": repr_name,
            "representation_label": repr_label,
            "mean_macro_f1": frame["macro_f1"].mean(),
            "fold_sd_macro_f1": frame["macro_f1"].std(ddof=0),
            "mean_weighted_f1": frame["weighted_f1"].mean(),
            "mean_accuracy": frame["accuracy"].mean(),
            "mean_macro_precision": frame["macro_precision"].mean(),
            "mean_macro_recall": frame["macro_recall"].mean(),
            "mean_fit_seconds": float(np.mean(fit_times)),
        })


# %% 05 - Compare with Word-TFIDF baseline

results = pd.DataFrame(summary_rows)
fold_results = pd.DataFrame(fold_rows)
class_results = pd.DataFrame(class_rows)

word_baseline = (
    results.loc[results["representation"].eq("word"), ["model", "mean_macro_f1"]]
    .rename(columns={"mean_macro_f1": "word_baseline_macro_f1"})
)

results = results.merge(word_baseline, on="model", how="left", validate="many_to_one")
results["delta_macro_f1_vs_word"] = (
    results["mean_macro_f1"] - results["word_baseline_macro_f1"]
)

results["rank_within_model"] = (
    results.groupby("model")["mean_macro_f1"]
    .rank(method="min", ascending=False)
    .astype(int)
)

results = results.sort_values(
    ["model", "rank_within_model", "mean_macro_f1"],
    ascending=[True, True, False],
).reset_index(drop=True)

results.to_csv(OUT_RESULTS, index=False, encoding="utf-8-sig")
fold_results.to_csv(OUT_FOLDS, index=False, encoding="utf-8-sig")
class_results.to_csv(OUT_CLASSES, index=False, encoding="utf-8-sig")

section("Representation screening results")
print(
    results[[
        "model", "representation", "mean_macro_f1",
        "fold_sd_macro_f1", "delta_macro_f1_vs_word", "mean_accuracy",
    ]].to_string(
        index=False,
        formatters={
            "mean_macro_f1": "{:.4f}".format,
            "fold_sd_macro_f1": "{:.4f}".format,
            "delta_macro_f1_vs_word": "{:+.4f}".format,
            "mean_accuracy": "{:.4f}".format,
        },
    )
)


# %% 06 - Save figure and manifest

plot = results.copy()
models = list(MODELS)
reprs = list(REPRESENTATIONS)
x = np.arange(len(models))
width = 0.24

fig, ax = plt.subplots(figsize=(7.2, 4.2))

fill_map = {
    "word": "white",
    "char": "0.65",
    "word_char": "0.30",
}
legend_labels = {
    "word": "Word TF-IDF",
    "char": "Character TF-IDF",
    "word_char": "Word + Character TF-IDF",
}

for index, repr_name in enumerate(reprs):
    values, errors = [], []
    for model_name in models:
        row = plot.loc[
            plot["model"].eq(model_name)
            & plot["representation"].eq(repr_name)
        ].iloc[0]
        values.append(row["mean_macro_f1"])
        errors.append(row["fold_sd_macro_f1"])

    offset = (index - 1) * width
    ax.bar(
        x + offset,
        values,
        width,
        yerr=errors,
        capsize=2,
        color=fill_map[repr_name],
        edgecolor="black",
        linewidth=0.8,
        ecolor="black",
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
        label=legend_labels[repr_name],
    )

ax.set_xticks(
    x,
    labels=["Logistic\nRegression", "Linear\nSVM", "Complement\nNaive Bayes"],
)
ax.set_ylabel("Mean screening CV Macro-F1")
ax.set_xlabel("Classifier")
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
ax.set_axisbelow(True)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    frameon=False,
    fontsize=8,
    ncol=3,
    handlelength=1.8,
    columnspacing=1.6,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.subplots_adjust(bottom=0.28)
fig.savefig(OUT_FIGURE, dpi=300, bbox_inches="tight")
plt.close(fig)

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "script": SCRIPT_PATH.name,
    "build": BUILD,
    "purpose": "training-only text representation ablation",
    "data_sha256": sha256_file(DATA_PATH),
    "split_path": str(SPLIT_PATH.relative_to(PROJECT_ROOT)),
    "split_sha256": sha256_file(SPLIT_PATH),
    "source_03a_pool_sha256": sha256_file(POOL_PATH),
    "screening_records": len(screen),
    "semantic_groups": int(screen[GROUP].nunique()),
    "holdout_scored": False,
    "shared_cv_folds": 2,
    "models_and_frozen_03a_parameters": {
        "logistic_regression": {"C": 1.0},
        "linear_svm": {"C": 1.0},
        "complement_naive_bayes": {"alpha": 0.5},
    },
    "representations": {
        "word": {"analyzer": "word", "ngram_range": WORD_NGRAM_RANGE},
        "char": {"analyzer": "char_wb", "ngram_range": CHAR_NGRAM_RANGE},
        "word_char": {"components": ["word", "char"]},
    },
    "common_selected_feature_budget": SELECT_K,
    "primary_metric": "macro_f1",
}

OUT_MANIFEST.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8",
)

section("Representation screening completed")
print(f"Results                   : {OUT_RESULTS.name}")
print(f"Fold results              : {OUT_FOLDS.name}")
print(f"Class results             : {OUT_CLASSES.name}")
print(f"Figure                    : {OUT_FIGURE.name}")
print(f"Manifest                  : {OUT_MANIFEST.name}")
