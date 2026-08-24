"""Final deterministic integrity audit for the CV-selected common feature-set analysis."""

# %% 00 - Load packages

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# %% 01 - Configure final audit

SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name.lower() != "src":
    raise RuntimeError(
        "Place 06_integrity_audit.py in the existing <project>/src folder."
    )

PROJECT_ROOT = SCRIPT_PATH.parents[1]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)
PREPARED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_it_support_prepared.csv"
)

M_TABLE = PROJECT_ROOT / "reports" / "tables" / "03_tickets_modeling"
M_MODEL = PROJECT_ROOT / "models" / "03_tickets_modeling"
M_CACHE = PROJECT_ROOT / "cache" / "03_tickets_modeling"

E_TABLE = PROJECT_ROOT / "reports" / "tables" / "04_evaluation"
E_FIG = PROJECT_ROOT / "reports" / "figures" / "04_evaluation"
E_CACHE = PROJECT_ROOT / "cache" / "04_evaluation"

X_TABLE = PROJECT_ROOT / "reports" / "tables" / "05_explainability"
X_FIG = PROJECT_ROOT / "reports" / "figures" / "05_explainability"

OUT_DIR = PROJECT_ROOT / "reports" / "tables" / "06_final_integrity_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_PATH = OUT_DIR / "06_integrity_audit.csv"
MANIFEST_PATH = OUT_DIR / "06_integrity_manifest.json"

SCRIPT_BUILD = "2026-08-17_final_integrity_audit_common_feature_selection_v4"

EXPECTED_BUILDS = {
    "modeling": "2026-08-16_final_comparative_common_feature_selection_v1",
    "evaluation": "2026-08-16_evaluation_common_feature_selection_v1",
    "explainability": "2026-08-16_explainability_common_feature_selection_v1",
}
EXPECTED_RUN_IDS = {
    "modeling": "final_comparative_common_feature_selection_v1",
    "evaluation": "evaluation_common_feature_selection_v1",
    "explainability": "explainability_common_feature_selection_v1",
}

RAW_SHA256 = (
    "f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0"
)

TARGET = "priority"
SEMANTIC_GROUP = "semantic_group_id"
EXACT_GROUP = "text_group_id"

MODEL_ORDER = [
    "logistic_regression",
    "linear_svm",
    "complement_naive_bayes",
    "knn_cosine",
    "random_forest",
    "xgboost",
    "lightgbm",
]
FEATURE_ORDER = ["FS1", "FS2", "FS3", "FS4", "FS5"]

EXPECTED = {
    "raw_records": 28_587,
    "prepared_records": 28_551,
    "excluded_records": 36,
    "semantic_groups": 20_825,
    "training_records": 22_842,
    "training_groups": 16_661,
    "holdout_records": 5_709,
    "holdout_groups": 4_164,
    "language_corrections": 3_330,
    "evaluation_pipelines": 8,
    "xai_models": 7,
    "global_h2_groups": 300,
    "local_cases": 12,
    "unique_explained_groups": 312,
    "common_features": 15_016,
    "fs5_features": 15_019,
    "knn_background_per_class": 3,
    "knn_cycles": 3,
    "final_cv_macro_f1": 0.5330,
    "final_holdout_macro_f1": 0.5597,
    "xgb_rho": 0.9666,
}

EXPECTED_PRIORITY_COUNTS = {
    "low": 5_883,
    "medium": 11_504,
    "high": 11_164,
}

EXPECTED_COMMON_MACRO_F1 = {
    "logistic_regression": 0.5397,
    "linear_svm": 0.5406,
    "complement_naive_bayes": 0.4893,
    "knn_cosine": 0.5478,
    "random_forest": 0.5566,
    "xgboost": 0.5112,
    "lightgbm": 0.5594,
}

EXPECTED_H1 = {
    "linear_svm": (0.0009, -0.0054, 0.0074, 1.000000, False),
    "complement_naive_bayes": (-0.0504, -0.0668, -0.0333, 1.000000, False),
    "knn_cosine": (0.0081, -0.0122, 0.0282, 0.854715, False),
    "random_forest": (0.0169, -0.0010, 0.0350, 0.161484, False),
    "xgboost": (-0.0285, -0.0462, -0.0103, 1.000000, False),
    "lightgbm": (0.0197, 0.0044, 0.0348, 0.023998, True),
}

EXPECTED_TOP10 = {
    "logistic_regression": 0.2657,
    "linear_svm": 0.2740,
    "complement_naive_bayes": 0.1882,
    "knn_cosine": 0.4238,
    "random_forest": 0.1916,
    "xgboost": 0.4286,
    "lightgbm": 0.4388,
}

EXPECTED_H2 = {
    "linear_svm": (-0.0083, -0.0106, -0.0062, 1.000000, False),
    "complement_naive_bayes": (0.0775, 0.0727, 0.0823, 0.000600, True),
    "knn_cosine": (-0.1582, -0.1670, -0.1489, 1.000000, False),
    "random_forest": (0.0741, 0.0676, 0.0805, 0.000600, True),
    "xgboost": (-0.1630, -0.1700, -0.1558, 1.000000, False),
    "lightgbm": (-0.1731, -0.1831, -0.1632, 1.000000, False),
}

PATHS = {
    # Modeling
    "modeling_manifest": M_TABLE / "03_modeling_manifest.json",
    "model_selection": M_TABLE / "03_modeling_model_selection.csv",
    "feature_set_selection": M_TABLE / "03_modeling_feature_set_selection.csv",
    "cv_results": M_TABLE / "03_modeling_cv_results.csv",
    "model_metadata": M_MODEL / "03_fitted_model_metadata.json",
    "split": M_CACHE / "03_split_assignments.parquet",
    "sentiment_features": M_CACHE / "03_sentiment_features.csv",
    "sentiment_metadata": M_CACHE / "03_sentiment_features_metadata.json",
    # Evaluation
    "evaluation_manifest": E_TABLE / "04_evaluation_manifest.json",
    "holdout_results": E_TABLE / "04_evaluation_holdout_results.csv",
    "h1_results": E_TABLE / "04_evaluation_h1_pairwise_results.csv",
    "final_intervals": E_TABLE / "04_evaluation_final_model_intervals.csv",
    "holdout_predictions": E_CACHE / "04_holdout_predictions.parquet",
    # Explainability
    "xai_manifest": X_TABLE / "05_explainability_manifest.json",
    "xai_sample": X_TABLE / "05_xai_sample.csv",
    "compactness_summary": X_TABLE / "05_shap_compactness_summary.csv",
    "h2_results": X_TABLE / "05_h2_pairwise_results.csv",
    "faithfulness_summary": X_TABLE / "05_faithfulness_summary.csv",
    "xai_validation": X_TABLE / "05_shap_validation_summary.csv",
    "xgb_sensitivity": X_TABLE / "05_xgboost_shap_sensitivity_summary.csv",
    # kNN SHAP sensitivity
    "knn_sensitivity_manifest": X_TABLE / "05a_knn_shap_sensitivity_manifest.json",
    "knn_sensitivity_sample": X_TABLE / "05a_knn_shap_sensitivity_sample.csv",
    "knn_sensitivity_summary": X_TABLE / "05a_knn_shap_sensitivity_summary.csv",
    "knn_sensitivity_comparisons": X_TABLE / "05a_knn_shap_sensitivity_comparisons.csv",
    "final_model_feature_groups": (
        X_TABLE / "05_final_FS5_lightgbm_feature_group_summary.csv"
    ),
    "final_model_validation": (
        X_TABLE / "05_final_FS5_lightgbm_validation.csv"
    ),
}

CENTRAL_FIGURES = [
    E_FIG / "04_common_feature_set_holdout_macro_f1.png",
    E_FIG / "04_H1_pairwise_macro_f1_differences.png",
    E_FIG / "04_final_model_confusion_matrix.png",
    X_FIG / "05_h2_compactness_by_model.png",
    X_FIG / "05_h2_pairwise_differences.png",
    X_FIG / "05_feature_group_shares_by_model.png",
    X_FIG / "05_top_features_FS5_lightgbm_low.png",
    X_FIG / "05_top_features_FS5_lightgbm_medium.png",
    X_FIG / "05_top_features_FS5_lightgbm_high.png",
]

TOL = 5e-4
checks = []


# %% 02 - Helpers

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload, path):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def bool_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def add(category, check, expected, observed, passed=None, detail=""):
    if passed is None:
        passed = observed == expected
    checks.append({
        "category": category,
        "check": check,
        "expected": str(expected),
        "observed": str(observed),
        "status": "PASS" if bool(passed) else "FAIL",
        "detail": detail,
    })


def close(category, check, expected, observed, tol=TOL):
    try:
        passed = np.isclose(float(observed), float(expected), atol=tol, rtol=0)
    except (TypeError, ValueError):
        passed = False
    add(category, check, expected, observed, passed, f"atol={tol}")


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def print_section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def check_manifest_fields(category, manifest, expected_fields):
    for field, expected in expected_fields.items():
        add(category, field, expected, manifest.get(field))


def check_hashes(category, manifest_hashes, path_map):
    for key, path in path_map.items():
        add(
            category,
            f"hash: {key}",
            manifest_hashes.get(key),
            sha256_file(path),
        )


def check_pairwise_table(
    category,
    frame,
    expected,
    model_column,
    delta_column,
    lower_column,
    upper_column,
    p_column,
    significance_column,
):
    add(category, "pairwise rows", 6, len(frame))

    for model, values in expected.items():
        delta, lower, upper, holm_p, significant = values
        row = frame.loc[frame[model_column].eq(model)]
        add(category, f"row: {model}", 1, len(row))
        if len(row) != 1:
            continue

        record = row.iloc[0]
        close(category, f"delta: {model}", delta, record[delta_column])
        close(category, f"CI lower: {model}", lower, record[lower_column])
        close(category, f"CI upper: {model}", upper, record[upper_column])
        close(category, f"Holm p: {model}", holm_p, record[p_column], tol=7e-4)
        add(
            category,
            f"significant: {model}",
            significant,
            bool_value(record[significance_column]),
        )


# %% 03 - Validate required inputs

required = [RAW_PATH, PREPARED_PATH, *PATHS.values(), *CENTRAL_FIGURES]
missing = [path for path in required if not path.exists()]

if missing:
    raise FileNotFoundError(
        "Missing final audit inputs:\n"
        + "\n".join(str(path) for path in missing)
    )

print_section("Final integrity audit input")
print(f"Audit build               : {SCRIPT_BUILD}")
print(f"Modeling build            : {EXPECTED_BUILDS['modeling']}")
print(f"Evaluation build          : {EXPECTED_BUILDS['evaluation']}")
print(f"Explainability build      : {EXPECTED_BUILDS['explainability']}")
print(f"Required input files      : {len(required)} / {len(required)} available")


# %% 04 - Validate data and frozen split

raw = pd.read_csv(RAW_PATH, low_memory=False)
prepared = pd.read_csv(PREPARED_PATH, low_memory=False)
split = pd.read_parquet(PATHS["split"])

require_columns(
    prepared,
    [
        "source_row_id",
        TARGET,
        SEMANTIC_GROUP,
        EXACT_GROUP,
        "language",
        "text",
        "text_clean",
        "type",
        "queue",
    ],
    "Prepared data",
)
require_columns(
    split,
    ["modeling_row_id", SEMANTIC_GROUP, EXACT_GROUP, "split"],
    "Split assignments",
)

raw_hash = sha256_file(RAW_PATH)
prepared_hash = sha256_file(PREPARED_PATH)

for name, expected, observed in [
    ("raw records", EXPECTED["raw_records"], len(raw)),
    ("prepared records", EXPECTED["prepared_records"], len(prepared)),
    ("excluded records", EXPECTED["excluded_records"], len(raw) - len(prepared)),
    ("semantic groups", EXPECTED["semantic_groups"], prepared[SEMANTIC_GROUP].nunique()),
    ("raw SHA-256", RAW_SHA256, raw_hash),
]:
    add("data", name, expected, observed)

priority_counts = (
    prepared[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .value_counts()
    .to_dict()
)
add("data", "priority class counts", EXPECTED_PRIORITY_COUNTS, priority_counts)

max_semantic_per_exact = (
    prepared.groupby(EXACT_GROUP)[SEMANTIC_GROUP].nunique().max()
)
add("data", "exact groups nested in semantic groups", 1, int(max_semantic_per_exact))

source_positions = (
    pd.to_numeric(prepared["source_row_id"], errors="raise")
    .astype(int)
    .to_numpy()
    - 1
)
source_map_ok = (
    source_positions.min() >= 0
    and source_positions.max() < len(raw)
)
add("data", "source-row mapping valid", True, source_map_ok, source_map_ok)

if source_map_ok and "language" in raw.columns:
    raw_language = (
        raw.iloc[source_positions]["language"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .reset_index(drop=True)
    )
    prepared_language = (
        prepared["language"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .reset_index(drop=True)
    )
    add(
        "data",
        "language-label corrections",
        EXPECTED["language_corrections"],
        int(raw_language.ne(prepared_language).sum()),
    )

prepared_check = prepared.copy()
prepared_check["modeling_row_id"] = np.arange(len(prepared_check), dtype=np.int64)
prepared_check = prepared_check.sort_values("modeling_row_id").reset_index(drop=True)
split_check = split.sort_values("modeling_row_id").reset_index(drop=True)

add(
    "split",
    "row order matches prepared data",
    True,
    np.array_equal(
        split_check["modeling_row_id"].to_numpy(dtype=np.int64),
        prepared_check["modeling_row_id"].to_numpy(dtype=np.int64),
    ),
)
add(
    "split",
    "semantic IDs match prepared data",
    True,
    np.array_equal(
        split_check[SEMANTIC_GROUP].astype(str).to_numpy(),
        prepared_check[SEMANTIC_GROUP].astype(str).to_numpy(),
    ),
)
add(
    "split",
    "exact IDs match prepared data",
    True,
    np.array_equal(
        split_check[EXACT_GROUP].astype(str).to_numpy(),
        prepared_check[EXACT_GROUP].astype(str).to_numpy(),
    ),
)

prepared_check["split"] = split_check["split"].to_numpy()
training = prepared_check.loc[prepared_check["split"].eq("training")]
holdout = prepared_check.loc[prepared_check["split"].eq("holdout")]

split_counts = {
    "training records": len(training),
    "training groups": training[SEMANTIC_GROUP].nunique(),
    "hold-out records": len(holdout),
    "hold-out groups": holdout[SEMANTIC_GROUP].nunique(),
}
for name, observed in split_counts.items():
    key = name.replace("-", "").replace(" ", "_")
    key = {
        "training_records": "training_records",
        "training_groups": "training_groups",
        "holdout_records": "holdout_records",
        "holdout_groups": "holdout_groups",
    }[key]
    add("split", name, EXPECTED[key], observed)

semantic_overlap = len(
    set(training[SEMANTIC_GROUP].astype(str))
    & set(holdout[SEMANTIC_GROUP].astype(str))
)
exact_overlap = len(
    set(training[EXACT_GROUP].astype(str))
    & set(holdout[EXACT_GROUP].astype(str))
)
add("split", "semantic-group overlap", 0, semantic_overlap)
add("split", "exact-text-group overlap", 0, exact_overlap)


# %% 05 - Validate Modeling

m_manifest = load_json(PATHS["modeling_manifest"])
selection = pd.read_csv(PATHS["model_selection"])
feature_set_selection = pd.read_csv(PATHS["feature_set_selection"])
cv_results = pd.read_csv(PATHS["cv_results"])
model_metadata = load_json(PATHS["model_metadata"])

check_manifest_fields(
    "modeling",
    m_manifest,
    {
        "script_build": EXPECTED_BUILDS["modeling"],
        "run_id": EXPECTED_RUN_IDS["modeling"],
        "data_sha256": prepared_hash,
        "prepared_records": EXPECTED["prepared_records"],
        "semantic_groups": EXPECTED["semantic_groups"],
        "training_records": EXPECTED["training_records"],
        "training_semantic_groups": EXPECTED["training_groups"],
        "holdout_records": EXPECTED["holdout_records"],
        "holdout_semantic_groups": EXPECTED["holdout_groups"],
        "group_column": SEMANTIC_GROUP,
        "exact_group_column": EXACT_GROUP,
        "strict_data_freeze": True,
        "holdout_scored_in_modeling": False,
        "holdout_used_for_parameter_tuning": False,
        "holdout_used_for_model_selection": False,
        "holdout_used_for_preprocessing": False,
        "holdout_used_for_fitted_preprocessing": False,
    },
)

add("modeling", "classifier order", MODEL_ORDER, m_manifest.get("models"))

require_columns(
    cv_results,
    [
        "feature_set",
        "model",
        "mean_macro_f1",
        "mean_weighted_f1",
        "mean_accuracy",
    ],
    "CV results",
)

feature_rank = {feature_set: index for index, feature_set in enumerate(FEATURE_ORDER)}
audit_feature_selection = (
    cv_results
    .groupby("feature_set", as_index=False)
    .agg(
        mean_macro_f1_across_models=("mean_macro_f1", "mean"),
        mean_weighted_f1_across_models=("mean_weighted_f1", "mean"),
        mean_accuracy_across_models=("mean_accuracy", "mean"),
    )
)
audit_feature_selection["feature_rank"] = (
    audit_feature_selection["feature_set"].map(feature_rank)
)
audit_feature_selection = (
    audit_feature_selection
    .sort_values(
        [
            "mean_macro_f1_across_models",
            "mean_weighted_f1_across_models",
            "mean_accuracy_across_models",
            "feature_rank",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    .reset_index(drop=True)
)
AUDIT_COMMON_FEATURE_SET = str(
    audit_feature_selection.iloc[0]["feature_set"]
)

common_selection_manifest = m_manifest.get(
    "common_feature_set_selection",
    {},
)
add(
    "modeling",
    "CV-selected common feature set",
    AUDIT_COMMON_FEATURE_SET,
    common_selection_manifest.get("selected_feature_set"),
)
add(
    "modeling",
    "common-set selection hold-out used",
    False,
    common_selection_manifest.get("holdout_used"),
)

require_columns(
    feature_set_selection,
    [
        "feature_set",
        "mean_macro_f1_across_models",
        "mean_weighted_f1_across_models",
        "mean_accuracy_across_models",
        "selected_for_common_comparison",
    ],
    "Feature-set selection",
)

selected_mask = (
    feature_set_selection["selected_for_common_comparison"]
    .map(bool_value)
)
selected_feature_rows = feature_set_selection.loc[selected_mask]
add(
    "modeling",
    "one common feature set selected",
    1,
    len(selected_feature_rows),
)
if len(selected_feature_rows) == 1:
    add(
        "modeling",
        "selection-table common feature set",
        AUDIT_COMMON_FEATURE_SET,
        str(selected_feature_rows.iloc[0]["feature_set"]),
    )

for audit_row in audit_feature_selection.itertuples(index=False):
    stored = feature_set_selection.loc[
        feature_set_selection["feature_set"].eq(audit_row.feature_set)
    ]
    add(
        "modeling",
        f"feature-selection row: {audit_row.feature_set}",
        1,
        len(stored),
    )
    if len(stored) == 1:
        stored_row = stored.iloc[0]
        close(
            "modeling",
            f"mean CV Macro-F1 across models: {audit_row.feature_set}",
            audit_row.mean_macro_f1_across_models,
            stored_row["mean_macro_f1_across_models"],
            tol=1e-10,
        )
        close(
            "modeling",
            f"mean CV Weighted-F1 across models: {audit_row.feature_set}",
            audit_row.mean_weighted_f1_across_models,
            stored_row["mean_weighted_f1_across_models"],
            tol=1e-10,
        )
        close(
            "modeling",
            f"mean CV Accuracy across models: {audit_row.feature_set}",
            audit_row.mean_accuracy_across_models,
            stored_row["mean_accuracy_across_models"],
            tol=1e-10,
        )

best_cv = cv_results.sort_values(
    ["mean_macro_f1", "mean_weighted_f1", "mean_accuracy"],
    ascending=False,
    kind="mergesort",
).iloc[0]
AUDIT_FINAL_MODEL_KEY = (
    str(best_cv["feature_set"]),
    str(best_cv["model"]),
)

final_model = m_manifest.get("final_model", {})
add(
    "modeling",
    "final feature set from CV",
    AUDIT_FINAL_MODEL_KEY[0],
    final_model.get("feature_set"),
)
add(
    "modeling",
    "final model from CV",
    AUDIT_FINAL_MODEL_KEY[1],
    final_model.get("model"),
)
close(
    "modeling",
    "CV-best Macro-F1",
    EXPECTED["final_cv_macro_f1"],
    best_cv["mean_macro_f1"],
)

h1_design = m_manifest.get("hypothesis_operationalization", {}).get("H1", {})
for field, expected in {
    "feature_set": AUDIT_COMMON_FEATURE_SET,
    "baseline": "logistic_regression",
    "pairwise": True,
    "family_mean_comparison": False,
}.items():
    add("modeling", f"H1 {field}", expected, h1_design.get(field))

h2_design = m_manifest.get("hypothesis_operationalization", {}).get("H2", {})
add(
    "modeling",
    "H2 feature_set",
    AUDIT_COMMON_FEATURE_SET,
    h2_design.get("feature_set"),
)

require_columns(
    selection,
    ["selection", "feature_set", "model", "macro_f1_cv", "model_artifact"],
    "Model selection",
)

final_rows = selection.loc[selection["selection"].eq("final_overall_model")]
add("modeling", "one final model selected", 1, len(final_rows))
if len(final_rows) == 1:
    row = final_rows.iloc[0]
    add(
        "modeling",
        "selection final feature set",
        AUDIT_FINAL_MODEL_KEY[0],
        row["feature_set"],
    )
    add(
        "modeling",
        "selection final model",
        AUDIT_FINAL_MODEL_KEY[1],
        row["model"],
    )
    close(
        "modeling",
        "final CV Macro-F1",
        EXPECTED["final_cv_macro_f1"],
        row["macro_f1_cv"],
    )

text_only_best = (
    cv_results.loc[cv_results["feature_set"].eq("FS1")]
    .sort_values(
        ["mean_macro_f1", "mean_weighted_f1", "mean_accuracy"],
        ascending=False,
        kind="mergesort",
    )
    .iloc[0]
)
expected_selected_pipelines = {
    ("FS1", str(text_only_best["model"])),
    *{(AUDIT_COMMON_FEATURE_SET, model) for model in MODEL_ORDER},
    AUDIT_FINAL_MODEL_KEY,
}
observed_pipelines = set(zip(selection["feature_set"], selection["model"]))
add(
    "modeling",
    "selected pipeline set",
    expected_selected_pipelines,
    observed_pipelines,
)

check_hashes(
    "modeling",
    m_manifest.get("artifact_hashes", {}),
    {
        "split": PATHS["split"],
        "feature_set_selection": PATHS["feature_set_selection"],
        "model_selection": PATHS["model_selection"],
        "fitted_model_metadata": PATHS["model_metadata"],
        "sentiment_features": PATHS["sentiment_features"],
        "sentiment_metadata": PATHS["sentiment_metadata"],
    },
)

metadata_models = model_metadata.get("models", {})
manifest_model_hashes = m_manifest.get("selected_model_hashes", {})
model_hashes = {}

for artifact in sorted(set(selection["model_artifact"])):
    path = M_MODEL / artifact
    add("modeling", f"model exists: {artifact}", True, path.exists(), path.exists())
    if not path.exists():
        continue

    observed_hash = sha256_file(path)
    model_hashes[artifact] = observed_hash
    add(
        "modeling",
        f"metadata model hash: {artifact}",
        metadata_models.get(artifact, {}).get("sha256"),
        observed_hash,
    )
    add(
        "modeling",
        f"manifest model hash: {artifact}",
        manifest_model_hashes.get(artifact),
        observed_hash,
    )


# %% 06 - Validate Evaluation and H1

e_manifest = load_json(PATHS["evaluation_manifest"])
holdout_results = pd.read_csv(PATHS["holdout_results"])
h1_results = pd.read_csv(PATHS["h1_results"])
final_intervals = pd.read_csv(PATHS["final_intervals"])
predictions = pd.read_parquet(PATHS["holdout_predictions"])

check_manifest_fields(
    "evaluation",
    e_manifest,
    {
        "script_build": EXPECTED_BUILDS["evaluation"],
        "run_id": EXPECTED_RUN_IDS["evaluation"],
        "modeling_build": EXPECTED_BUILDS["modeling"],
        "modeling_run_id": EXPECTED_RUN_IDS["modeling"],
        "data_sha256": prepared_hash,
        "training_records": EXPECTED["training_records"],
        "holdout_records": EXPECTED["holdout_records"],
        "holdout_semantic_groups": EXPECTED["holdout_groups"],
        "model_training_performed": False,
        "hyperparameter_tuning_performed": False,
        "model_selection_performed": False,
        "holdout_used_for_model_selection": False,
    },
)

e_final = e_manifest.get("final_model", {})
add(
    "evaluation",
    "final feature set",
    AUDIT_FINAL_MODEL_KEY[0],
    e_final.get("feature_set"),
)
add(
    "evaluation",
    "final model",
    AUDIT_FINAL_MODEL_KEY[1],
    e_final.get("model"),
)

evaluation_common = e_manifest.get("common_feature_set_selection", {})
add(
    "evaluation",
    "common feature set inherited from Modeling",
    AUDIT_COMMON_FEATURE_SET,
    evaluation_common.get("feature_set"),
)
add(
    "evaluation",
    "feature-set selection performed in Evaluation",
    False,
    evaluation_common.get("selection_performed_in_evaluation"),
)

manifest_pipelines = {
    (row.get("feature_set"), row.get("model"))
    for row in e_manifest.get("evaluated_pipelines", [])
}
expected_evaluation_pipelines = {
    *{(AUDIT_COMMON_FEATURE_SET, model) for model in MODEL_ORDER},
    AUDIT_FINAL_MODEL_KEY,
}
add(
    "evaluation",
    "evaluated pipeline set",
    expected_evaluation_pipelines,
    manifest_pipelines,
)
add(
    "evaluation",
    "evaluated pipeline count",
    EXPECTED["evaluation_pipelines"],
    len(manifest_pipelines),
)

require_columns(
    holdout_results,
    ["feature_set", "model", "macro_f1"],
    "Hold-out results",
)

for model, expected_macro_f1 in EXPECTED_COMMON_MACRO_F1.items():
    row = holdout_results.loc[
        holdout_results["feature_set"].eq(AUDIT_COMMON_FEATURE_SET)
        & holdout_results["model"].eq(model)
    ]
    add(
        "evaluation",
        f"common-set row: {model}",
        1,
        len(row),
    )
    if len(row) == 1:
        close(
            "evaluation",
            f"common-set Macro-F1: {model}",
            expected_macro_f1,
            row.iloc[0]["macro_f1"],
        )

final_row = holdout_results.loc[
    holdout_results["feature_set"].eq(AUDIT_FINAL_MODEL_KEY[0])
    & holdout_results["model"].eq(AUDIT_FINAL_MODEL_KEY[1])
]
add("evaluation", "final hold-out row", 1, len(final_row))
if len(final_row) == 1:
    close(
        "evaluation",
        "final hold-out Macro-F1",
        EXPECTED["final_holdout_macro_f1"],
        final_row.iloc[0]["macro_f1"],
    )

h1_manifest = e_manifest.get("h1", {})
add(
    "evaluation",
    "H1 feature set",
    AUDIT_COMMON_FEATURE_SET,
    h1_manifest.get("feature_set"),
)
for field, expected in {
    "pairwise": True,
    "family_mean_comparison": False,
    "number_of_comparisons": 6,
    "supported": True,
}.items():
    add("evaluation", f"H1 {field}", expected, h1_manifest.get(field))
add(
    "evaluation",
    "H1 supporting alternatives",
    ["lightgbm"],
    h1_manifest.get("supporting_alternatives"),
)

require_columns(
    h1_results,
    [
        "alternative_model",
        "delta_macro_f1",
        "ci_lower_95",
        "ci_upper_95",
        "p_value_one_sided_holm",
        "significant_higher_holm",
    ],
    "H1 pairwise results",
)
check_pairwise_table(
    "H1",
    h1_results,
    EXPECTED_H1,
    "alternative_model",
    "delta_macro_f1",
    "ci_lower_95",
    "ci_upper_95",
    "p_value_one_sided_holm",
    "significant_higher_holm",
)

require_columns(
    final_intervals,
    ["result_type", "metric", "observed"],
    "Final-model intervals",
)
macro_row = final_intervals.loc[
    final_intervals["result_type"].eq("overall_metric")
    & final_intervals["metric"].eq("macro_f1")
]
add("evaluation", "final Macro-F1 interval row", 1, len(macro_row))
if len(macro_row) == 1:
    close(
        "evaluation",
        "final interval Macro-F1",
        EXPECTED["final_holdout_macro_f1"],
        macro_row.iloc[0]["observed"],
    )

expected_prediction_columns = {
    *{
        f"pred_{AUDIT_COMMON_FEATURE_SET}__{model}"
        for model in MODEL_ORDER
    },
    f"pred_{AUDIT_FINAL_MODEL_KEY[0]}__{AUDIT_FINAL_MODEL_KEY[1]}",
}
add("evaluation", "prediction rows", EXPECTED["holdout_records"], len(predictions))
add(
    "evaluation",
    "prediction columns",
    True,
    expected_prediction_columns.issubset(predictions.columns),
)

check_hashes(
    "evaluation",
    e_manifest.get("output_hashes", {}),
    {
        "holdout_results": PATHS["holdout_results"],
        "h1_pairwise_results": PATHS["h1_results"],
        "final_model_intervals": PATHS["final_intervals"],
        "holdout_predictions": PATHS["holdout_predictions"],
    },
)


# %% 07 - Validate Explainability and H2

x_manifest = load_json(PATHS["xai_manifest"])
xai_sample = pd.read_csv(PATHS["xai_sample"], low_memory=False)
compactness = pd.read_csv(PATHS["compactness_summary"])
h2_results = pd.read_csv(PATHS["h2_results"])
faithfulness = pd.read_csv(PATHS["faithfulness_summary"])
xai_validation = pd.read_csv(PATHS["xai_validation"])
xgb_summary = pd.read_csv(PATHS["xgb_sensitivity"])
final_groups = pd.read_csv(PATHS["final_model_feature_groups"])
final_validation = pd.read_csv(PATHS["final_model_validation"])

check_manifest_fields(
    "explainability",
    x_manifest,
    {
        "script_build": EXPECTED_BUILDS["explainability"],
        "modeling_build": EXPECTED_BUILDS["modeling"],
        "evaluation_build": EXPECTED_BUILDS["evaluation"],
        "run_id": EXPECTED_RUN_IDS["explainability"],
        "data_sha256": prepared_hash,
        "feature_set": AUDIT_COMMON_FEATURE_SET,
        "global_h2_groups": EXPECTED["global_h2_groups"],
        "local_cases": EXPECTED["local_cases"],
        "unique_explained_groups": EXPECTED["unique_explained_groups"],
        "knn_permutation_background_groups_per_class": EXPECTED["knn_background_per_class"],
        "knn_permutation_cycles": EXPECTED["knn_cycles"],
        "model_training_performed": False,
        "hyperparameter_tuning_performed": False,
        "model_selection_performed": False,
        "h2_model_eligibility_based_on_performance": False,
    },
)

add("explainability", "all frozen classifiers", MODEL_ORDER, x_manifest.get("all_frozen_models"))
add("explainability", "H2 model set", MODEL_ORDER, x_manifest.get("h2_models"))
add(
    "explainability",
    "H2 model count",
    EXPECTED["xai_models"],
    len(x_manifest.get("h2_models", [])),
)

xai_common = x_manifest.get("common_feature_set_selection", {})
add(
    "explainability",
    "common feature set inherited from Modeling",
    AUDIT_COMMON_FEATURE_SET,
    xai_common.get("feature_set"),
)
add(
    "explainability",
    "feature-set selection performed in Explainability",
    False,
    xai_common.get("selection_performed_in_explainability"),
)

h2_manifest = x_manifest.get("h2", {})
add(
    "explainability",
    "H2 feature set",
    AUDIT_COMMON_FEATURE_SET,
    h2_manifest.get("feature_set"),
)
for field, expected in {
    "baseline": "logistic_regression",
    "pairwise": True,
    "family_mean_comparison": False,
    "number_of_comparisons": 6,
    "supported": True,
}.items():
    add("explainability", f"H2 {field}", expected, h2_manifest.get(field))

add(
    "explainability",
    "H2 supporting alternatives",
    {"complement_naive_bayes", "random_forest"},
    set(h2_manifest.get("supporting_alternatives", [])),
)

for flag, expected in [
    ("is_global_h2_case", EXPECTED["global_h2_groups"]),
    ("is_local_case", EXPECTED["local_cases"]),
]:
    if flag in xai_sample.columns:
        add(
            "explainability",
            f"{flag} count",
            expected,
            int(xai_sample[flag].map(bool_value).sum()),
        )

for column in [
    "sentiment_negative",
    "sentiment_neutral",
    "sentiment_positive",
    "pred_final_model",
]:
    add(
        "explainability",
        f"XAI sample column: {column}",
        True,
        column in xai_sample.columns,
    )

require_columns(
    compactness,
    ["model", "scope", "top_10_share_mean"],
    "Compactness summary",
)
full_compactness = compactness.loc[
    compactness["scope"].eq("full_attribution")
]
for model, expected_top10 in EXPECTED_TOP10.items():
    row = full_compactness.loc[full_compactness["model"].eq(model)]
    add("explainability", f"compactness row: {model}", 1, len(row))
    if len(row) == 1:
        close(
            "explainability",
            f"Top-10 concentration: {model}",
            expected_top10,
            row.iloc[0]["top_10_share_mean"],
        )

require_columns(
    h2_results,
    [
        "alternative_model",
        "delta_lr_minus_alternative",
        "ci_lower_95",
        "ci_upper_95",
        "p_value_one_sided_holm",
        "significant_lr_more_compact_holm",
    ],
    "H2 pairwise results",
)
check_pairwise_table(
    "H2",
    h2_results,
    EXPECTED_H2,
    "alternative_model",
    "delta_lr_minus_alternative",
    "ci_lower_95",
    "ci_upper_95",
    "p_value_one_sided_holm",
    "significant_lr_more_compact_holm",
)

require_columns(
    xai_validation,
    [
        "model",
        "transformed_features",
        "max_additivity_error",
        "prediction_reconstruction_matches",
    ],
    "SHAP validation",
)
add("explainability", "common-set validation rows", 7, len(xai_validation))

feature_counts = (
    xai_validation[["model", "transformed_features"]]
    .drop_duplicates()
    .set_index("model")["transformed_features"]
    .astype(int)
    .to_dict()
)
add(
    "explainability",
    "common-set transformed feature counts",
    {model: EXPECTED["common_features"] for model in MODEL_ORDER},
    feature_counts,
)

max_common_error = float(xai_validation["max_additivity_error"].max())
add(
    "explainability",
    "common-set additivity <= 0.001",
    True,
    max_common_error <= 0.001,
    max_common_error <= 0.001,
    f"max={max_common_error:.8f}",
)
add(
    "explainability",
    "common-set prediction reconstruction",
    True,
    xai_validation["prediction_reconstruction_matches"]
    .map(bool_value)
    .all(),
)

require_columns(
    xgb_summary,
    ["metric", "approximate_value"],
    "XGBoost sensitivity",
)
rho_row = xgb_summary.loc[
    xgb_summary["metric"].eq("mean_abs_shap_spearman")
]
add("explainability", "XGB sensitivity row", 1, len(rho_row))
if len(rho_row) == 1:
    close(
        "explainability",
        "XGB exact/approximate rho",
        EXPECTED["xgb_rho"],
        rho_row.iloc[0]["approximate_value"],
    )

final_xai = x_manifest.get("final_model_explainability", {})
for field, expected in {
    "feature_set": AUDIT_FINAL_MODEL_KEY[0],
    "model": AUDIT_FINAL_MODEL_KEY[1],
    "global_groups": EXPECTED["global_h2_groups"],
    "local_cases": EXPECTED["local_cases"],
}.items():
    add("explainability", f"final model {field}", expected, final_xai.get(field))

add(
    "explainability",
    "final model feature groups",
    {"text", "type", "language", "queue", "sentiment"},
    set(final_xai.get("feature_groups", [])),
)

require_columns(
    final_validation,
    [
        "feature_set",
        "model",
        "transformed_features",
        "max_additivity_error",
        "prediction_reconstruction_matches",
    ],
    "Final FS5 LightGBM validation",
)
add("explainability", "final FS5 validation rows", 1, len(final_validation))
if len(final_validation) == 1:
    row = final_validation.iloc[0]
    add(
        "explainability",
        "final-model feature set",
        AUDIT_FINAL_MODEL_KEY[0],
        row["feature_set"],
    )
    add(
        "explainability",
        "final-model classifier",
        AUDIT_FINAL_MODEL_KEY[1],
        row["model"],
    )
    add(
        "explainability",
        "final FS5 transformed features",
        EXPECTED["fs5_features"],
        int(row["transformed_features"]),
    )
    final_error = float(row["max_additivity_error"])
    add(
        "explainability",
        "final FS5 additivity <= 0.001",
        True,
        final_error <= 0.001,
        final_error <= 0.001,
        f"max={final_error:.8f}",
    )
    add(
        "explainability",
        "final FS5 prediction reconstruction",
        True,
        bool_value(row["prediction_reconstruction_matches"]),
    )

require_columns(
    final_groups,
    ["class_output", "feature_group", "share_of_total_abs_shap"],
    "Final FS5 feature-group summary",
)
add(
    "explainability",
    "final FS5 feature groups",
    {"text", "type", "language", "queue", "sentiment"},
    set(final_groups["feature_group"].astype(str)),
)
add(
    "explainability",
    "final FS5 class outputs",
    {"low", "medium", "high", "predicted_class"},
    set(final_groups["class_output"].astype(str)),
)

sentiment_rows = final_groups.loc[
    final_groups["feature_group"].eq("sentiment")
]
add("explainability", "sentiment group rows", 4, len(sentiment_rows))
if len(sentiment_rows):
    sentiment_shares = sentiment_rows["share_of_total_abs_shap"].to_numpy(dtype=float)
    valid_sentiment = (
        np.isfinite(sentiment_shares).all()
        and (sentiment_shares >= 0).all()
    )
    add(
        "explainability",
        "sentiment SHAP shares valid",
        True,
        valid_sentiment,
        valid_sentiment,
    )

add(
    "explainability",
    "faithfulness summary present",
    True,
    len(faithfulness) > 0,
    len(faithfulness) > 0,
)

check_hashes(
    "explainability",
    x_manifest.get("output_hashes", {}),
    {
        "xai_sample": PATHS["xai_sample"],
        "compactness_summary": PATHS["compactness_summary"],
        "h2_pairwise_results": PATHS["h2_results"],
        "faithfulness_summary": PATHS["faithfulness_summary"],
        "validation_summary": PATHS["xai_validation"],
        "xgboost_sensitivity_summary": PATHS["xgb_sensitivity"],
        "final_model_feature_group_summary": PATHS["final_model_feature_groups"],
        "final_model_validation": PATHS["final_model_validation"],
    },
)


# %% 08 - Validate central figures

for path in CENTRAL_FIGURES:
    add(
        "figures",
        path.name,
        True,
        path.exists(),
        path.exists(),
    )

xai_figure_hashes = x_manifest.get("figure_hashes", {})
for path in CENTRAL_FIGURES:
    if path.parent == X_FIG:
        add(
            "figures",
            f"XAI figure hash: {path.name}",
            xai_figure_hashes.get(path.name),
            sha256_file(path),
        )



# %% 08a - Validate kNN SHAP sensitivity provenance

knn_sensitivity_manifest = load_json(PATHS["knn_sensitivity_manifest"])
knn_sensitivity_sample = pd.read_csv(PATHS["knn_sensitivity_sample"])
knn_sensitivity_summary = pd.read_csv(PATHS["knn_sensitivity_summary"])
knn_sensitivity_comparisons = pd.read_csv(PATHS["knn_sensitivity_comparisons"])

add(
    "knn_sensitivity",
    "reference XAI build",
    EXPECTED_BUILDS["explainability"],
    knn_sensitivity_manifest.get("reference_xai_build"),
)
add(
    "knn_sensitivity",
    "reference XAI run ID",
    EXPECTED_RUN_IDS["explainability"],
    knn_sensitivity_manifest.get("reference_xai_run_id"),
)
add(
    "knn_sensitivity",
    "reference XAI manifest hash",
    sha256_file(PATHS["xai_manifest"]),
    knn_sensitivity_manifest.get("reference_xai_manifest_sha256"),
)
add(
    "knn_sensitivity",
    "model",
    "knn_cosine",
    knn_sensitivity_manifest.get("model"),
)
add(
    "knn_sensitivity",
    "feature set",
    AUDIT_COMMON_FEATURE_SET,
    knn_sensitivity_manifest.get("feature_set"),
)
add(
    "knn_sensitivity",
    "current 05 background groups/class",
    EXPECTED["knn_background_per_class"],
    knn_sensitivity_manifest.get("current_05_knn_background_groups_per_class"),
)
add(
    "knn_sensitivity",
    "current 05 permutation cycles",
    EXPECTED["knn_cycles"],
    knn_sensitivity_manifest.get("current_05_knn_permutation_cycles"),
)
add(
    "knn_sensitivity",
    "sensitivity groups/class",
    30,
    knn_sensitivity_manifest.get("sensitivity_groups_per_class"),
)
add(
    "knn_sensitivity",
    "sensitivity groups",
    90,
    knn_sensitivity_manifest.get("sensitivity_groups"),
)
add(
    "knn_sensitivity",
    "sample hash",
    knn_sensitivity_manifest.get("sample_sha256"),
    sha256_file(PATHS["knn_sensitivity_sample"]),
)
add(
    "knn_sensitivity",
    "summary hash",
    knn_sensitivity_manifest.get("summary_sha256"),
    sha256_file(PATHS["knn_sensitivity_summary"]),
)
add(
    "knn_sensitivity",
    "comparison hash",
    knn_sensitivity_manifest.get("comparison_sha256"),
    sha256_file(PATHS["knn_sensitivity_comparisons"]),
)

expected_configs = {
    "reference": (3, 3),
    "reduced_background": (1, 3),
    "reduced_cycles": (3, 1),
}
manifest_configs = {
    str(item.get("config")): (
        int(item.get("background_groups_per_class", -1)),
        int(item.get("cycles", -1)),
    )
    for item in knn_sensitivity_manifest.get("configs", [])
}
add(
    "knn_sensitivity",
    "configuration design",
    expected_configs,
    manifest_configs,
)
add("knn_sensitivity", "summary rows", 3, len(knn_sensitivity_summary))
add("knn_sensitivity", "comparison rows", 2, len(knn_sensitivity_comparisons))
add(
    "knn_sensitivity",
    "sample semantic groups",
    90,
    knn_sensitivity_sample[SEMANTIC_GROUP].nunique(),
)

if "max_additivity_error" in knn_sensitivity_summary.columns:
    observed_max_error = pd.to_numeric(
        knn_sensitivity_summary["max_additivity_error"],
        errors="coerce",
    ).max()
    add(
        "knn_sensitivity",
        "max additivity error <= 1e-4",
        True,
        bool(pd.notna(observed_max_error) and observed_max_error <= 1e-4),
    )


# %% 09 - Save audit and report status

audit = pd.DataFrame(checks)
audit.to_csv(AUDIT_PATH, index=False, encoding="utf-8-sig")

passed = int(audit["status"].eq("PASS").sum())
failed = int(audit["status"].eq("FAIL").sum())
status = "PASS" if failed == 0 else "FAIL"

save_json(
    {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": SCRIPT_PATH.name,
        "script_build": SCRIPT_BUILD,
        "overall_status": status,
        "checks_performed": len(audit),
        "passed_checks": passed,
        "failed_checks": failed,
        "expected_builds": EXPECTED_BUILDS,
        "golden_results": EXPECTED,
        "expected_h1": EXPECTED_H1,
        "expected_h2": EXPECTED_H2,
        "raw_sha256": raw_hash,
        "prepared_sha256": prepared_hash,
        "model_hashes": model_hashes,
        "audit_results_file": str(AUDIT_PATH.relative_to(PROJECT_ROOT)),
    },
    MANIFEST_PATH,
)

print_section("Final integrity audit")
print(f"Checks performed          : {len(audit)}")
print(f"Passed                    : {passed}")
print(f"Failed                    : {failed}")
print(f"Overall status            : {status}")

print_section("Frozen final results")
print(f"Prepared records          : {EXPECTED['prepared_records']:,}".replace(",", "."))
print(f"Semantic groups           : {EXPECTED['semantic_groups']:,}".replace(",", "."))
print(f"Training / hold-out       : {EXPECTED['training_records']:,} / {EXPECTED['holdout_records']:,}".replace(",", "."))
print(f"Common comparison set     : {AUDIT_COMMON_FEATURE_SET}")
print(f"Final model               : {AUDIT_FINAL_MODEL_KEY[0]} + {AUDIT_FINAL_MODEL_KEY[1]}")
print(f"Final CV Macro-F1         : {EXPECTED['final_cv_macro_f1']:.4f}")
print(f"Final hold-out Macro-F1   : {EXPECTED['final_holdout_macro_f1']:.4f}")
print(
    f"kNN SHAP final design     : "
    f"{EXPECTED['knn_background_per_class'] * 3} background groups, "
    f"{EXPECTED['knn_cycles']} cycles"
)
print(f"Final FS5 SHAP features   : {EXPECTED['fs5_features']:,}".replace(",", "."))
print(f"XGB sensitivity rho       : {EXPECTED['xgb_rho']:.4f}")

if failed:
    print_section("Failed checks")
    print(
        audit.loc[
            audit["status"].eq("FAIL"),
            ["category", "check", "expected", "observed", "detail"],
        ].to_string(index=False)
    )
    raise RuntimeError(
        "Final integrity audit failed. "
        "The exact failed checks are printed directly above this traceback "
        "and saved in reports/tables/06_final_integrity_audit/06_integrity_audit.csv."
    )
