"""CRISP-DM Evaluation for the CV-selected common feature-set design."""

# %% 00 - Load packages

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% 01 - Configure Evaluation

SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name.lower() != "src":
    raise RuntimeError("Place this file in <project>/src before running it.")

PROJECT_ROOT = SCRIPT_PATH.parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_it_support_prepared.csv"
MODELING_TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "03_tickets_modeling"
MODELING_MODEL_DIR = PROJECT_ROOT / "models" / "03_tickets_modeling"
MODELING_CACHE_DIR = PROJECT_ROOT / "cache" / "03_tickets_modeling"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "04_evaluation"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "04_evaluation"
CACHE_DIR = PROJECT_ROOT / "cache" / "04_evaluation"

for directory in [TABLE_DIR, FIGURE_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RUN_ID = "evaluation_common_feature_selection_v1"
SCRIPT_BUILD = "2026-08-16_evaluation_common_feature_selection_v1"
EXPECTED_MODELING_BUILD = "2026-08-16_final_comparative_common_feature_selection_v1"
EXPECTED_MODELING_RUN_ID = "final_comparative_common_feature_selection_v1"

ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_RANDOM_STATE = 42
BOOTSTRAP_BATCH_SIZE = 200
REUSE_VALID_RESULTS = True

EXPECTED_RECORDS = 28_551
EXPECTED_SEMANTIC_GROUPS = 20_825
EXPECTED_SEMANTIC_CONFLICT_GROUPS = 105
EXPECTED_TRAINING_RECORDS = 22_842
EXPECTED_TRAINING_GROUPS = 16_661
EXPECTED_HOLDOUT_RECORDS = 5_709
EXPECTED_HOLDOUT_GROUPS = 4_164

TARGET_COLUMN = "priority"
GROUP_COLUMN = "semantic_group_id"
EXACT_GROUP_COLUMN = "text_group_id"
TEXT_COLUMN = "text_clean"
SENTIMENT_TEXT_COLUMN = "text"
SENTIMENT_COLUMNS = ["sentiment_negative", "sentiment_neutral", "sentiment_positive"]

CLASS_ORDER = ["low", "medium", "high"]
CLASS_TO_INT = {label: i for i, label in enumerate(CLASS_ORDER)}

MODEL_ORDER = [
    "logistic_regression",
    "linear_svm",
    "complement_naive_bayes",
    "knn_cosine",
    "random_forest",
    "xgboost",
    "lightgbm",
]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "complement_naive_bayes": "Complement Naive Bayes",
    "knn_cosine": "kNN (cosine)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

H1_BASELINE_MODEL = "logistic_regression"
H1_ALTERNATIVES = [model for model in MODEL_ORDER if model != H1_BASELINE_MODEL]
VALID_FEATURE_SETS = {"FS1", "FS2", "FS3", "FS4", "FS5"}

MODEL_SELECTION_PATH = MODELING_TABLE_DIR / "03_modeling_model_selection.csv"
FEATURE_SET_SELECTION_PATH = MODELING_TABLE_DIR / "03_modeling_feature_set_selection.csv"
MODELING_MANIFEST_PATH = MODELING_TABLE_DIR / "03_modeling_manifest.json"
FITTED_MODEL_METADATA_PATH = MODELING_MODEL_DIR / "03_fitted_model_metadata.json"
SPLIT_PATH = MODELING_CACHE_DIR / "03_split_assignments.parquet"
SENTIMENT_PATH = MODELING_CACHE_DIR / "03_sentiment_features.csv"
SENTIMENT_META_PATH = MODELING_CACHE_DIR / "03_sentiment_features_metadata.json"

HOLDOUT_RESULTS_PATH = TABLE_DIR / "04_evaluation_holdout_results.csv"
HOLDOUT_CLASS_RESULTS_PATH = TABLE_DIR / "04_evaluation_holdout_class_results.csv"
H1_RESULTS_PATH = TABLE_DIR / "04_evaluation_h1_pairwise_results.csv"
FINAL_INTERVALS_PATH = TABLE_DIR / "04_evaluation_final_model_intervals.csv"
MANIFEST_PATH = TABLE_DIR / "04_evaluation_manifest.json"

HOLDOUT_PREDICTIONS_PATH = CACHE_DIR / "04_holdout_predictions.parquet"
HOLDOUT_PREDICTIONS_META_PATH = CACHE_DIR / "04_holdout_predictions_metadata.json"
BOOTSTRAP_DISTRIBUTIONS_PATH = CACHE_DIR / "04_bootstrap_distributions.parquet"
BOOTSTRAP_META_PATH = CACHE_DIR / "04_bootstrap_metadata.json"

COMMON_FEATURE_FIGURE_PATH = FIGURE_DIR / "04_common_feature_set_holdout_macro_f1.png"
H1_FIGURE_PATH = FIGURE_DIR / "04_H1_pairwise_macro_f1_differences.png"
CONFUSION_FIGURE_PATH = FIGURE_DIR / "04_final_model_confusion_matrix.png"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


# %% 02 - Define helpers

def fmt_int(value):
    return f"{int(value):,}".replace(",", ".")


def section(title):
    print(f"\n## {title}\n")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def atomic_write(path, writer):
    temp = path.with_name(path.name + ".tmp")
    try:
        writer(temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def save_json(payload, path):
    atomic_write(
        path,
        lambda temp: temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        ),
    )


def save_csv(frame, path):
    atomic_write(path, lambda temp: frame.to_csv(temp, index=False))


def save_parquet(frame, path):
    atomic_write(path, lambda temp: frame.to_parquet(temp, index=False))


def require_paths(paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required frozen artifacts are missing:\n" + "\n".join(missing))


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Unexpected {label}: {actual!r} != {expected!r}")


def pipeline_key(feature_set, model):
    return f"{feature_set}__{model}"


def percentile_interval(values):
    return np.quantile(np.asarray(values, dtype=float), [ALPHA / 2, 1 - ALPHA / 2])


def confusion_from_codes(actual, predicted):
    n_classes = len(CLASS_ORDER)
    flat = np.bincount(
        np.asarray(actual, dtype=np.int8) * n_classes + np.asarray(predicted, dtype=np.int8),
        minlength=n_classes ** 2,
    )
    return flat.reshape(n_classes, n_classes)


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=float)
    tp = np.diag(matrix)
    predicted_count = matrix.sum(axis=0)
    actual_count = matrix.sum(axis=1)
    precision = np.divide(tp, predicted_count, out=np.zeros_like(tp), where=predicted_count > 0)
    recall = np.divide(tp, actual_count, out=np.zeros_like(tp), where=actual_count > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": tp.sum() / matrix.sum(),
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
        "weighted_f1": np.average(f1, weights=actual_count),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual_count,
    }


def batch_metrics_from_flat_confusions(flat_confusions):
    n_classes = len(CLASS_ORDER)
    matrices = np.asarray(flat_confusions, dtype=float).reshape(-1, n_classes, n_classes)
    tp = np.diagonal(matrices, axis1=1, axis2=2)
    predicted_count = matrices.sum(axis=1)
    actual_count = matrices.sum(axis=2)
    precision = np.divide(tp, predicted_count, out=np.zeros_like(tp), where=predicted_count > 0)
    recall = np.divide(tp, actual_count, out=np.zeros_like(tp), where=actual_count > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": tp.sum(axis=1) / matrices.sum(axis=(1, 2)),
        "macro_precision": precision.mean(axis=1),
        "macro_recall": recall.mean(axis=1),
        "macro_f1": f1.mean(axis=1),
        "weighted_f1": (f1 * actual_count).sum(axis=1) / actual_count.sum(axis=1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def centered_bootstrap_p_values(distribution, observed):
    distribution = np.asarray(distribution, dtype=float)
    null_distribution = distribution - distribution.mean()
    denominator = len(null_distribution) + 1
    p_one_sided = (1 + np.sum(null_distribution >= observed)) / denominator
    p_two_sided = (1 + np.sum(np.abs(null_distribution) >= abs(observed))) / denominator
    return float(p_one_sided), float(min(1.0, p_two_sided))


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    sorted_values = p_values[order]
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, sorted_values * np.arange(len(sorted_values), 0, -1))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def save_png(figure, path):
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


# %% 03 - Validate frozen Modeling provenance

require_paths([
    DATA_PATH,
    MODEL_SELECTION_PATH,
    FEATURE_SET_SELECTION_PATH,
    MODELING_MANIFEST_PATH,
    FITTED_MODEL_METADATA_PATH,
    SPLIT_PATH,
    SENTIMENT_PATH,
    SENTIMENT_META_PATH,
])

DATA_SHA256 = sha256_file(DATA_PATH)
MODELING_MANIFEST_SHA256 = sha256_file(MODELING_MANIFEST_PATH)
modeling_manifest = load_json(MODELING_MANIFEST_PATH)

expected_manifest = {
    "script_build": EXPECTED_MODELING_BUILD,
    "run_id": EXPECTED_MODELING_RUN_ID,
    "data_sha256": DATA_SHA256,
    "group_column": GROUP_COLUMN,
    "exact_group_column": EXACT_GROUP_COLUMN,
    "prepared_records": EXPECTED_RECORDS,
    "semantic_groups": EXPECTED_SEMANTIC_GROUPS,
    "semantic_conflict_groups": EXPECTED_SEMANTIC_CONFLICT_GROUPS,
    "training_records": EXPECTED_TRAINING_RECORDS,
    "training_semantic_groups": EXPECTED_TRAINING_GROUPS,
    "holdout_records": EXPECTED_HOLDOUT_RECORDS,
    "holdout_semantic_groups": EXPECTED_HOLDOUT_GROUPS,
}
for field, expected in expected_manifest.items():
    require_equal(modeling_manifest.get(field), expected, f"Modeling manifest field {field}")

if modeling_manifest.get("strict_data_freeze") is not True:
    raise ValueError("Modeling manifest does not confirm strict data freeze.")

for field in [
    "holdout_scored_in_modeling",
    "holdout_used_for_parameter_tuning",
    "holdout_used_for_model_selection",
    "holdout_used_for_preprocessing",
    "holdout_used_for_fitted_preprocessing",
]:
    if modeling_manifest.get(field) is not False:
        raise ValueError(f"Modeling manifest does not confirm hold-out exclusion: {field}")

require_equal(modeling_manifest.get("models"), MODEL_ORDER, "Modeling model order")

common_selection_manifest = modeling_manifest.get("common_feature_set_selection", {})
H1_FEATURE_SET = str(common_selection_manifest.get("selected_feature_set", "")).strip()
if H1_FEATURE_SET not in VALID_FEATURE_SETS:
    raise ValueError(
        "Modeling manifest does not contain a valid CV-selected common feature set."
    )

h1_manifest = modeling_manifest.get("hypothesis_operationalization", {}).get("H1", {})
require_equal(h1_manifest.get("feature_set"), H1_FEATURE_SET, "H1 feature set")
require_equal(h1_manifest.get("baseline"), H1_BASELINE_MODEL, "H1 baseline")
require_equal(set(h1_manifest.get("alternatives", [])), set(H1_ALTERNATIVES), "H1 alternatives")
if h1_manifest.get("pairwise") is not True or h1_manifest.get("family_mean_comparison") is not False:
    raise ValueError("Modeling manifest contains an incompatible H1 comparison design.")
if "holm" not in str(h1_manifest.get("multiple_testing", "")).lower():
    raise ValueError("Modeling manifest does not document Holm correction for H1.")

h2_manifest = modeling_manifest.get("hypothesis_operationalization", {}).get("H2", {})
require_equal(h2_manifest.get("feature_set"), H1_FEATURE_SET, "H2 common feature set")

feature_set_selection = pd.read_csv(FEATURE_SET_SELECTION_PATH)
required_feature_selection_columns = {
    "feature_set",
    "mean_macro_f1_across_models",
    "mean_weighted_f1_across_models",
    "mean_accuracy_across_models",
    "selected_for_common_comparison",
}
if not required_feature_selection_columns.issubset(feature_set_selection.columns):
    raise ValueError("Feature-set selection table has an incompatible schema.")

selected_mask = (
    feature_set_selection["selected_for_common_comparison"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin({"true", "1"})
)
selected_feature_rows = feature_set_selection.loc[selected_mask]
if len(selected_feature_rows) != 1:
    raise ValueError("Feature-set selection table must contain exactly one selected row.")

require_equal(
    str(selected_feature_rows.iloc[0]["feature_set"]),
    H1_FEATURE_SET,
    "CV-selected common feature set",
)

manifest_final = modeling_manifest.get("final_model", {})
FINAL_MODEL_KEY = (
    str(manifest_final.get("feature_set", "")).strip(),
    str(manifest_final.get("model", "")).strip(),
)
if FINAL_MODEL_KEY[0] not in VALID_FEATURE_SETS or FINAL_MODEL_KEY[1] not in MODEL_ORDER:
    raise ValueError("Modeling manifest does not contain a valid final overall model.")

H1_KEYS = [(H1_FEATURE_SET, model) for model in MODEL_ORDER]
EVALUATION_KEYS = list(dict.fromkeys(H1_KEYS + [FINAL_MODEL_KEY]))

artifact_hashes = modeling_manifest.get("artifact_hashes", {})
for name, path in {
    "split": SPLIT_PATH,
    "feature_set_selection": FEATURE_SET_SELECTION_PATH,
    "model_selection": MODEL_SELECTION_PATH,
    "fitted_model_metadata": FITTED_MODEL_METADATA_PATH,
    "sentiment_features": SENTIMENT_PATH,
    "sentiment_metadata": SENTIMENT_META_PATH,
}.items():
    require_equal(sha256_file(path), artifact_hashes.get(name), f"artifact hash {name}")

model_selection = pd.read_csv(MODEL_SELECTION_PATH)
required_selection_columns = {"selection", "feature_set", "model", "model_family", "model_artifact"}
if not required_selection_columns.issubset(model_selection.columns):
    raise ValueError("Model-selection table has an incompatible schema.")

final_role = model_selection.loc[
    model_selection["selection"].eq("final_overall_model"), ["feature_set", "model"]
]
if len(final_role) != 1 or tuple(final_role.iloc[0]) != FINAL_MODEL_KEY:
    raise ValueError("Model-selection table does not contain the expected final model.")

unique_models = model_selection[
    ["feature_set", "model", "model_family", "model_artifact"]
].drop_duplicates()

evaluation_models = unique_models.loc[
    [(fs, model) in set(EVALUATION_KEYS) for fs, model in zip(unique_models["feature_set"], unique_models["model"])]
].copy()

if set(zip(evaluation_models["feature_set"], evaluation_models["model"])) != set(EVALUATION_KEYS):
    raise ValueError("Frozen fitted models do not cover the complete Evaluation design.")

model_metadata = load_json(FITTED_MODEL_METADATA_PATH)
if model_metadata.get("status") != "complete":
    raise ValueError("Fitted-model metadata are not marked complete.")

metadata_models = model_metadata.get("models", {})
manifest_model_hashes = modeling_manifest.get("selected_model_hashes", {})
MODEL_HASHES = {}

for row in evaluation_models.itertuples(index=False):
    model_path = MODELING_MODEL_DIR / row.model_artifact
    if not model_path.exists():
        raise FileNotFoundError(f"Frozen model not found: {model_path}")

    current_hash = sha256_file(model_path)
    metadata_entry = metadata_models.get(row.model_artifact, {})
    require_equal(metadata_entry.get("feature_set"), row.feature_set, f"feature set {row.model_artifact}")
    require_equal(metadata_entry.get("model"), row.model, f"classifier {row.model_artifact}")
    require_equal(metadata_entry.get("sha256"), current_hash, f"metadata hash {row.model_artifact}")
    require_equal(manifest_model_hashes.get(row.model_artifact), current_hash, f"manifest hash {row.model_artifact}")
    MODEL_HASHES[row.model_artifact] = current_hash

section("Evaluation input")
print(f"Evaluation build          : {SCRIPT_BUILD}")
print(f"Modeling build            : {EXPECTED_MODELING_BUILD}")
print(f"Evaluated pipelines       : {len(EVALUATION_KEYS)}")
print(f"Common comparison set     : {H1_FEATURE_SET}")
print(f"H1 classifiers evaluated  : {len(H1_KEYS)}")
print(f"H1 pairwise contrasts     : {len(H1_ALTERNATIVES)}")
print(f"Significance level        : alpha={ALPHA:.2f}")
print(f"Bootstrap iterations      : {fmt_int(BOOTSTRAP_ITERATIONS)}")


# %% 04 - Reconstruct and validate the fixed hold-out

data = pd.read_csv(DATA_PATH, low_memory=False)
data["modeling_row_id"] = np.arange(len(data), dtype=np.int64)

required_columns = {
    "source_row_id",
    TARGET_COLUMN,
    GROUP_COLUMN,
    EXACT_GROUP_COLUMN,
    TEXT_COLUMN,
    SENTIMENT_TEXT_COLUMN,
    "type",
    "language",
    "queue",
}
missing_columns = sorted(required_columns - set(data.columns))
if missing_columns:
    raise ValueError(f"Missing prepared columns: {missing_columns}")

for column in [GROUP_COLUMN, EXACT_GROUP_COLUMN, TEXT_COLUMN, SENTIMENT_TEXT_COLUMN, "type", "language", "queue"]:
    data[column] = data[column].fillna("").astype(str).str.strip()

data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str).str.lower().str.strip()
data["source_row_id"] = pd.to_numeric(data["source_row_id"], errors="raise").astype(np.int64)

require_equal(len(data), EXPECTED_RECORDS, "prepared record count")
require_equal(data[GROUP_COLUMN].nunique(), EXPECTED_SEMANTIC_GROUPS, "semantic-group count")
require_equal(set(data[TARGET_COLUMN]), set(CLASS_ORDER), "target classes")
if data["source_row_id"].duplicated().any():
    raise ValueError("source_row_id must be unique.")

sentiment_meta = load_json(SENTIMENT_META_PATH)
sentiment = pd.read_csv(SENTIMENT_PATH)
sentiment_text_hash = stable_hash(data[SENTIMENT_TEXT_COLUMN].astype(str).tolist())
require_equal(
    modeling_manifest.get("sentiment", {}).get("text_hash"),
    sentiment_text_hash,
    "sentiment text hash",
)
require_equal(sentiment_meta.get("records"), len(data), "sentiment metadata record count")
require_equal(len(sentiment), len(data), "sentiment cache record count")

if "modeling_row_id" not in sentiment.columns:
    raise ValueError("Sentiment cache lacks modeling_row_id.")
if not np.array_equal(
    sentiment["modeling_row_id"].to_numpy(dtype=np.int64),
    data["modeling_row_id"].to_numpy(dtype=np.int64),
):
    raise ValueError("Sentiment cache row IDs do not match prepared data.")

for column in SENTIMENT_COLUMNS:
    if column not in sentiment.columns:
        raise ValueError(f"Missing frozen sentiment feature: {column}")
    values = pd.to_numeric(sentiment[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Invalid sentiment values in {column}.")
    data[column] = values.to_numpy(dtype=float)

sentiment_matrix = data[SENTIMENT_COLUMNS].to_numpy(dtype=float)
if (
    (sentiment_matrix < 0).any()
    or (sentiment_matrix > 1).any()
    or not np.allclose(sentiment_matrix.sum(axis=1), 1.0, atol=1e-4)
):
    raise ValueError("Frozen sentiment probabilities are invalid.")

data["priority_encoded"] = data[TARGET_COLUMN].map(CLASS_TO_INT).astype(np.int8)

split_assignments = pd.read_parquet(SPLIT_PATH).sort_values("modeling_row_id").reset_index(drop=True)
data = data.sort_values("modeling_row_id").reset_index(drop=True)

required_split_columns = {
    "modeling_row_id",
    "source_row_id",
    GROUP_COLUMN,
    EXACT_GROUP_COLUMN,
    TARGET_COLUMN,
    "split",
}
if not required_split_columns.issubset(split_assignments.columns):
    raise ValueError("Frozen split artifact has an incompatible schema.")

require_equal(len(split_assignments), len(data), "split record count")
if not np.array_equal(
    split_assignments["modeling_row_id"].to_numpy(dtype=np.int64),
    data["modeling_row_id"].to_numpy(dtype=np.int64),
):
    raise ValueError("Split assignments do not match prepared row order.")

for column in ["source_row_id", GROUP_COLUMN, EXACT_GROUP_COLUMN, TARGET_COLUMN]:
    if not np.array_equal(
        split_assignments[column].astype(str).to_numpy(),
        data[column].astype(str).to_numpy(),
    ):
        raise ValueError(f"Frozen split disagrees with prepared data on {column}.")

require_equal(set(split_assignments["split"]), {"training", "holdout"}, "split labels")
data["split"] = split_assignments["split"].to_numpy()

training_data = data.loc[data["split"].eq("training")].reset_index(drop=True)
holdout_data = data.loc[data["split"].eq("holdout")].reset_index(drop=True)

require_equal(len(training_data), EXPECTED_TRAINING_RECORDS, "training record count")
require_equal(training_data[GROUP_COLUMN].nunique(), EXPECTED_TRAINING_GROUPS, "training group count")
require_equal(len(holdout_data), EXPECTED_HOLDOUT_RECORDS, "hold-out record count")
require_equal(holdout_data[GROUP_COLUMN].nunique(), EXPECTED_HOLDOUT_GROUPS, "hold-out group count")

if set(training_data[GROUP_COLUMN]).intersection(holdout_data[GROUP_COLUMN]):
    raise ValueError("Semantic-group overlap between training and hold-out.")
if set(training_data[EXACT_GROUP_COLUMN]).intersection(holdout_data[EXACT_GROUP_COLUMN]):
    raise ValueError("Exact-text overlap between training and hold-out.")

section("Fixed hold-out")
print(f"Training records          : {fmt_int(len(training_data))}")
print(f"Training semantic groups  : {fmt_int(EXPECTED_TRAINING_GROUPS)}")
print(f"Hold-out records          : {fmt_int(len(holdout_data))}")
print(f"Hold-out semantic groups  : {fmt_int(EXPECTED_HOLDOUT_GROUPS)}")
print("Semantic-group overlap    : 0")
print("Exact-text overlap        : 0")


# %% 05 - Evaluate or reuse frozen hold-out predictions

evaluation_signature = stable_hash({
    "script_build": SCRIPT_BUILD,
    "modeling_manifest_sha256": MODELING_MANIFEST_SHA256,
    "data_sha256": DATA_SHA256,
    "split_sha256": sha256_file(SPLIT_PATH),
    "sentiment_sha256": sha256_file(SENTIMENT_PATH),
    "evaluated_pipelines": EVALUATION_KEYS,
    "model_hashes": MODEL_HASHES,
    "class_order": CLASS_ORDER,
})

prediction_columns = {key: f"pred_{pipeline_key(*key)}" for key in EVALUATION_KEYS}
prediction_table = holdout_data[
    ["modeling_row_id", GROUP_COLUMN, EXACT_GROUP_COLUMN, TARGET_COLUMN, "priority_encoded"]
].copy()
predictions = {}
prediction_source = "new hold-out scoring"

if REUSE_VALID_RESULTS and HOLDOUT_PREDICTIONS_PATH.exists() and HOLDOUT_PREDICTIONS_META_PATH.exists():
    try:
        cached_meta = load_json(HOLDOUT_PREDICTIONS_META_PATH)
        cached = pd.read_parquet(HOLDOUT_PREDICTIONS_PATH)
        compatible = (
            cached_meta.get("signature") == evaluation_signature
            and len(cached) == len(holdout_data)
            and np.array_equal(
                cached["modeling_row_id"].to_numpy(dtype=np.int64),
                holdout_data["modeling_row_id"].to_numpy(dtype=np.int64),
            )
            and all(column in cached.columns for column in prediction_columns.values())
            and not cached[list(prediction_columns.values())].isna().any().any()
        )
        if compatible:
            prediction_table = cached.copy()
            predictions = {
                key: prediction_table[column].to_numpy(dtype=np.int8)
                for key, column in prediction_columns.items()
            }
            prediction_source = "validated completed hold-out predictions reused"
    except (ValueError, KeyError, json.JSONDecodeError):
        pass

model_lookup = {
    (row.feature_set, row.model): row
    for row in evaluation_models.itertuples(index=False)
}

if not predictions:
    for key in EVALUATION_KEYS:
        row = model_lookup[key]
        pipeline = joblib.load(MODELING_MODEL_DIR / row.model_artifact)
        predicted = np.asarray(pipeline.predict(holdout_data), dtype=np.int8)

        if len(predicted) != len(holdout_data) or not np.isin(predicted, np.arange(len(CLASS_ORDER))).all():
            raise ValueError(f"Invalid hold-out predictions for {key}.")

        predictions[key] = predicted
        prediction_table[prediction_columns[key]] = predicted

    save_parquet(prediction_table, HOLDOUT_PREDICTIONS_PATH)
    save_json({
        "signature": evaluation_signature,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_records": len(holdout_data),
        "holdout_semantic_groups": EXPECTED_HOLDOUT_GROUPS,
        "evaluated_pipelines": [{"feature_set": fs, "model": model} for fs, model in EVALUATION_KEYS],
        "model_training_performed": False,
        "hyperparameter_tuning_performed": False,
        "model_selection_performed": False,
    }, HOLDOUT_PREDICTIONS_META_PATH)

actual = holdout_data["priority_encoded"].to_numpy(dtype=np.int8)
metric_rows, class_rows, confusions = [], [], {}

for key in EVALUATION_KEYS:
    row = model_lookup[key]
    matrix = confusion_from_codes(actual, predictions[key])
    metrics = metrics_from_confusion(matrix)
    confusions[key] = matrix
    role = "final_overall_model" if key == FINAL_MODEL_KEY else "H1_common_feature_comparison"

    metric_rows.append({
        "feature_set": row.feature_set,
        "model": row.model,
        "model_label": MODEL_LABELS[row.model],
        "model_family": row.model_family,
        "evaluation_role": role,
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
    })

    for class_index, label in enumerate(CLASS_ORDER):
        class_rows.append({
            "feature_set": row.feature_set,
            "model": row.model,
            "model_label": MODEL_LABELS[row.model],
            "evaluation_role": role,
            "priority": label,
            "precision": metrics["precision"][class_index],
            "recall": metrics["recall"][class_index],
            "f1": metrics["f1"][class_index],
            "support": int(metrics["support"][class_index]),
        })

holdout_results = pd.DataFrame(metric_rows)
holdout_class_results = pd.DataFrame(class_rows)

section("Hold-out scoring")
print(f"Prediction source         : {prediction_source}")
print(f"Evaluated pipelines       : {len(EVALUATION_KEYS)}")


# %% 06 - Run or reuse paired semantic-group bootstrap

group_codes, unique_groups = pd.factorize(holdout_data[GROUP_COLUMN], sort=True)
number_of_groups = len(unique_groups)
number_of_classes = len(CLASS_ORDER)
require_equal(number_of_groups, EXPECTED_HOLDOUT_GROUPS, "bootstrap group count")

group_confusions = {}
for key, predicted in predictions.items():
    contribution = np.zeros((number_of_groups, number_of_classes ** 2), dtype=np.int16)
    np.add.at(contribution, (group_codes, actual * number_of_classes + predicted), 1)
    group_confusions[key] = contribution

bootstrap_signature = stable_hash({
    "evaluation_signature": evaluation_signature,
    "prediction_sha256": sha256_file(HOLDOUT_PREDICTIONS_PATH),
    "iterations": BOOTSTRAP_ITERATIONS,
    "random_state": BOOTSTRAP_RANDOM_STATE,
    "sampling_unit": GROUP_COLUMN,
    "alpha": ALPHA,
})

bootstrap_distributions = None
bootstrap_source = "new paired group-aware bootstrap"

if REUSE_VALID_RESULTS and BOOTSTRAP_DISTRIBUTIONS_PATH.exists() and BOOTSTRAP_META_PATH.exists():
    try:
        cached_meta = load_json(BOOTSTRAP_META_PATH)
        cached = pd.read_parquet(BOOTSTRAP_DISTRIBUTIONS_PATH)
        required_columns = [f"macro_f1_{pipeline_key(*key)}" for key in EVALUATION_KEYS]
        required_columns += [f"h1_delta_{model}_vs_{H1_BASELINE_MODEL}" for model in H1_ALTERNATIVES]
        compatible = (
            cached_meta.get("signature") == bootstrap_signature
            and len(cached) == BOOTSTRAP_ITERATIONS
            and all(column in cached.columns for column in required_columns)
            and not cached[required_columns].isna().any().any()
        )
        if compatible:
            bootstrap_distributions = cached
            bootstrap_source = "validated completed bootstrap reused"
    except (ValueError, KeyError, json.JSONDecodeError):
        pass

if bootstrap_distributions is None:
    bootstrap = {
        f"macro_f1_{pipeline_key(*key)}": np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float32)
        for key in EVALUATION_KEYS
    }
    final_metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]

    for metric in final_metrics:
        bootstrap[f"final_{metric}"] = np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float32)
    for label in CLASS_ORDER:
        for metric in ["precision", "recall", "f1"]:
            bootstrap[f"final_{metric}_{label}"] = np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float32)
    for alternative in H1_ALTERNATIVES:
        bootstrap[f"h1_delta_{alternative}_vs_{H1_BASELINE_MODEL}"] = np.empty(
            BOOTSTRAP_ITERATIONS, dtype=np.float32
        )

    rng = np.random.default_rng(BOOTSTRAP_RANDOM_STATE)
    probabilities = np.full(number_of_groups, 1.0 / number_of_groups)

    section("Paired group-aware bootstrap")
    print(f"Sampling unit             : {GROUP_COLUMN}")
    print(f"Hold-out semantic groups  : {fmt_int(number_of_groups)}")
    print(f"Iterations                : {fmt_int(BOOTSTRAP_ITERATIONS)}")

    for start in range(0, BOOTSTRAP_ITERATIONS, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_ITERATIONS)
        counts = rng.multinomial(number_of_groups, probabilities, size=stop - start)
        batch_macro = {}

        for key in EVALUATION_KEYS:
            batch_metrics = batch_metrics_from_flat_confusions(counts @ group_confusions[key])
            bootstrap[f"macro_f1_{pipeline_key(*key)}"][start:stop] = batch_metrics["macro_f1"]
            batch_macro[key] = batch_metrics["macro_f1"]

            if key == FINAL_MODEL_KEY:
                for metric in final_metrics:
                    bootstrap[f"final_{metric}"][start:stop] = batch_metrics[metric]
                for class_index, label in enumerate(CLASS_ORDER):
                    for metric in ["precision", "recall", "f1"]:
                        bootstrap[f"final_{metric}_{label}"][start:stop] = batch_metrics[metric][:, class_index]

        baseline_values = batch_macro[(H1_FEATURE_SET, H1_BASELINE_MODEL)]
        for alternative in H1_ALTERNATIVES:
            bootstrap[f"h1_delta_{alternative}_vs_{H1_BASELINE_MODEL}"][start:stop] = (
                batch_macro[(H1_FEATURE_SET, alternative)] - baseline_values
            )

    bootstrap_distributions = pd.DataFrame(bootstrap)
    save_parquet(bootstrap_distributions, BOOTSTRAP_DISTRIBUTIONS_PATH)
    save_json({
        "signature": bootstrap_signature,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iterations": BOOTSTRAP_ITERATIONS,
        "random_state": BOOTSTRAP_RANDOM_STATE,
        "sampling_unit": GROUP_COLUMN,
        "holdout_semantic_groups": number_of_groups,
        "paired_across_models": True,
    }, BOOTSTRAP_META_PATH)

section("Bootstrap source")
print(f"Source                    : {bootstrap_source}")
print(f"Iterations                : {fmt_int(BOOTSTRAP_ITERATIONS)}")
print(f"Sampling unit             : {GROUP_COLUMN}")
print("Paired across models      : YES")


# %% 07 - Summarize hold-out performance and absolute intervals

observed = {
    (row.feature_set, row.model): row
    for row in holdout_results.itertuples(index=False)
}

for key in EVALUATION_KEYS:
    lower, upper = percentile_interval(
        bootstrap_distributions[f"macro_f1_{pipeline_key(*key)}"]
    )
    mask = holdout_results["feature_set"].eq(key[0]) & holdout_results["model"].eq(key[1])
    holdout_results.loc[mask, "macro_f1_ci_lower_95"] = lower
    holdout_results.loc[mask, "macro_f1_ci_upper_95"] = upper

holdout_results = holdout_results.sort_values(
    ["evaluation_role", "macro_f1"], ascending=[True, False], kind="mergesort"
).reset_index(drop=True)
holdout_class_results = holdout_class_results.sort_values(
    ["evaluation_role", "model", "priority"], kind="mergesort"
).reset_index(drop=True)

save_csv(holdout_results, HOLDOUT_RESULTS_PATH)
save_csv(holdout_class_results, HOLDOUT_CLASS_RESULTS_PATH)

section(f"Common feature-set hold-out performance ({H1_FEATURE_SET})")
common_holdout = holdout_results.loc[
    holdout_results["feature_set"].eq(H1_FEATURE_SET)
].sort_values("macro_f1", ascending=False)

print(common_holdout[
    [
        "model_label",
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1_ci_lower_95",
        "macro_f1_ci_upper_95",
    ]
].to_string(
    index=False,
    formatters={
        column: "{:.4f}".format
        for column in [
            "macro_f1",
            "weighted_f1",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1_ci_lower_95",
            "macro_f1_ci_upper_95",
        ]
    },
))


# %% 08 - Test H1 pairwise against Logistic Regression

baseline_key = (H1_FEATURE_SET, H1_BASELINE_MODEL)
baseline_macro_f1 = observed[baseline_key].macro_f1
h1_rows = []

for alternative in H1_ALTERNATIVES:
    key = (H1_FEATURE_SET, alternative)
    observed_delta = observed[key].macro_f1 - baseline_macro_f1
    distribution = bootstrap_distributions[
        f"h1_delta_{alternative}_vs_{H1_BASELINE_MODEL}"
    ].to_numpy(dtype=float)

    ci_lower, ci_upper = percentile_interval(distribution)
    p_one_sided, p_two_sided = centered_bootstrap_p_values(distribution, observed_delta)

    h1_rows.append({
        "feature_set": H1_FEATURE_SET,
        "baseline_model": H1_BASELINE_MODEL,
        "baseline_label": MODEL_LABELS[H1_BASELINE_MODEL],
        "alternative_model": alternative,
        "alternative_label": MODEL_LABELS[alternative],
        "baseline_macro_f1": baseline_macro_f1,
        "alternative_macro_f1": observed[key].macro_f1,
        "delta_macro_f1": observed_delta,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "lower_bound_95_one_sided": np.quantile(distribution, ALPHA),
        "p_value_one_sided_raw": p_one_sided,
        "p_value_two_sided_raw": p_two_sided,
    })

h1_results = pd.DataFrame(h1_rows)
h1_results["p_value_one_sided_holm"] = holm_adjust(h1_results["p_value_one_sided_raw"])
h1_results["p_value_two_sided_holm"] = holm_adjust(h1_results["p_value_two_sided_raw"])
h1_results["significant_higher_holm"] = (
    (h1_results["delta_macro_f1"] > 0)
    & (h1_results["p_value_one_sided_holm"] < ALPHA)
)
h1_supported = bool(h1_results["significant_higher_holm"].any())

h1_results = h1_results.sort_values(
    ["p_value_one_sided_holm", "delta_macro_f1"],
    ascending=[True, False],
    kind="mergesort",
).reset_index(drop=True)

save_csv(h1_results, H1_RESULTS_PATH)

section("H1 pairwise tests: alternative vs Logistic Regression")
print(h1_results[
    [
        "alternative_label",
        "baseline_macro_f1",
        "alternative_macro_f1",
        "delta_macro_f1",
        "ci_lower_95",
        "ci_upper_95",
        "p_value_one_sided_raw",
        "p_value_one_sided_holm",
        "significant_higher_holm",
    ]
].to_string(
    index=False,
    formatters={
        "baseline_macro_f1": "{:.4f}".format,
        "alternative_macro_f1": "{:.4f}".format,
        "delta_macro_f1": "{:+.4f}".format,
        "ci_lower_95": "{:+.4f}".format,
        "ci_upper_95": "{:+.4f}".format,
        "p_value_one_sided_raw": "{:.6f}".format,
        "p_value_one_sided_holm": "{:.6f}".format,
    },
))

# %% 09 - Summarize final overall model

final_confusion = confusions[FINAL_MODEL_KEY]
final_observed = metrics_from_confusion(final_confusion)
final_rows = []

for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
    lower, upper = percentile_interval(bootstrap_distributions[f"final_{metric}"])
    final_rows.append({
        "result_type": "overall_metric",
        "priority": "all",
        "metric": metric,
        "observed": final_observed[metric],
        "ci_lower_95": lower,
        "ci_upper_95": upper,
    })

for class_index, label in enumerate(CLASS_ORDER):
    for metric in ["precision", "recall", "f1"]:
        lower, upper = percentile_interval(
            bootstrap_distributions[f"final_{metric}_{label}"]
        )
        final_rows.append({
            "result_type": "class_metric",
            "priority": label,
            "metric": metric,
            "observed": final_observed[metric][class_index],
            "ci_lower_95": lower,
            "ci_upper_95": upper,
        })

final_intervals = pd.DataFrame(final_rows)
save_csv(final_intervals, FINAL_INTERVALS_PATH)

section("Final overall model")
print(f"Model                     : {FINAL_MODEL_KEY[0]} + {MODEL_LABELS[FINAL_MODEL_KEY[1]]}")
print(final_intervals.loc[
    final_intervals["result_type"].eq("overall_metric"),
    ["metric", "observed", "ci_lower_95", "ci_upper_95"],
].to_string(
    index=False,
    formatters={
        "observed": "{:.4f}".format,
        "ci_lower_95": "{:.4f}".format,
        "ci_upper_95": "{:.4f}".format,
    },
))

section("Final overall model by priority")
final_class_display = holdout_class_results.loc[
    holdout_class_results["feature_set"].eq(FINAL_MODEL_KEY[0])
    & holdout_class_results["model"].eq(FINAL_MODEL_KEY[1]),
    ["priority", "precision", "recall", "f1", "support"],
]
print(final_class_display.to_string(
    index=False,
    formatters={
        "precision": "{:.4f}".format,
        "recall": "{:.4f}".format,
        "f1": "{:.4f}".format,
        "support": fmt_int,
    },
))


# %% 10 - Create grayscale Evaluation figures

common_plot = common_holdout.set_index("model").reindex(MODEL_ORDER).reset_index()
positions = np.arange(len(common_plot))

figure, axis = plt.subplots(figsize=(6.7, 4.2))
axis.errorbar(
    common_plot["macro_f1"],
    positions,
    xerr=np.vstack([
        common_plot["macro_f1"] - common_plot["macro_f1_ci_lower_95"],
        common_plot["macro_f1_ci_upper_95"] - common_plot["macro_f1"],
    ]),
    fmt="o",
    color="black",
    ecolor="black",
    markersize=4.5,
    capsize=3.5,
    linewidth=1.0,
)
axis.set_yticks(positions, labels=[MODEL_LABELS[model] for model in MODEL_ORDER])
axis.invert_yaxis()
axis.set_xlabel("Hold-out Macro-F1 with 95% bootstrap interval")
axis.grid(axis="x", color="0.75", linestyle=":", linewidth=0.5)
axis.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    axis.spines[spine].set_visible(False)
axis.tick_params(axis="y", length=0)
figure.tight_layout()
save_png(figure, COMMON_FEATURE_FIGURE_PATH)

h1_plot = h1_results.set_index("alternative_model").reindex(H1_ALTERNATIVES).reset_index()
positions = np.arange(len(h1_plot))

figure, axis = plt.subplots(figsize=(6.8, 3.8))
axis.errorbar(
    h1_plot["delta_macro_f1"],
    positions,
    xerr=np.vstack([
        h1_plot["delta_macro_f1"] - h1_plot["ci_lower_95"],
        h1_plot["ci_upper_95"] - h1_plot["delta_macro_f1"],
    ]),
    fmt="o",
    color="black",
    ecolor="black",
    markersize=4.5,
    capsize=3.5,
    linewidth=1.0,
)
axis.axvline(0, color="0.35", linestyle="--", linewidth=0.9)
axis.set_yticks(positions, labels=[MODEL_LABELS[model] for model in H1_ALTERNATIVES])
axis.invert_yaxis()
axis.set_xlabel("Delta hold-out Macro-F1 vs Logistic Regression (95% bootstrap interval)")
axis.grid(axis="x", color="0.75", linestyle=":", linewidth=0.5)
axis.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    axis.spines[spine].set_visible(False)
axis.tick_params(axis="y", length=0)
figure.tight_layout()
save_png(figure, H1_FIGURE_PATH)

normalized_matrix = final_confusion / final_confusion.sum(axis=1, keepdims=True)
figure, axis = plt.subplots(figsize=(5.3, 4.5))
axis.imshow(normalized_matrix, cmap="Greys", vmin=0, vmax=1)
axis.set_xticks(np.arange(len(CLASS_ORDER)), labels=[label.capitalize() for label in CLASS_ORDER])
axis.set_yticks(np.arange(len(CLASS_ORDER)), labels=[label.capitalize() for label in CLASS_ORDER])
axis.set_xlabel("Predicted priority")
axis.set_ylabel("True priority")

for row_index in range(len(CLASS_ORDER)):
    for column_index in range(len(CLASS_ORDER)):
        value = normalized_matrix[row_index, column_index]
        axis.text(
            column_index,
            row_index,
            f"{value:.2f}",
            ha="center",
            va="center",
            color="white" if value > 0.55 else "black",
        )

figure.tight_layout()
save_png(figure, CONFUSION_FIGURE_PATH)


# %% 11 - Save compact Evaluation manifest

output_paths = {
    "holdout_results": HOLDOUT_RESULTS_PATH,
    "holdout_class_results": HOLDOUT_CLASS_RESULTS_PATH,
    "h1_pairwise_results": H1_RESULTS_PATH,
    "final_model_intervals": FINAL_INTERVALS_PATH,
    "holdout_predictions": HOLDOUT_PREDICTIONS_PATH,
    "holdout_predictions_metadata": HOLDOUT_PREDICTIONS_META_PATH,
    "bootstrap_distributions": BOOTSTRAP_DISTRIBUTIONS_PATH,
    "bootstrap_metadata": BOOTSTRAP_META_PATH,
    "common_feature_set_holdout_figure": COMMON_FEATURE_FIGURE_PATH,
    "h1_pairwise_figure": H1_FIGURE_PATH,
    "final_confusion_figure": CONFUSION_FIGURE_PATH,
}

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "script": SCRIPT_PATH.name,
    "script_sha256": sha256_file(SCRIPT_PATH),
    "script_build": SCRIPT_BUILD,
    "run_id": RUN_ID,
    "modeling_build": EXPECTED_MODELING_BUILD,
    "modeling_run_id": EXPECTED_MODELING_RUN_ID,
    "modeling_manifest_sha256": MODELING_MANIFEST_SHA256,
    "data_sha256": DATA_SHA256,
    "records": len(data),
    "semantic_groups": int(data[GROUP_COLUMN].nunique()),
    "training_records": len(training_data),
    "training_semantic_groups": int(training_data[GROUP_COLUMN].nunique()),
    "holdout_records": len(holdout_data),
    "holdout_semantic_groups": number_of_groups,
    "group_column": GROUP_COLUMN,
    "exact_group_column": EXACT_GROUP_COLUMN,
    "evaluated_pipelines": [
        {"feature_set": feature_set, "model": model}
        for feature_set, model in EVALUATION_KEYS
    ],
    "final_model": {"feature_set": FINAL_MODEL_KEY[0], "model": FINAL_MODEL_KEY[1]},
    "common_feature_set_selection": {
        "feature_set": H1_FEATURE_SET,
        "source": "Modeling training-only cross-validation",
        "selection_performed_in_evaluation": False,
        "selection_artifact": FEATURE_SET_SELECTION_PATH.name,
    },
    "h1": {
        "feature_set": H1_FEATURE_SET,
        "baseline": H1_BASELINE_MODEL,
        "alternatives": H1_ALTERNATIVES,
        "contrast": "Macro-F1(alternative) - Macro-F1(logistic_regression)",
        "pairwise": True,
        "family_mean_comparison": False,
        "directional_alternative": "delta > 0",
        "number_of_comparisons": len(H1_ALTERNATIVES),
        "multiple_testing": "Holm correction across six one-sided pairwise H1 tests",
        "alpha": ALPHA,
        "supported": h1_supported,
        "supporting_alternatives": h1_results.loc[
            h1_results["significant_higher_holm"], "alternative_model"
        ].tolist(),
    },
    "bootstrap": {
        "iterations": BOOTSTRAP_ITERATIONS,
        "random_state": BOOTSTRAP_RANDOM_STATE,
        "sampling_unit": GROUP_COLUMN,
        "paired_across_models": True,
        "interval": "two-sided percentile 95%",
        "h1_p_value": "centered paired bootstrap, one-sided",
    },
    "reuse": {
        "reuse_valid_results": REUSE_VALID_RESULTS,
        "prediction_source": prediction_source,
        "bootstrap_source": bootstrap_source,
    },
    "model_training_performed": False,
    "hyperparameter_tuning_performed": False,
    "model_selection_performed": False,
    "holdout_used_for_model_selection": False,
    "figures_grayscale_only": True,
    "input_model_hashes": MODEL_HASHES,
    "output_hashes": {name: sha256_file(path) for name, path in output_paths.items()},
}

save_json(manifest, MANIFEST_PATH)


# %% 12 - Finish

section("CRISP-DM Evaluation completed")
print(f"Evaluation build          : {SCRIPT_BUILD}")
print(f"Common comparison set     : {H1_FEATURE_SET}")
print(f"H1 classifiers evaluated  : {len(H1_KEYS)}")
print(f"H1 pairwise contrasts     : {len(H1_ALTERNATIVES)}")
print(f"Final overall model       : {FINAL_MODEL_KEY[0]} + {MODEL_LABELS[FINAL_MODEL_KEY[1]]}")
print(f"Hold-out records          : {fmt_int(len(holdout_data))}")
print(f"Hold-out semantic groups  : {fmt_int(number_of_groups)}")
print(f"Bootstrap iterations      : {fmt_int(BOOTSTRAP_ITERATIONS)}")
print(f"Prediction source         : {prediction_source}")
print(f"Bootstrap source          : {bootstrap_source}")
print(f"Manifest                  : {MANIFEST_PATH.name}")
