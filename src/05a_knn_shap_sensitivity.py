"""kNN SHAP sensitivity analysis."""

# %% 00 - Load packages

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
import shap


# %% 01 - Configure

SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name.lower() != "src":
    raise RuntimeError(
        "Place this file in the existing <project>/src folder before running it. "
        f"Current location: {SCRIPT_PATH}"
    )
PROJECT_ROOT = SCRIPT_PATH.parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_it_support_prepared.csv"

MODELING_TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "03_tickets_modeling"
MODELING_MODEL_DIR = PROJECT_ROOT / "models" / "03_tickets_modeling"

XAI_TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "05_explainability"
XAI_CACHE_DIR = PROJECT_ROOT / "cache" / "05_explainability"

TABLE_DIR = XAI_TABLE_DIR
CACHE_DIR = XAI_CACHE_DIR / "05a_knn_sensitivity"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_BUILD = "2026-08-16_knn_shap_sensitivity_common_feature_selection_v2"
EXPECTED_XAI_BUILD = "2026-08-16_explainability_common_feature_selection_v1"
EXPECTED_XAI_RUN_ID = "explainability_common_feature_selection_v1"
EXPECTED_MAIN_KNN_BACKGROUND_PER_CLASS = 3
EXPECTED_MAIN_KNN_CYCLES = 3
REUSE_VALID_RESULTS = True

RANDOM_STATE = 42
SENSITIVITY_GROUPS_PER_CLASS = 30
TOP_K = 10
MASS_THRESHOLD = 0.80
SHAP_ZERO_TOLERANCE = 1e-12

TARGET_COLUMN = "priority"
GROUP_COLUMN = "semantic_group_id"
CLASS_ORDER = ["low", "medium", "high"]

MODEL_SELECTION_PATH = MODELING_TABLE_DIR / "03_modeling_model_selection.csv"
XAI_SAMPLE_PATH = XAI_TABLE_DIR / "05_xai_sample.csv"
XAI_MANIFEST_PATH = XAI_TABLE_DIR / "05_explainability_manifest.json"
LINEAR_BACKGROUND_PATH = XAI_CACHE_DIR / "05_linear_shap_background_sample.parquet"
MAIN_KNN_BACKGROUND_PATH = XAI_CACHE_DIR / "05_knn_permutation_background_sample.parquet"

SAMPLE_PATH = TABLE_DIR / "05a_knn_shap_sensitivity_sample.csv"
SUMMARY_PATH = TABLE_DIR / "05a_knn_shap_sensitivity_summary.csv"
COMPARISON_PATH = TABLE_DIR / "05a_knn_shap_sensitivity_comparisons.csv"
MANIFEST_PATH = TABLE_DIR / "05a_knn_shap_sensitivity_manifest.json"

CONFIGS = []  # Populated after validating the current Explainability manifest.


# %% 02 - Helpers

def fmt_int(value):
    return f"{int(value):,}".replace(",", ".")


def print_section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload, path):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload):
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_csr(matrix):
    if sp.issparse(matrix):
        output = matrix.tocsr().astype(np.float32)
    else:
        output = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    if not np.isfinite(output.data).all():
        raise ValueError("Non-finite transformed values.")
    return output


def compactness_metrics(vector):
    values = np.asarray(vector, dtype=np.float64)
    total = float(values.sum())

    if total <= SHAP_ZERO_TOLERANCE:
        return {
            "top_10_share": np.nan,
            "k_80": np.nan,
            "normalized_entropy": np.nan,
        }

    shares = np.sort(values)[::-1] / total
    cumulative = np.cumsum(shares)
    positive = shares[shares > 0]

    return {
        "top_10_share": float(shares[:TOP_K].sum()),
        "k_80": int(np.searchsorted(cumulative, MASS_THRESHOLD) + 1),
        "normalized_entropy": (
            -float(np.sum(positive * np.log(positive))) / np.log(len(values))
            if len(values) > 1
            else 0.0
        ),
    }


def build_background(reference_ids, background_pool, per_class, reference_per_class):
    """Build a deterministic class-balanced background around the current 05 reference."""
    if per_class < 1:
        raise ValueError("Background groups per class must be at least one.")

    rng = np.random.default_rng(RANDOM_STATE + 5_000 + per_class)
    parts = []

    for label in CLASS_ORDER:
        reference_class = (
            reference_ids.loc[reference_ids[TARGET_COLUMN].eq(label)]
            .sort_values("modeling_row_id")
            .reset_index(drop=True)
        )
        if len(reference_class) != reference_per_class:
            raise ValueError(
                f"Current 05 background for {label} has {len(reference_class)} rows; "
                f"expected {reference_per_class}."
            )

        if per_class <= reference_per_class:
            # Keep a deterministic nested subset of the current 05 background.
            class_rows = reference_class.iloc[:per_class].copy()
        else:
            pool_class = (
                background_pool.loc[
                    background_pool[TARGET_COLUMN].eq(label)
                    & ~background_pool["modeling_row_id"].isin(
                        reference_class["modeling_row_id"]
                    )
                ]
                .sort_values("modeling_row_id")
                .reset_index(drop=True)
            )
            needed = per_class - reference_per_class
            if len(pool_class) < needed:
                raise ValueError(f"Insufficient background rows for class {label}.")
            selected = pool_class.iloc[
                rng.choice(len(pool_class), size=needed, replace=False)
            ]
            class_rows = pd.concat(
                [reference_class, selected],
                ignore_index=True,
            )

        parts.append(class_rows)

    output = (
        pd.concat(parts, ignore_index=True)
        .sort_values([TARGET_COLUMN, "modeling_row_id"])
        .reset_index(drop=True)
    )

    expected = {label: per_class for label in CLASS_ORDER}
    actual = (
        output.groupby(TARGET_COLUMN)
        .size()
        .reindex(CLASS_ORDER)
        .to_dict()
    )
    if actual != expected:
        raise ValueError("Background class balance mismatch.")
    if output["modeling_row_id"].duplicated().any():
        raise ValueError("Background contains duplicate modeling rows.")
    if GROUP_COLUMN in output.columns and output[GROUP_COLUMN].duplicated().any():
        raise ValueError("Background contains duplicate semantic groups.")

    return output


def calculate_knn_permutation_shap(
    estimator,
    analysis_matrix,
    background_matrix,
    cycles,
):
    estimated_dense_bytes = (
        analysis_matrix.shape[0] * analysis_matrix.shape[1]
        + background_matrix.shape[0] * background_matrix.shape[1]
    ) * np.dtype(np.float32).itemsize
    if estimated_dense_bytes > 2_000_000_000:
        raise MemoryError(
            "Dense kNN SHAP conversion would exceed the 2 GB safety budget."
        )

    analysis_dense = np.asarray(
        analysis_matrix.toarray(),
        dtype=np.float32,
    )
    background_dense = np.asarray(
        background_matrix.toarray(),
        dtype=np.float32,
    )

    n_samples, n_features = analysis_dense.shape
    n_classes = len(CLASS_ORDER)

    def probability_function(masked_matrix):
        return np.asarray(
            estimator.predict_proba(
                sp.csr_matrix(masked_matrix)
            ),
            dtype=np.float64,
        )

    explainer = shap.PermutationExplainer(
        probability_function,
        background_dense,
        seed=RANDOM_STATE,
    )

    values = np.zeros(
        (n_samples, n_classes, n_features),
        dtype=np.float32,
    )
    base_values = np.zeros(
        (n_samples, n_classes),
        dtype=np.float64,
    )

    started = time.perf_counter()

    for row_index in range(n_samples):
        row = analysis_dense[row_index]

        varying = np.any(
            ~np.isclose(
                background_dense,
                row[None, :],
                rtol=0.0,
                atol=1e-12,
            ),
            axis=0,
        )
        varying_features = int(varying.sum())

        if varying_features == 0:
            base_values[row_index] = probability_function(
                row.reshape(1, -1)
            )[0]
            continue

        max_evals = cycles * (2 * varying_features + 1)

        explanation = explainer(
            row.reshape(1, -1),
            max_evals=max_evals,
            batch_size=64,
            silent=True,
        )

        row_values = np.asarray(
            explanation.values[0],
            dtype=np.float64,
        )

        if row_values.shape == (n_features, n_classes):
            row_values = row_values.T
        elif row_values.shape != (n_classes, n_features):
            raise ValueError(
                f"Unexpected SHAP output shape: {row_values.shape}"
            )

        values[row_index] = row_values.astype(np.float32)
        base_values[row_index] = np.asarray(
            explanation.base_values[0],
            dtype=np.float64,
        )

        if (
            (row_index + 1) % 15 == 0
            or row_index + 1 == n_samples
        ):
            print(
                f"Progress                  : "
                f"{row_index + 1}/{n_samples}"
            )

    elapsed = time.perf_counter() - started

    probabilities = np.asarray(
        estimator.predict_proba(analysis_matrix),
        dtype=np.float64,
    )
    reconstructed = (
        base_values
        + values.astype(np.float64).sum(axis=2)
    )
    max_error = float(
        np.max(np.abs(reconstructed - probabilities))
    )

    if max_error > 0.0001:
        raise ValueError(
            "kNN SHAP additivity check failed: "
            f"max error={max_error:.8f}, tolerance=0.00010000."
        )

    predictions = np.asarray(
        estimator.predict(analysis_matrix),
        dtype=np.int8,
    )
    reconstructed_predictions = (
        reconstructed.argmax(axis=1).astype(np.int8)
    )

    if not np.array_equal(
        predictions,
        reconstructed_predictions,
    ):
        raise ValueError(
            "Reconstructed SHAP outputs do not reproduce kNN predictions."
        )

    return values, {
        "elapsed_seconds": elapsed,
        "max_additivity_error": max_error,
        "prediction_reconstruction_matches": True,
    }


# %% 03 - Validate inputs

required_paths = [
    DATA_PATH,
    MODEL_SELECTION_PATH,
    XAI_SAMPLE_PATH,
    XAI_MANIFEST_PATH,
    LINEAR_BACKGROUND_PATH,
    MAIN_KNN_BACKGROUND_PATH,
]
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Required files are missing:\n" + "\n".join(missing)
    )

xai_manifest = load_json(XAI_MANIFEST_PATH)
if xai_manifest.get("script_build") != EXPECTED_XAI_BUILD:
    raise ValueError("Unexpected Explainability build.")
if xai_manifest.get("run_id") != EXPECTED_XAI_RUN_ID:
    raise ValueError("Unexpected Explainability run ID.")

xai_output_hashes = xai_manifest.get("output_hashes", {})
for artifact_name, artifact_path in {
    "xai_sample": XAI_SAMPLE_PATH,
    "linear_shap_background_sample": LINEAR_BACKGROUND_PATH,
    "knn_permutation_background_sample": MAIN_KNN_BACKGROUND_PATH,
}.items():
    expected_hash = xai_output_hashes.get(artifact_name)
    observed_hash = sha256_file(artifact_path)
    if expected_hash != observed_hash:
        raise ValueError(
            f"Explainability artifact hash mismatch: {artifact_name}."
        )

H2_FEATURE_SET = str(xai_manifest.get("feature_set", "")).strip()
if H2_FEATURE_SET not in {"FS1", "FS2", "FS3", "FS4", "FS5"}:
    raise ValueError(
        f"Unexpected Explainability feature set: {H2_FEATURE_SET!r}."
    )
if int(xai_manifest.get("global_h2_groups", -1)) != 300:
    raise ValueError("Unexpected global H2 group count.")

MAIN_BACKGROUND_PER_CLASS = int(
    xai_manifest.get("knn_permutation_background_groups_per_class", -1)
)
MAIN_CYCLES = int(xai_manifest.get("knn_permutation_cycles", -1))
if MAIN_BACKGROUND_PER_CLASS != EXPECTED_MAIN_KNN_BACKGROUND_PER_CLASS:
    raise ValueError(
        "Unexpected current kNN SHAP background size in Explainability: "
        f"{MAIN_BACKGROUND_PER_CLASS}."
    )
if MAIN_CYCLES != EXPECTED_MAIN_KNN_CYCLES:
    raise ValueError(
        "Unexpected current kNN SHAP permutation-cycle count in Explainability: "
        f"{MAIN_CYCLES}."
    )

CONFIGS = [
    {
        "config": "reference",
        "background_groups_per_class": MAIN_BACKGROUND_PER_CLASS,
        "cycles": MAIN_CYCLES,
        "role": "current_05_design",
    },
    {
        "config": "reduced_background",
        "background_groups_per_class": 1,
        "cycles": MAIN_CYCLES,
        "role": "background_sensitivity",
    },
    {
        "config": "reduced_cycles",
        "background_groups_per_class": MAIN_BACKGROUND_PER_CLASS,
        "cycles": 1,
        "role": "permutation_budget_sensitivity",
    },
]
expected_xai_models = [
    "logistic_regression",
    "linear_svm",
    "complement_naive_bayes",
    "knn_cosine",
    "random_forest",
    "xgboost",
    "lightgbm",
]
manifest_xai_models = xai_manifest.get(
    "h2_models",
    xai_manifest.get("h2_eligible_models"),
)
if manifest_xai_models != expected_xai_models:
    raise ValueError(
        "Unexpected all-seven XAI model set. "
        f"Found: {manifest_xai_models}"
    )

model_selection = pd.read_csv(MODEL_SELECTION_PATH)
knn_row = model_selection.loc[
    model_selection["feature_set"].eq(H2_FEATURE_SET)
    & model_selection["model"].eq("knn_cosine")
].drop_duplicates()

if len(knn_row) != 1:
    raise ValueError(f"Expected exactly one frozen {H2_FEATURE_SET} kNN model.")

model_artifact = knn_row.iloc[0]["model_artifact"]
model_path = MODELING_MODEL_DIR / model_artifact
if not model_path.exists():
    raise FileNotFoundError(f"Frozen kNN model missing: {model_path}")

expected_model_hash = xai_manifest.get("model_hashes", {}).get("knn_cosine")
if expected_model_hash != sha256_file(model_path):
    raise ValueError("Frozen kNN model hash mismatch.")

print_section("kNN SHAP sensitivity input")
print(f"Build                     : {SCRIPT_BUILD}")
print(f"Reference XAI build       : {EXPECTED_XAI_BUILD}")
print(f"Model                     : kNN (cosine), {H2_FEATURE_SET}")
print(f"Configurations            : {len(CONFIGS)}")
print(f"Groups per class          : {SENSITIVITY_GROUPS_PER_CLASS}")
print(f"Current 05 background     : {MAIN_BACKGROUND_PER_CLASS} per class")
print(f"Current 05 cycles         : {MAIN_CYCLES}")


# %% 04 - Reconstruct sensitivity sample and backgrounds

data = pd.read_csv(DATA_PATH, low_memory=False)
data["modeling_row_id"] = np.arange(len(data), dtype=np.int64)

xai_sample = pd.read_csv(XAI_SAMPLE_PATH, low_memory=False)
global_flag = xai_sample["is_global_h2_case"]
if global_flag.dtype == bool:
    global_mask = global_flag
else:
    global_mask = (
        global_flag.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    if global_mask.isna().any():
        raise ValueError("Invalid is_global_h2_case values in 05_xai_sample.csv.")

global_sample = xai_sample.loc[global_mask].copy()

rng = np.random.default_rng(RANDOM_STATE + 4_000)
sample_parts = []

for label in CLASS_ORDER:
    class_rows = global_sample.loc[
        global_sample[TARGET_COLUMN].eq(label)
    ].copy()

    if len(class_rows) < SENSITIVITY_GROUPS_PER_CLASS:
        raise ValueError(f"Insufficient global XAI rows for class {label}.")

    sample_parts.append(
        class_rows.iloc[
            rng.choice(
                len(class_rows),
                size=SENSITIVITY_GROUPS_PER_CLASS,
                replace=False,
            )
        ]
    )

sensitivity_sample = (
    pd.concat(sample_parts, ignore_index=True)
    .sort_values([TARGET_COLUMN, "modeling_row_id"])
    .reset_index(drop=True)
)

if sensitivity_sample[GROUP_COLUMN].duplicated().any():
    raise ValueError("Sensitivity sample contains duplicate semantic groups.")

expected_counts = {
    label: SENSITIVITY_GROUPS_PER_CLASS
    for label in CLASS_ORDER
}
actual_counts = (
    sensitivity_sample.groupby(TARGET_COLUMN)
    .size()
    .reindex(CLASS_ORDER)
    .to_dict()
)
if actual_counts != expected_counts:
    raise ValueError("Sensitivity sample is not class-balanced.")

sensitivity_sample[
    [
        "modeling_row_id",
        GROUP_COLUMN,
        TARGET_COLUMN,
        "priority_encoded",
    ]
].to_csv(SAMPLE_PATH, index=False)

background_pool_ids = pd.read_parquet(
    LINEAR_BACKGROUND_PATH
)
reference_background_ids = pd.read_parquet(
    MAIN_KNN_BACKGROUND_PATH
)

if (
    reference_background_ids.groupby(TARGET_COLUMN)
    .size()
    .reindex(CLASS_ORDER)
    .to_dict()
    != {label: MAIN_BACKGROUND_PER_CLASS for label in CLASS_ORDER}
):
    raise ValueError("Unexpected current 05 kNN background composition.")

required_background_ids = set(
    background_pool_ids["modeling_row_id"].astype(int)
)
if not required_background_ids.issubset(
    set(data["modeling_row_id"].astype(int))
):
    raise ValueError("Background IDs do not match prepared data.")

print_section("Sensitivity sample")
print(f"Cases                     : {fmt_int(len(sensitivity_sample))}")
print(f"Semantic groups           : {fmt_int(sensitivity_sample[GROUP_COLUMN].nunique())}")
print(f"Classes                   : 30 / 30 / 30")


# %% 05 - Load frozen kNN pipeline and transform data

pipeline = joblib.load(model_path)
preprocessor = pipeline.named_steps.get("preprocessor")
estimator = pipeline.named_steps.get("classifier")

if preprocessor is None or estimator is None:
    raise ValueError("Unexpected frozen kNN pipeline structure.")

analysis_matrix = ensure_csr(
    preprocessor.transform(sensitivity_sample)
)

stored_predictions = sensitivity_sample[
    "pred_knn_cosine"
].to_numpy(dtype=np.int8)

pipeline_predictions = np.asarray(
    pipeline.predict(sensitivity_sample),
    dtype=np.int8,
)

if not np.array_equal(
    stored_predictions,
    pipeline_predictions,
):
    raise ValueError(
        "Frozen kNN predictions differ from the Explainability sample."
    )


# %% 06 - Run or reuse sensitivity configurations

results = {}
summary_rows = []

for config in CONFIGS:
    config_name = config["config"]
    per_class = int(config["background_groups_per_class"])
    cycles = int(config["cycles"])

    print_section(f"Configuration: {config_name}")
    print(f"Background per class      : {per_class}")
    print(f"Permutation cycles        : {cycles}")

    background_ids = build_background(
        reference_background_ids,
        background_pool_ids,
        per_class,
        MAIN_BACKGROUND_PER_CLASS,
    )

    background_data = background_ids.merge(
        data,
        on="modeling_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("_id", ""),
    )

    background_matrix = ensure_csr(
        preprocessor.transform(background_data)
    )

    signature = stable_hash({
        "script_build": SCRIPT_BUILD,
        "reference_xai_build": EXPECTED_XAI_BUILD,
        "reference_xai_manifest_sha256": sha256_file(XAI_MANIFEST_PATH),
        "feature_set": H2_FEATURE_SET,
        "model_sha256": sha256_file(model_path),
        "xai_sample_sha256": sha256_file(XAI_SAMPLE_PATH),
        "sample_ids": sensitivity_sample["modeling_row_id"].tolist(),
        "background_ids": background_ids["modeling_row_id"].tolist(),
        "cycles": cycles,
        "package_shap": shap.__version__,
    })

    cache_path = CACHE_DIR / f"05a_{config_name}.joblib"
    cached = None

    if REUSE_VALID_RESULTS and cache_path.exists():
        try:
            candidate = joblib.load(cache_path)
            if candidate.get("signature") == signature:
                cached = candidate
        except Exception:
            cached = None

    if cached is None:
        values, validation = calculate_knn_permutation_shap(
            estimator,
            analysis_matrix,
            background_matrix,
            cycles,
        )

        selected_abs = np.abs(
            values[
                np.arange(len(sensitivity_sample)),
                pipeline_predictions,
                :,
            ]
        ).astype(np.float64)

        case_rows = []
        for row_index, row in enumerate(
            sensitivity_sample.itertuples(index=False)
        ):
            metrics = compactness_metrics(
                selected_abs[row_index]
            )
            case_rows.append({
                "config": config_name,
                "modeling_row_id": int(row.modeling_row_id),
                GROUP_COLUMN: getattr(row, GROUP_COLUMN),
                TARGET_COLUMN: getattr(row, TARGET_COLUMN),
                **metrics,
            })

        case_results = pd.DataFrame(case_rows)
        global_mean_abs = selected_abs.mean(axis=0)

        cached = {
            "signature": signature,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "case_results": case_results,
            "global_mean_abs": global_mean_abs.astype(np.float32),
            "validation": validation,
            "background_ids": background_ids[
                "modeling_row_id"
            ].astype(int).tolist(),
        }
        joblib.dump(cached, cache_path, compress=3)
        source = "new"
    else:
        source = "reused"

    results[config_name] = cached

    case_results = cached["case_results"]
    validation = cached["validation"]

    summary_rows.append({
        "config": config_name,
        "background_groups_per_class": per_class,
        "background_groups_total": per_class * len(CLASS_ORDER),
        "cycles": cycles,
        "cases": len(case_results),
        "top_10_share_mean": case_results["top_10_share"].mean(),
        "top_10_share_median": case_results["top_10_share"].median(),
        "k_80_mean": case_results["k_80"].mean(),
        "k_80_median": case_results["k_80"].median(),
        "normalized_entropy_mean": case_results["normalized_entropy"].mean(),
        "max_additivity_error": validation["max_additivity_error"],
        "elapsed_seconds": validation["elapsed_seconds"],
        "source": source,
    })

    print(f"Source                    : {source}")
    print(
        f"Top-10 mean               : "
        f"{case_results['top_10_share'].mean():.4f}"
    )
    print(
        f"K80 median                : "
        f"{case_results['k_80'].median():.1f}"
    )
    print(
        f"Entropy mean              : "
        f"{case_results['normalized_entropy'].mean():.4f}"
    )
    print(
        f"Max additivity error      : "
        f"{validation['max_additivity_error']:.8f}"
    )
    print(
        f"Elapsed                   : "
        f"{validation['elapsed_seconds'] / 60:.1f} min"
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)


# %% 07 - Compare configurations with the reference

reference_cases = (
    results["reference"]["case_results"]
    .set_index("modeling_row_id")
    .sort_index()
)
reference_global = np.asarray(
    results["reference"]["global_mean_abs"],
    dtype=np.float64,
)

comparison_rows = []

for config_name in [
    config["config"] for config in CONFIGS
    if config["config"] != "reference"
]:
    alternative_cases = (
        results[config_name]["case_results"]
        .set_index("modeling_row_id")
        .reindex(reference_cases.index)
    )

    if alternative_cases.isna().any().any():
        raise ValueError(
            f"Incomplete case alignment for {config_name}."
        )

    alternative_global = np.asarray(
        results[config_name]["global_mean_abs"],
        dtype=np.float64,
    )

    rank_mask = (
        (reference_global > SHAP_ZERO_TOLERANCE)
        | (alternative_global > SHAP_ZERO_TOLERANCE)
    )

    if rank_mask.sum() >= 2:
        global_rank_rho = float(
            spearmanr(
                reference_global[rank_mask],
                alternative_global[rank_mask],
            ).statistic
        )
    else:
        global_rank_rho = np.nan

    comparison_rows.append({
        "comparison": f"reference_vs_{config_name}",
        "top_10_mean_difference":
            alternative_cases["top_10_share"].mean()
            - reference_cases["top_10_share"].mean(),
        "top_10_mean_absolute_case_difference":
            np.abs(
                alternative_cases["top_10_share"]
                - reference_cases["top_10_share"]
            ).mean(),
        "top_10_case_spearman":
            spearmanr(
                reference_cases["top_10_share"],
                alternative_cases["top_10_share"],
            ).statistic,
        "k_80_median_difference":
            alternative_cases["k_80"].median()
            - reference_cases["k_80"].median(),
        "k_80_mean_absolute_case_difference":
            np.abs(
                alternative_cases["k_80"]
                - reference_cases["k_80"]
            ).mean(),
        "k_80_case_spearman":
            spearmanr(
                reference_cases["k_80"],
                alternative_cases["k_80"],
            ).statistic,
        "entropy_mean_difference":
            alternative_cases["normalized_entropy"].mean()
            - reference_cases["normalized_entropy"].mean(),
        "entropy_mean_absolute_case_difference":
            np.abs(
                alternative_cases["normalized_entropy"]
                - reference_cases["normalized_entropy"]
            ).mean(),
        "entropy_case_spearman":
            spearmanr(
                reference_cases["normalized_entropy"],
                alternative_cases["normalized_entropy"],
            ).statistic,
        "global_mean_abs_shap_rank_spearman":
            global_rank_rho,
    })

comparisons = pd.DataFrame(comparison_rows)
comparisons.to_csv(COMPARISON_PATH, index=False)

print_section("Sensitivity comparisons")
print(
    comparisons[
        [
            "comparison",
            "top_10_mean_difference",
            "k_80_median_difference",
            "entropy_mean_difference",
            "global_mean_abs_shap_rank_spearman",
        ]
    ].to_string(
        index=False,
        formatters={
            "top_10_mean_difference": "{:+.4f}".format,
            "k_80_median_difference": "{:+.1f}".format,
            "entropy_mean_difference": "{:+.4f}".format,
            "global_mean_abs_shap_rank_spearman": "{:.4f}".format,
        },
    )
)


# %% 08 - Save manifest

save_json({
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "script_build": SCRIPT_BUILD,
    "script": SCRIPT_PATH.name,
    "script_sha256": sha256_file(SCRIPT_PATH),
    "reference_xai_build": EXPECTED_XAI_BUILD,
    "reference_xai_run_id": EXPECTED_XAI_RUN_ID,
    "reference_xai_manifest_sha256": sha256_file(XAI_MANIFEST_PATH),
    "model": "knn_cosine",
    "feature_set": H2_FEATURE_SET,
    "current_05_knn_background_groups_per_class": MAIN_BACKGROUND_PER_CLASS,
    "current_05_knn_permutation_cycles": MAIN_CYCLES,
    "reference_configuration": "reference",
    "model_sha256": sha256_file(model_path),
    "sensitivity_groups_per_class": SENSITIVITY_GROUPS_PER_CLASS,
    "sensitivity_groups": len(sensitivity_sample),
    "configs": CONFIGS,
    "sample_sha256": sha256_file(SAMPLE_PATH),
    "summary_sha256": sha256_file(SUMMARY_PATH),
    "comparison_sha256": sha256_file(COMPARISON_PATH),
}, MANIFEST_PATH)

print_section("kNN SHAP sensitivity completed")
print(f"Configurations completed  : {len(CONFIGS)}")
print(f"Sensitivity groups        : {fmt_int(len(sensitivity_sample))}")
print(f"Summary                   : {SUMMARY_PATH.name}")
print(f"Comparisons               : {COMPARISON_PATH.name}")
print(f"Manifest                  : {MANIFEST_PATH.name}")
