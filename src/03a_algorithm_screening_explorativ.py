"""03a - Training-only algorithm screening."""

# %% 00 - Load packages

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import hashlib
import importlib
import importlib.util
import json
import platform
import random
import sys

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import sklearn
import xgboost

from nltk.corpus import stopwords
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# %% 01 - Configure screening

SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name.lower() != "src":
    raise RuntimeError("Place this file in <project>/src before running it.")

PROJECT_ROOT = SCRIPT_PATH.parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_it_support_prepared.csv"
MODELING_CACHE_DIR = PROJECT_ROOT / "cache" / "03_tickets_modeling"
SPLIT_PATH = MODELING_CACHE_DIR / "03_split_assignments.parquet"
MODELING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "03a_algorithm_screening"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "03a_algorithm_screening"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SCREENING_BUILD = "2026-08-11_controlled_training_only_screening_v3"
RANDOM_STATE = 42
N_JOBS = 2

TARGET = "priority"
SEMANTIC_GROUP = "semantic_group_id"
EXACT_GROUP = "text_group_id"
CLASSICAL_TEXT = "text_clean"
TRANSFORMER_TEXT = "text"

CLASS_ORDER = ["low", "medium", "high"]
CLASS_TO_INT = {label: i for i, label in enumerate(CLASS_ORDER)}

# Frozen reference values from the final thesis split design.
EXPECTED_PREPARED_RECORDS = 28_551
EXPECTED_TRAINING_RECORDS = 22_842
EXPECTED_HOLDOUT_RECORDS = 5_709
EXPECTED_TRAINING_SEMANTIC_GROUPS = 16_661
EXPECTED_HOLDOUT_SEMANTIC_GROUPS = 4_164

# Shared screening sample: two of five training-only group folds.
OUTER_N_SPLITS = 5
SELECTED_SCREEN_FOLDS = (0, 1)

# Common classical representation.
TFIDF_MAX_FEATURES = 30_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
SELECT_PERCENTILE = 50
USE_BILINGUAL_STOPWORDS = True

# Equal small screening budget.
SEARCH_BUDGET = 2

# Candidate families.
RUN_LIGHTGBM = True
RUN_CATBOOST = True
RUN_DISTILMBERT = True
REQUIRE_ALL_CONFIGURED_MODELS = True

# Transformer screening configuration.
DISTILMBERT_MODEL_ID = "distilbert/distilbert-base-multilingual-cased"
DISTILMBERT_MODEL_REVISION = "45c032ab32cc946ad88a166f7cb282f58c753c2e"
DISTILMBERT_MAX_LENGTH = 160
DISTILMBERT_BATCH_SIZE = 8
DISTILMBERT_EPOCHS = 1
DISTILMBERT_LEARNING_RATES = (2e-5, 3e-5)
DISTILMBERT_WEIGHT_DECAY = 0.01
DISTILMBERT_GRAD_CLIP = 1.0
DISTILMBERT_FREEZE_LOWER_LAYERS = 0

SUMMARY_PATH = TABLE_DIR / "03a_algorithm_screening_summary.csv"
CANDIDATE_PATH = TABLE_DIR / "03a_algorithm_screening_candidate_results.csv"
FOLD_PATH = TABLE_DIR / "03a_algorithm_screening_fold_results.csv"
CLASS_PATH = TABLE_DIR / "03a_algorithm_screening_class_results.csv"
POOL_PATH = TABLE_DIR / "03a_algorithm_screening_pool.csv"
FAILED_PATH = TABLE_DIR / "03a_algorithm_screening_failed_evaluations.csv"
MANIFEST_PATH = TABLE_DIR / "03a_algorithm_screening_manifest.json"
FIGURE_PATH = FIGURE_DIR / "03a_algorithm_screening_macro_f1.png"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


# %% 02 - Helpers

def section(title):
    print(f"\n## {title}\n")


def fmt_int(value):
    return f"{int(value):,}".replace(",", ".")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(payload, path):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def module_version(name):
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def bilingual_stopwords():
    try:
        words = set(stopwords.words("english")) | set(stopwords.words("german"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("english")) | set(stopwords.words("german"))
    return sorted(words)


def metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_ORDER)),
        zero_division=0,
    )
    overall = {
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
    }
    per_class = pd.DataFrame({
        "priority": CLASS_ORDER,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    return overall, per_class


def common_pipeline(estimator, stop_words):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            max_features=TFIDF_MAX_FEATURES,
            stop_words=stop_words if USE_BILINGUAL_STOPWORDS else None,
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("selector", SelectPercentile(chi2, percentile=SELECT_PERCENTILE)),
        ("classifier", estimator),
    ])


def compact_params(params):
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)


def cid(model, number):
    return f"{model}__candidate_{number:02d}"


def require_optional_packages():
    required = []
    if RUN_LIGHTGBM:
        required.append("lightgbm")
    if RUN_CATBOOST:
        required.append("catboost")
    if RUN_DISTILMBERT:
        required += ["torch", "transformers"]

    missing = [p for p in required if importlib.util.find_spec(p) is None]

    if missing and REQUIRE_ALL_CONFIGURED_MODELS:
        raise RuntimeError(
            "Complete screening requested, but packages are missing: "
            + ", ".join(missing)
            + "\nInstall them in the active environment before rerunning, e.g.:\n"
            + "python -m pip install "
            + " ".join(missing)
        )
    return missing


# %% 03 - Preflight

missing_optional = require_optional_packages()

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Prepared data not found: {DATA_PATH}")



# %% 04 - Load data and reconstruct frozen training/hold-out split

data = pd.read_csv(DATA_PATH, low_memory=False)
data["modeling_row_id"] = np.arange(len(data), dtype=np.int64)

required_columns = {
    "source_row_id",
    TARGET,
    SEMANTIC_GROUP,
    EXACT_GROUP,
    CLASSICAL_TEXT,
    TRANSFORMER_TEXT,
    "type",
    "language",
    "queue",
}
missing_columns = sorted(required_columns - set(data.columns))
if missing_columns:
    raise ValueError(f"Prepared data are missing columns: {missing_columns}")

for column in [
    SEMANTIC_GROUP, EXACT_GROUP, CLASSICAL_TEXT, TRANSFORMER_TEXT,
    "type", "language", "queue",
]:
    data[column] = data[column].fillna("").astype(str).str.strip()

data[TARGET] = data[TARGET].astype(str).str.lower().str.strip()
data["source_row_id"] = pd.to_numeric(
    data["source_row_id"], errors="raise"
).astype(np.int64)

if set(data[TARGET]) != set(CLASS_ORDER):
    raise ValueError(
        f"Expected classes {CLASS_ORDER}; observed {sorted(data[TARGET].unique())}."
    )
if len(data) != EXPECTED_PREPARED_RECORDS:
    raise ValueError(
        f"Unexpected prepared record count: {len(data)} != {EXPECTED_PREPARED_RECORDS}."
    )
if data["modeling_row_id"].duplicated().any():
    raise ValueError("modeling_row_id must be unique.")
if data["source_row_id"].duplicated().any():
    raise ValueError("source_row_id must be unique.")

data["priority_encoded"] = data[TARGET].map(CLASS_TO_INT).astype(np.int8)
data = data.sort_values("modeling_row_id").reset_index(drop=True)

required_split_columns = {
    "modeling_row_id", "source_row_id", SEMANTIC_GROUP, EXACT_GROUP, TARGET, "split"
}

if SPLIT_PATH.exists():
    split = (
        pd.read_parquet(SPLIT_PATH)
        .sort_values("modeling_row_id")
        .reset_index(drop=True)
    )
    split_source = "validated existing split"
else:
    splitter = StratifiedGroupKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    training_index, _ = next(
        splitter.split(
            data,
            data["priority_encoded"],
            groups=data[SEMANTIC_GROUP],
        )
    )

    split = data[[
        "modeling_row_id",
        "source_row_id",
        SEMANTIC_GROUP,
        EXACT_GROUP,
        TARGET,
    ]].copy()
    split["split"] = "holdout"
    split.loc[training_index, "split"] = "training"
    split.to_parquet(SPLIT_PATH, index=False)
    split_source = "new deterministic split"

if not required_split_columns.issubset(split.columns):
    raise ValueError("Split artifact has an incompatible schema.")
if split["modeling_row_id"].duplicated().any():
    raise ValueError("Split artifact contains duplicate modeling_row_id values.")
if set(split["split"]) != {"training", "holdout"}:
    raise ValueError("Unexpected split labels.")
if len(split) != len(data):
    raise ValueError("Split assignments do not cover the prepared dataset.")
if not np.array_equal(
    split["modeling_row_id"].to_numpy(dtype=np.int64),
    data["modeling_row_id"].to_numpy(dtype=np.int64),
):
    raise ValueError("Split row order does not match prepared data.")

for column in ["source_row_id", SEMANTIC_GROUP, EXACT_GROUP, TARGET]:
    if not np.array_equal(
        split[column].astype(str).to_numpy(),
        data[column].astype(str).to_numpy(),
    ):
        raise ValueError(f"Split disagrees with prepared data on {column}.")

data["split"] = split["split"].to_numpy()
training = data.loc[data["split"].eq("training")].copy().reset_index(drop=True)
holdout = data.loc[data["split"].eq("holdout")].copy().reset_index(drop=True)

if set(training[SEMANTIC_GROUP]) & set(holdout[SEMANTIC_GROUP]):
    raise ValueError("Semantic-group overlap exists between training and hold-out.")
if set(training[EXACT_GROUP]) & set(holdout[EXACT_GROUP]):
    raise ValueError("Exact-text overlap exists between training and hold-out.")

frozen_counts = {
    "training_records": (len(training), EXPECTED_TRAINING_RECORDS),
    "holdout_records": (len(holdout), EXPECTED_HOLDOUT_RECORDS),
    "training_semantic_groups": (
        training[SEMANTIC_GROUP].nunique(), EXPECTED_TRAINING_SEMANTIC_GROUPS
    ),
    "holdout_semantic_groups": (
        holdout[SEMANTIC_GROUP].nunique(), EXPECTED_HOLDOUT_SEMANTIC_GROUPS
    ),
}
for label, (observed, expected) in frozen_counts.items():
    if int(observed) != int(expected):
        raise ValueError(
            f"Frozen split mismatch for {label}: observed {observed}, expected {expected}."
        )



# %% 05 - Create shared training-only screening pool

outer = StratifiedGroupKFold(
    n_splits=OUTER_N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

training["screen_outer_fold"] = -1

for fold, (_, idx) in enumerate(
    outer.split(training, training["priority_encoded"], groups=training[SEMANTIC_GROUP])
):
    training.loc[idx, "screen_outer_fold"] = fold

if (training["screen_outer_fold"] < 0).any():
    raise RuntimeError("At least one training record did not receive a screening fold.")

screen_pool = training.loc[
    training["screen_outer_fold"].isin(SELECTED_SCREEN_FOLDS)
].copy().reset_index(drop=True)

unused_training = training.loc[
    ~training["screen_outer_fold"].isin(SELECTED_SCREEN_FOLDS)
].copy()

fold_map = {source: target for target, source in enumerate(SELECTED_SCREEN_FOLDS)}
screen_pool["screen_cv_fold"] = (
    screen_pool["screen_outer_fold"].map(fold_map).astype(np.int8)
)

if set(screen_pool["modeling_row_id"]) & set(holdout["modeling_row_id"]):
    raise ValueError("Hold-out records entered the screening pool.")

if set(screen_pool[SEMANTIC_GROUP]) & set(unused_training[SEMANTIC_GROUP]):
    raise ValueError("Screening pool overlaps unused training semantic groups.")

for validation_fold in [0, 1]:
    tr = screen_pool.loc[screen_pool["screen_cv_fold"].ne(validation_fold)]
    va = screen_pool.loc[screen_pool["screen_cv_fold"].eq(validation_fold)]

    if set(tr[SEMANTIC_GROUP]) & set(va[SEMANTIC_GROUP]):
        raise ValueError(f"Semantic-group overlap inside screening fold {validation_fold}.")
    if set(tr[EXACT_GROUP]) & set(va[EXACT_GROUP]):
        raise ValueError(f"Exact-text overlap inside screening fold {validation_fold}.")

screen_pool[[
    "modeling_row_id", SEMANTIC_GROUP, EXACT_GROUP, TARGET,
    "screen_outer_fold", "screen_cv_fold",
]].to_csv(POOL_PATH, index=False, encoding="utf-8-sig")



# %% 06 - Define equal-budget classical candidate grids

stops = bilingual_stopwords()

model_specs = {
    "dummy_majority": {
        "family": "naive baseline",
        "candidates": [
            ({"strategy": "most_frequent"}, DummyClassifier(strategy="most_frequent")),
        ],
    },
    "complement_naive_bayes": {
        "family": "probabilistic",
        "candidates": [
            ({"alpha": 0.5}, ComplementNB(alpha=0.5)),
            ({"alpha": 1.0}, ComplementNB(alpha=1.0)),
        ],
    },
    "logistic_regression": {
        "family": "linear",
        "candidates": [
            ({"C": 0.3}, LogisticRegression(
                C=0.3, solver="saga", max_iter=3000, random_state=RANDOM_STATE
            )),
            ({"C": 1.0}, LogisticRegression(
                C=1.0, solver="saga", max_iter=3000, random_state=RANDOM_STATE
            )),
        ],
    },
    "linear_svm": {
        "family": "linear",
        "candidates": [
            ({"C": 0.3}, LinearSVC(C=0.3, max_iter=10000, random_state=RANDOM_STATE)),
            ({"C": 1.0}, LinearSVC(C=1.0, max_iter=10000, random_state=RANDOM_STATE)),
        ],
    },
    "decision_tree": {
        "family": "single tree",
        "candidates": [
            ({"max_depth": 20, "min_samples_leaf": 5}, DecisionTreeClassifier(
                max_depth=20, min_samples_leaf=5, random_state=RANDOM_STATE
            )),
            ({"max_depth": 40, "min_samples_leaf": 3}, DecisionTreeClassifier(
                max_depth=40, min_samples_leaf=3, random_state=RANDOM_STATE
            )),
        ],
    },
    "knn_cosine": {
        "family": "instance based",
        "candidates": [
            ({"n_neighbors": 7, "weights": "distance"}, KNeighborsClassifier(
                n_neighbors=7, weights="distance", metric="cosine",
                algorithm="brute", n_jobs=N_JOBS
            )),
            ({"n_neighbors": 15, "weights": "distance"}, KNeighborsClassifier(
                n_neighbors=15, weights="distance", metric="cosine",
                algorithm="brute", n_jobs=N_JOBS
            )),
        ],
    },
    "random_forest": {
        "family": "bagging ensemble",
        "candidates": [
            ({"n_estimators": 250, "max_depth": 30, "min_samples_leaf": 5,
              "max_samples": 0.50}, RandomForestClassifier(
                n_estimators=250, max_depth=30, min_samples_leaf=5,
                max_features="sqrt", max_samples=0.50,
                n_jobs=N_JOBS, random_state=RANDOM_STATE
            )),
            ({"n_estimators": 350, "max_depth": None, "min_samples_leaf": 3,
              "max_samples": 0.75}, RandomForestClassifier(
                n_estimators=350, max_depth=None, min_samples_leaf=3,
                max_features="sqrt", max_samples=0.75,
                n_jobs=N_JOBS, random_state=RANDOM_STATE
            )),
        ],
    },
    "adaboost": {
        "family": "boosting ensemble",
        "candidates": [
            ({"n_estimators": 100, "learning_rate": 0.5, "base_depth": 1},
             AdaBoostClassifier(
                 estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
                 n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE
             )),
            ({"n_estimators": 200, "learning_rate": 0.2, "base_depth": 2},
             AdaBoostClassifier(
                 estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
                 n_estimators=200, learning_rate=0.2, random_state=RANDOM_STATE
             )),
        ],
    },
    "xgboost": {
        "family": "gradient boosted trees",
        "candidates": [
            ({"n_estimators": 250, "max_depth": 3, "learning_rate": 0.05,
              "min_child_weight": 5, "reg_lambda": 5.0}, XGBClassifier(
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                tree_method="hist", n_estimators=250, learning_rate=0.05,
                max_depth=3, min_child_weight=5, subsample=0.8,
                colsample_bytree=0.5, reg_lambda=5.0,
                n_jobs=N_JOBS, random_state=RANDOM_STATE
            )),
            ({"n_estimators": 350, "max_depth": 5, "learning_rate": 0.05,
              "min_child_weight": 10, "reg_lambda": 10.0}, XGBClassifier(
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                tree_method="hist", n_estimators=350, learning_rate=0.05,
                max_depth=5, min_child_weight=10, subsample=0.8,
                colsample_bytree=0.5, reg_lambda=10.0,
                n_jobs=N_JOBS, random_state=RANDOM_STATE
            )),
        ],
    },
}

if RUN_LIGHTGBM:
    from lightgbm import LGBMClassifier

    model_specs["lightgbm"] = {
        "family": "gradient boosted trees",
        "candidates": [
            ({"n_estimators": 250, "num_leaves": 31, "learning_rate": 0.05},
             LGBMClassifier(
                 objective="multiclass", num_class=3, n_estimators=250,
                 learning_rate=0.05, num_leaves=31, subsample=0.8,
                 colsample_bytree=0.8, n_jobs=N_JOBS,
                 random_state=RANDOM_STATE, verbosity=-1
             )),
            ({"n_estimators": 350, "num_leaves": 63, "learning_rate": 0.03},
             LGBMClassifier(
                 objective="multiclass", num_class=3, n_estimators=350,
                 learning_rate=0.03, num_leaves=63, subsample=0.8,
                 colsample_bytree=0.8, n_jobs=N_JOBS,
                 random_state=RANDOM_STATE, verbosity=-1
             )),
        ],
    }

if RUN_CATBOOST:
    from catboost import CatBoostClassifier

    model_specs["catboost"] = {
        "family": "gradient boosted trees",
        "candidates": [
            ({"iterations": 250, "depth": 5, "learning_rate": 0.05},
             CatBoostClassifier(
                 iterations=250, depth=5, learning_rate=0.05,
                 loss_function="MultiClass", random_seed=RANDOM_STATE,
                 thread_count=N_JOBS, verbose=False, allow_writing_files=False
             )),
            ({"iterations": 350, "depth": 7, "learning_rate": 0.03},
             CatBoostClassifier(
                 iterations=350, depth=7, learning_rate=0.03,
                 loss_function="MultiClass", random_seed=RANDOM_STATE,
                 thread_count=N_JOBS, verbose=False, allow_writing_files=False
             )),
        ],
    }

for model_name, spec in model_specs.items():
    expected = 1 if model_name == "dummy_majority" else SEARCH_BUDGET
    if len(spec["candidates"]) != expected:
        raise ValueError(f"{model_name}: expected {expected} candidate(s).")

configured_algorithms = len(model_specs) + int(RUN_DISTILMBERT)
configured_candidates = sum(len(spec["candidates"]) for spec in model_specs.values())
if RUN_DISTILMBERT:
    configured_candidates += len(DISTILMBERT_LEARNING_RATES)

section("Algorithm screening started")
print(f"Dataset                   : {DATA_PATH.name}")
print(f"Prepared records          : {fmt_int(len(data))}")
print(f"Training records          : {fmt_int(len(training))}")
print(f"Hold-out records          : {fmt_int(len(holdout))}")
print(f"Screening records         : {fmt_int(len(screen_pool))}")
print(f"Screening semantic groups : {fmt_int(screen_pool[SEMANTIC_GROUP].nunique())}")
print(f"Split source              : {split_source}")
print(f"Algorithms                : {configured_algorithms}")
print(f"Candidate settings        : {configured_candidates}")


# %% 07 - Shared screening CV indices

y_all = screen_pool["priority_encoded"].to_numpy(np.int8)
fold_array = screen_pool["screen_cv_fold"].to_numpy(np.int8)

shared_splits = []
for validation_fold in [0, 1]:
    train_idx = np.flatnonzero(fold_array != validation_fold)
    valid_idx = np.flatnonzero(fold_array == validation_fold)
    shared_splits.append((train_idx, valid_idx))


# %% 08 - Classical screening

candidate_rows = []
fold_rows = []
class_rows = []
failed_rows = []

for model_name, spec in model_specs.items():
    for candidate_no, (params, estimator) in enumerate(spec["candidates"], start=1):
        candidate_id = cid(model_name, candidate_no)
        candidate_metrics = []
        candidate_fit_times = []
        failed = False

        for fold, (train_idx, valid_idx) in enumerate(shared_splits):
            pipe = common_pipeline(clone(estimator), stops)

            x_train = screen_pool.iloc[train_idx][CLASSICAL_TEXT]
            y_train = y_all[train_idx]
            x_valid = screen_pool.iloc[valid_idx][CLASSICAL_TEXT]
            y_valid = y_all[valid_idx]

            start = perf_counter()
            try:
                pipe.fit(x_train, y_train)
            except Exception as error:
                failed = True
                failed_rows.append({
                    "model": model_name,
                    "candidate_id": candidate_id,
                    "fold": fold,
                    "reason": f"{type(error).__name__}: {error}",
                })
                break

            fit_seconds = perf_counter() - start
            start = perf_counter()
            pred = np.asarray(pipe.predict(x_valid), dtype=np.int8)
            predict_seconds = perf_counter() - start

            overall, per_class = metrics(y_valid, pred)
            candidate_metrics.append(overall)
            candidate_fit_times.append(fit_seconds)

            fold_rows.append({
                "model": model_name,
                "model_family": spec["family"],
                "candidate_id": candidate_id,
                "candidate_number": candidate_no,
                "parameters": compact_params(params),
                "representation": "shared TF-IDF text-only",
                "fold": fold,
                **overall,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            })

            for row in per_class.itertuples(index=False):
                class_rows.append({
                    "model": model_name,
                    "model_family": spec["family"],
                    "candidate_id": candidate_id,
                    "candidate_number": candidate_no,
                    "fold": fold,
                    "priority": row.priority,
                    "precision": float(row.precision),
                    "recall": float(row.recall),
                    "f1": float(row.f1),
                    "support": int(row.support),
                })


        if failed:
            continue

        frame = pd.DataFrame(candidate_metrics)
        candidate_rows.append({
            "model": model_name,
            "model_family": spec["family"],
            "candidate_id": candidate_id,
            "candidate_number": candidate_no,
            "parameters": compact_params(params),
            "representation": "shared TF-IDF text-only",
            "search_budget": 1 if model_name == "dummy_majority" else SEARCH_BUDGET,
            "mean_macro_f1": frame["macro_f1"].mean(),
            "fold_sd_macro_f1": frame["macro_f1"].std(ddof=0),
            "mean_weighted_f1": frame["weighted_f1"].mean(),
            "mean_accuracy": frame["accuracy"].mean(),
            "mean_macro_precision": frame["macro_precision"].mean(),
            "mean_macro_recall": frame["macro_recall"].mean(),
            "mean_fit_seconds": float(np.mean(candidate_fit_times)),
        })


# %% 09 - Multilingual DistilBERT screening on the same folds

if RUN_DISTILMBERT:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if len(DISTILMBERT_LEARNING_RATES) != SEARCH_BUDGET:
        raise ValueError("DistilBERT candidate count must equal SEARCH_BUDGET.")

    tokenizer = AutoTokenizer.from_pretrained(
        DISTILMBERT_MODEL_ID,
        revision=DISTILMBERT_MODEL_REVISION,
        use_fast=True,
    )
    encoded_pool = tokenizer(
        screen_pool[TRANSFORMER_TEXT].tolist(),
        truncation=True,
        max_length=DISTILMBERT_MAX_LENGTH,
        padding=False,
        return_attention_mask=True,
    )

    class TicketDataset(Dataset):
        def __init__(self, encodings, labels, indices):
            self.encodings = encodings
            self.labels = np.asarray(labels, dtype=np.int64)
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, pos):
            index = int(self.indices[pos])
            item = {key: self.encodings[key][index] for key in self.encodings}
            item["labels"] = int(self.labels[index])
            return item

    def collate(batch):
        labels = torch.tensor([item.pop("labels") for item in batch], dtype=torch.long)
        encoded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        encoded["labels"] = labels
        return encoded

    for candidate_no, learning_rate in enumerate(
        DISTILMBERT_LEARNING_RATES, start=1
    ):
        model_name = "distilmbert_multilingual"
        candidate_id = cid(model_name, candidate_no)
        params = {
            "model_id": DISTILMBERT_MODEL_ID,
            "model_revision": DISTILMBERT_MODEL_REVISION,
            "learning_rate": learning_rate,
            "epochs": DISTILMBERT_EPOCHS,
            "batch_size": DISTILMBERT_BATCH_SIZE,
            "max_length": DISTILMBERT_MAX_LENGTH,
            "weight_decay": DISTILMBERT_WEIGHT_DECAY,
            "freeze_lower_layers": DISTILMBERT_FREEZE_LOWER_LAYERS,
            "loss": "standard_cross_entropy",
        }

        candidate_metrics = []
        candidate_fit_times = []
        failed = False

        for fold, (train_idx, valid_idx) in enumerate(shared_splits):
            seed = RANDOM_STATE + candidate_no * 100 + fold
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            torch.use_deterministic_algorithms(True, warn_only=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cpu":
                torch.set_num_threads(max(1, N_JOBS))

            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    DISTILMBERT_MODEL_ID,
                    revision=DISTILMBERT_MODEL_REVISION,
                    num_labels=len(CLASS_ORDER),
                    id2label={i: label for i, label in enumerate(CLASS_ORDER)},
                    label2id=CLASS_TO_INT,
                )
            except Exception as error:
                failed = True
                failed_rows.append({
                    "model": model_name,
                    "candidate_id": candidate_id,
                    "fold": fold,
                    "reason": f"{type(error).__name__}: {error}",
                })
                break

            if DISTILMBERT_FREEZE_LOWER_LAYERS > 0:
                body = getattr(model, "distilbert", None)
                if body is None or not hasattr(body, "transformer"):
                    raise RuntimeError("Unable to locate DistilBERT layers for freezing.")
                layers = body.transformer.layer
                for layer in layers[:min(DISTILMBERT_FREEZE_LOWER_LAYERS, len(layers))]:
                    for parameter in layer.parameters():
                        parameter.requires_grad = False

            model.to(device)

            train_loader = DataLoader(
                TicketDataset(encoded_pool, y_all, train_idx),
                batch_size=DISTILMBERT_BATCH_SIZE,
                shuffle=True,
                collate_fn=collate,
            )
            valid_loader = DataLoader(
                TicketDataset(encoded_pool, y_all, valid_idx),
                batch_size=DISTILMBERT_BATCH_SIZE,
                shuffle=False,
                collate_fn=collate,
            )

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=DISTILMBERT_WEIGHT_DECAY,
            )
            loss_function = torch.nn.CrossEntropyLoss()

            start = perf_counter()
            model.train()

            for epoch in range(DISTILMBERT_EPOCHS):
                for batch_no, batch in enumerate(train_loader, start=1):
                    labels = batch.pop("labels").to(device)
                    inputs = {key: value.to(device) for key, value in batch.items()}

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(**inputs).logits
                    loss = loss_function(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), DISTILMBERT_GRAD_CLIP
                    )
                    optimizer.step()


            fit_seconds = perf_counter() - start
            candidate_fit_times.append(fit_seconds)

            start = perf_counter()
            model.eval()
            predictions = []

            with torch.no_grad():
                for batch in valid_loader:
                    batch.pop("labels")
                    inputs = {key: value.to(device) for key, value in batch.items()}
                    logits = model(**inputs).logits
                    predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())

            predict_seconds = perf_counter() - start
            predictions = np.asarray(predictions, dtype=np.int8)

            overall, per_class = metrics(y_all[valid_idx], predictions)
            candidate_metrics.append(overall)

            fold_rows.append({
                "model": model_name,
                "model_family": "transformer",
                "candidate_id": candidate_id,
                "candidate_number": candidate_no,
                "parameters": compact_params(params),
                "representation": "multilingual DistilBERT text-only",
                "fold": fold,
                **overall,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            })

            for row in per_class.itertuples(index=False):
                class_rows.append({
                    "model": model_name,
                    "model_family": "transformer",
                    "candidate_id": candidate_id,
                    "candidate_number": candidate_no,
                    "fold": fold,
                    "priority": row.priority,
                    "precision": float(row.precision),
                    "recall": float(row.recall),
                    "f1": float(row.f1),
                    "support": int(row.support),
                })


            del model, optimizer, train_loader, valid_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if failed:
            continue

        frame = pd.DataFrame(candidate_metrics)
        candidate_rows.append({
            "model": model_name,
            "model_family": "transformer",
            "candidate_id": candidate_id,
            "candidate_number": candidate_no,
            "parameters": compact_params(params),
            "representation": "multilingual DistilBERT text-only",
            "search_budget": SEARCH_BUDGET,
            "mean_macro_f1": frame["macro_f1"].mean(),
            "fold_sd_macro_f1": frame["macro_f1"].std(ddof=0),
            "mean_weighted_f1": frame["weighted_f1"].mean(),
            "mean_accuracy": frame["accuracy"].mean(),
            "mean_macro_precision": frame["macro_precision"].mean(),
            "mean_macro_recall": frame["macro_recall"].mean(),
            "mean_fit_seconds": float(np.mean(candidate_fit_times)),
        })


# %% 10 - Consolidate results

candidate_results = pd.DataFrame(candidate_rows)
fold_results = pd.DataFrame(fold_rows)
class_results = pd.DataFrame(class_rows)
failed_results = pd.DataFrame(
    failed_rows,
    columns=["model", "candidate_id", "fold", "reason"],
)

if candidate_results.empty:
    raise RuntimeError("No screening candidate completed successfully.")

candidate_results = candidate_results.sort_values(
    ["model", "mean_macro_f1", "fold_sd_macro_f1", "mean_weighted_f1"],
    ascending=[True, False, True, False],
).reset_index(drop=True)

best = (
    candidate_results
    .sort_values(
        ["model", "mean_macro_f1", "fold_sd_macro_f1", "mean_weighted_f1"],
        ascending=[True, False, True, False],
    )
    .groupby("model", as_index=False, sort=False)
    .head(1)
    .sort_values(
        ["mean_macro_f1", "fold_sd_macro_f1", "mean_weighted_f1"],
        ascending=[False, True, False],
    )
    .reset_index(drop=True)
)

best.insert(0, "screening_rank", np.arange(1, len(best) + 1))

lr = best.loc[best["model"].eq("logistic_regression"), "mean_macro_f1"]
best["delta_macro_f1_vs_lr"] = (
    best["mean_macro_f1"] - float(lr.iloc[0])
    if len(lr) == 1 else np.nan
)

candidate_results.to_csv(CANDIDATE_PATH, index=False, encoding="utf-8-sig")
fold_results.to_csv(FOLD_PATH, index=False, encoding="utf-8-sig")
class_results.to_csv(CLASS_PATH, index=False, encoding="utf-8-sig")
best.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
failed_results.to_csv(FAILED_PATH, index=False, encoding="utf-8-sig")

section("Algorithm screening results")

print(
    best[[
        "screening_rank", "model", "model_family",
        "mean_macro_f1", "fold_sd_macro_f1",
        "mean_weighted_f1", "mean_accuracy",
        "delta_macro_f1_vs_lr",
    ]].to_string(
        index=False,
        formatters={
            "mean_macro_f1": "{:.4f}".format,
            "fold_sd_macro_f1": "{:.4f}".format,
            "mean_weighted_f1": "{:.4f}".format,
            "mean_accuracy": "{:.4f}".format,
            "delta_macro_f1_vs_lr": "{:+.4f}".format,
        },
    )
)

if not failed_results.empty:
    section("Failed candidate evaluations")
    print(failed_results.to_string(index=False))


# %% 11 - Save grayscale screening figure

plot_data = best.sort_values("mean_macro_f1", ascending=True)

fig, ax = plt.subplots(figsize=(8.0, max(4.2, 0.40 * len(plot_data))))
bars = ax.barh(
    plot_data["model"],
    plot_data["mean_macro_f1"],
    xerr=plot_data["fold_sd_macro_f1"],
    color="0.75",
    edgecolor="black",
    linewidth=0.6,
    capsize=2,
)
ax.bar_label(
    bars,
    labels=[f"{v:.3f}" for v in plot_data["mean_macro_f1"]],
    padding=3,
    fontsize=8,
)
ax.set_xlabel("Mean screening CV Macro-F1")
ax.set_ylabel("Algorithm")
ax.set_xlim(
    0,
    min(1.0, max(0.65, float(plot_data["mean_macro_f1"].max()) + 0.08)),
)
ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)


# %% 12 - Save reproducibility manifest

search_space = {
    model: [params for params, _ in spec["candidates"]]
    for model, spec in model_specs.items()
}
if RUN_DISTILMBERT:
    search_space["distilmbert_multilingual"] = [
        {
            "learning_rate": lr,
            "epochs": DISTILMBERT_EPOCHS,
            "batch_size": DISTILMBERT_BATCH_SIZE,
            "max_length": DISTILMBERT_MAX_LENGTH,
            "freeze_lower_layers": DISTILMBERT_FREEZE_LOWER_LAYERS,
        }
        for lr in DISTILMBERT_LEARNING_RATES
    ]

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "script": SCRIPT_PATH.name,
    "screening_build": SCREENING_BUILD,
    "purpose": "algorithm screening",
    "random_state": RANDOM_STATE,
    "data": {
        "path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(DATA_PATH),
        "prepared_records": len(data),
    },
    "frozen_split": {
        "path": str(SPLIT_PATH.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(SPLIT_PATH),
        "source": split_source,
        "training_records": len(training),
        "holdout_records": len(holdout),
        "holdout_predictions_generated": False,
        "holdout_used_for_model_selection": False,
    },
    "screening": {
        "splitter": "StratifiedGroupKFold",
        "outer_n_splits": OUTER_N_SPLITS,
        "selected_outer_folds": SELECTED_SCREEN_FOLDS,
        "screening_records": len(screen_pool),
        "screening_semantic_groups": int(screen_pool[SEMANTIC_GROUP].nunique()),
        "unused_training_records": len(unused_training),
        "shared_cv_folds": 2,
        "primary_metric": "macro_f1",
        "same_records_for_all_algorithms": True,
        "same_group_disjoint_folds_for_all_algorithms": True,
        "same_information_scope": "ticket text only",
        "search_budget_per_substantive_algorithm": SEARCH_BUDGET,
        "class_weighting_in_screening": "none",
    },
    "classical_representation": {
        "text_column": CLASSICAL_TEXT,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "ngram_range": TFIDF_NGRAM_RANGE,
        "min_df": TFIDF_MIN_DF,
        "bilingual_stopwords": USE_BILINGUAL_STOPWORDS,
        "feature_selection": f"chi2 SelectPercentile {SELECT_PERCENTILE}%",
        "fit_inside_each_fold": True,
    },
    "transformer": {
        "enabled": RUN_DISTILMBERT,
        "model_id": DISTILMBERT_MODEL_ID,
        "model_revision": DISTILMBERT_MODEL_REVISION,
        "text_column": TRANSFORMER_TEXT,
        "learning_rates": DISTILMBERT_LEARNING_RATES,
        "epochs": DISTILMBERT_EPOCHS,
        "batch_size": DISTILMBERT_BATCH_SIZE,
        "max_length": DISTILMBERT_MAX_LENGTH,
        "freeze_lower_layers": DISTILMBERT_FREEZE_LOWER_LAYERS,
        "loss": "standard cross entropy",
    },
    "search_space": search_space,
    "package_versions": {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": module_version("lightgbm"),
        "catboost": module_version("catboost"),
        "torch": module_version("torch"),
        "transformers": module_version("transformers"),
    },
    "successful_algorithms": best["model"].tolist(),
    "failed_evaluations": failed_results.to_dict(orient="records"),
    "outputs": {
        "summary": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
        "candidate_results": str(CANDIDATE_PATH.relative_to(PROJECT_ROOT)),
        "fold_results": str(FOLD_PATH.relative_to(PROJECT_ROOT)),
        "class_results": str(CLASS_PATH.relative_to(PROJECT_ROOT)),
        "screening_pool": str(POOL_PATH.relative_to(PROJECT_ROOT)),
        "failed_evaluations": str(FAILED_PATH.relative_to(PROJECT_ROOT)),
        "figure": str(FIGURE_PATH.relative_to(PROJECT_ROOT)),
    },
}

save_json(manifest, MANIFEST_PATH)

section("Algorithm screening completed")
print(f"Successful algorithms     : {len(best)}")
print(f"Failed evaluations        : {len(failed_results)}")
print(f"Best screening model      : {best.iloc[0]['model']}")
print(f"Best mean CV Macro-F1     : {best.iloc[0]['mean_macro_f1']:.4f}")
print(f"Summary                   : {SUMMARY_PATH.name}")
print(f"Figure                    : {FIGURE_PATH.name}")
print(f"Manifest                  : {MANIFEST_PATH.name}")
