"""CRISP-DM Modeling for multilingual IT-support ticket prioritization."""

# %% 00 - Load packages

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from time import perf_counter
import hashlib
import json
import platform
import sys

import joblib
import lightgbm
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import sklearn
import xgboost

from lightgbm import LGBMClassifier
from nltk.corpus import stopwords
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


# %% 01 - Configure final experiment

SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name.lower() != "src":
    raise RuntimeError("Place 03_modeling.py in <project>/src before running it.")

PROJECT_ROOT = SCRIPT_PATH.parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_it_support_prepared.csv"
DATA_PREPARATION_SUMMARY_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "02_tickets_data_preparation"
    / "data_preparation_summary.csv"
)
DATA_PREPARATION_COHESION_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "02_tickets_data_preparation"
    / "semantic_group_cohesion.csv"
)
SCREENING_A_MANIFEST_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "03a_algorithm_screening"
    / "03a_algorithm_screening_manifest.json"
)
SCREENING_B_MANIFEST_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "03b_representation_screening"
    / "03b_representation_screening_manifest.json"
)

RUN_ID = "final_comparative_common_feature_selection_v1"
SCRIPT_BUILD = "2026-08-16_final_comparative_common_feature_selection_v1"
ANALYTICAL_DESIGN_ID = "final_comparative_v2"
RANDOM_STATE = 42
N_SPLITS = 5
N_JOBS = 1
SEARCH_BUDGET = 12
STRICT_DATA_FREEZE = True
REUSE_VALID_RESULTS = True
RESUME_PARTIAL_RESULTS = True
FAIL_ON_STALE_MODEL_ARTIFACTS = True
MINIMUM_ALLOWED_GROUP_SIMILARITY = 0.949999

# Frozen reference values from the completed Data Preparation / split design.
EXPECTED_PREPARED_RECORDS = 28_551
EXPECTED_SEMANTIC_GROUPS = 20_825
EXPECTED_SEMANTIC_CONFLICT_GROUPS = 105
EXPECTED_TARGET_COUNTS = {"low": 5_883, "medium": 11_504, "high": 11_164}
EXPECTED_TRAINING_RECORDS = 22_842
EXPECTED_HOLDOUT_RECORDS = 5_709
EXPECTED_TRAINING_SEMANTIC_GROUPS = 16_661
EXPECTED_HOLDOUT_SEMANTIC_GROUPS = 4_164
EXPECTED_GROUPING_METHOD = "complete_linkage_within_radius_components"

TARGET = "priority"
GROUP = "semantic_group_id"
EXACT_GROUP = "text_group_id"
TEXT = "text_clean"
SENTIMENT_TEXT = "text"
CLASS_ORDER = ["low", "medium", "high"]
CLASS_TO_INT = {label: i for i, label in enumerate(CLASS_ORDER)}

# Frozen after 03b: same representation for all seven classifiers.
WORD_MAX_FEATURES = 30_000
CHAR_MAX_FEATURES = 30_000
SELECTED_TEXT_FEATURES = 15_000
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)

SENTIMENT_COLUMNS = ["sentiment_negative", "sentiment_neutral", "sentiment_positive"]
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SENTIMENT_MODEL_REVISION = "968fc69b266c3b535556f8342ecc49a146986b4d"
SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
SENTIMENT_BATCH_SIZE = 16

TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "03_tickets_modeling"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "03_tickets_modeling"
MODEL_DIR = PROJECT_ROOT / "models" / "03_tickets_modeling"
CACHE_DIR = PROJECT_ROOT / "cache" / "03_tickets_modeling"
for folder in [TABLE_DIR, FIGURE_DIR, MODEL_DIR, CACHE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

SPLIT_PATH = CACHE_DIR / "03_split_assignments.parquet"
FOLD_PATH = CACHE_DIR / "03_cv_fold_assignments.parquet"
PARAMETER_PATH = CACHE_DIR / "03_model_parameters.json"
TUNING_CHECKPOINT_PATH = CACHE_DIR / "03_tuning_checkpoint.json"
OOF_PATH = CACHE_DIR / "03_final_oof_predictions.parquet"
FOLD_METRICS_PATH = CACHE_DIR / "03_cv_metrics_by_fold.parquet"
FOLD_CLASS_METRICS_PATH = CACHE_DIR / "03_cv_class_metrics_by_fold.parquet"
SENTIMENT_PATH = CACHE_DIR / "03_sentiment_features.csv"
SENTIMENT_META_PATH = CACHE_DIR / "03_sentiment_features_metadata.json"
OOF_META_PATH = CACHE_DIR / "03_final_oof_metadata.json"

DESIGN_SUMMARY_PATH = TABLE_DIR / "03_modeling_design_summary.csv"
SEARCH_CANDIDATES_PATH = TABLE_DIR / "03_modeling_search_candidates.csv"
SEARCH_SUMMARY_PATH = TABLE_DIR / "03_modeling_search_summary.csv"
CV_RESULTS_PATH = TABLE_DIR / "03_modeling_cv_results.csv"
FEATURE_SET_SELECTION_PATH = TABLE_DIR / "03_modeling_feature_set_selection.csv"
CV_CLASS_RESULTS_PATH = TABLE_DIR / "03_modeling_cv_class_results.csv"
DUMMY_PATH = TABLE_DIR / "03_modeling_dummy_baseline.csv"
MODEL_SELECTION_PATH = TABLE_DIR / "03_modeling_model_selection.csv"
MANIFEST_PATH = TABLE_DIR / "03_modeling_manifest.json"
STOPWORD_PATH = TABLE_DIR / "03_bilingual_stopwords.txt"
MODEL_META_PATH = MODEL_DIR / "03_fitted_model_metadata.json"
FIGURE_PATH = FIGURE_DIR / "03_cv_macro_f1_matrix.png"
CLASS_F1_FIGURE_PATH = FIGURE_DIR / "03_cv_class_f1_best_feature_set.png"

FEATURE_SETS = {
    "FS1": {"categorical": [], "sentiment": False, "label": "Text"},
    "FS2": {"categorical": ["type"], "sentiment": False, "label": "Text + type"},
    "FS3": {"categorical": ["type", "language"], "sentiment": False, "label": "Text + type + language"},
    "FS4": {"categorical": ["type", "language", "queue"], "sentiment": False, "label": "Text + type + language + queue"},
    "FS5": {"categorical": ["type", "language", "queue"], "sentiment": True, "label": "Text + type + language + queue + sentiment"},
}
FEATURE_ORDER = list(FEATURE_SETS)

MODEL_FAMILIES = {
    "logistic_regression": "linear_baseline",
    "linear_svm": "linear",
    "complement_naive_bayes": "probabilistic",
    "knn_cosine": "instance_based",
    "random_forest": "bagging_ensemble",
    "xgboost": "gradient_boosting",
    "lightgbm": "gradient_boosting",
}
MODEL_ORDER = list(MODEL_FAMILIES)
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "complement_naive_bayes": "Complement Naive Bayes",
    "knn_cosine": "kNN (cosine)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

BASE_MODELS = {
    "logistic_regression": LogisticRegression(solver="saga", max_iter=3_000, random_state=RANDOM_STATE),
    "linear_svm": LinearSVC(max_iter=10_000, random_state=RANDOM_STATE),
    "complement_naive_bayes": ComplementNB(),
    "knn_cosine": KNeighborsClassifier(metric="cosine", algorithm="brute", n_jobs=1),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_features="sqrt", max_samples=0.75, n_jobs=1,
        random_state=RANDOM_STATE,
    ),
    "xgboost": XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        tree_method="hist", n_estimators=400, learning_rate=0.05,
        subsample=0.80, colsample_bytree=0.60, reg_alpha=0.10,
        n_jobs=1, random_state=RANDOM_STATE,
    ),
    "lightgbm": LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=400, learning_rate=0.05,
        subsample=0.80, colsample_bytree=0.80, n_jobs=1,
        random_state=RANDOM_STATE, verbosity=-1,
    ),
}

MODEL_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.03, 0.10, 0.30, 1.00, 3.00, 10.00],
        "classifier__class_weight": [None, "balanced"],
    },
    "linear_svm": {
        "classifier__C": [0.03, 0.10, 0.30, 1.00, 3.00, 10.00],
        "classifier__class_weight": [None, "balanced"],
    },
    "complement_naive_bayes": {
        "classifier__alpha": [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00],
    },
    "knn_cosine": {
        "classifier__n_neighbors": [5, 9, 15, 25, 40, 60],
        "classifier__weights": ["uniform", "distance"],
    },
    "random_forest": {
        "classifier__max_depth": [20, 40, None],
        "classifier__min_samples_leaf": [2, 5],
        "classifier__class_weight": [None, "balanced_subsample"],
    },
    "xgboost": {
        "classifier__max_depth": [3, 5, 7],
        "classifier__min_child_weight": [1, 5],
        "classifier__reg_lambda": [1.0, 10.0],
    },
    "lightgbm": {
        "classifier__num_leaves": [15, 31, 63],
        "classifier__min_child_samples": [10, 30],
        "classifier__reg_lambda": [0.0, 5.0],
    },
}

SCORING = {"macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "accuracy": "accuracy"}
SORT_METRICS = ["mean_macro_f1", "mean_weighted_f1", "mean_accuracy"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
})


# %% 02 - Helpers

def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def fmt_int(value):
    return f"{int(value):,}".replace(",", ".")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _atomic_replace(temp_path, target_path):
    try:
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def save_json(payload, path):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _atomic_replace(temp, path)


def save_csv(frame, path):
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    _atomic_replace(temp, path)


def save_parquet(frame, path):
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    _atomic_replace(temp, path)


def save_joblib(model, path):
    temp = path.with_name(path.name + ".tmp")
    joblib.dump(model, temp)
    _atomic_replace(temp, path)


def version_of(package):
    try:
        return package_version(package)
    except PackageNotFoundError:
        return "not installed"


def validate_preparation_artifacts(frame):
    for path in [DATA_PREPARATION_SUMMARY_PATH, DATA_PREPARATION_COHESION_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required Data Preparation artifact missing: {path}")

    summary = pd.read_csv(DATA_PREPARATION_SUMMARY_PATH, encoding="utf-8-sig")
    if not {"indicator", "value"}.issubset(summary.columns):
        raise ValueError("Data Preparation summary has an incompatible schema.")
    lookup = summary.drop_duplicates("indicator", keep="last").set_index("indicator")["value"].to_dict()

    records = pd.to_numeric(pd.Series([lookup.get("final_prepared_records")]), errors="coerce").iloc[0]
    if pd.isna(records) or int(records) != len(frame):
        raise ValueError("Prepared dataset and Data Preparation summary disagree on record count.")
    if str(lookup.get("split_control_column", "")).strip() != GROUP:
        raise ValueError("Data Preparation declares a different split-control column.")

    grouping_method = str(lookup.get("semantic_grouping_method", "")).strip()
    if not grouping_method:
        raise ValueError("Semantic grouping method is missing from Data Preparation summary.")

    cohesion = pd.read_csv(DATA_PREPARATION_COHESION_PATH)
    if cohesion.empty or "pair_similarity_min" not in cohesion.columns:
        raise ValueError("Semantic-group cohesion audit is missing or incompatible.")
    similarities = pd.to_numeric(cohesion["pair_similarity_min"], errors="coerce").dropna()
    if similarities.empty:
        raise ValueError("Semantic-group cohesion audit contains no valid similarities.")
    minimum_similarity = float(similarities.min())
    if minimum_similarity < MINIMUM_ALLOWED_GROUP_SIMILARITY:
        raise ValueError("At least one semantic group violates the frozen similarity threshold.")

    return grouping_method, minimum_similarity


def validate_screening_provenance():
    hashes, payloads = {}, {}
    for label, path in {
        "03a_manifest": SCREENING_A_MANIFEST_PATH,
        "03b_manifest": SCREENING_B_MANIFEST_PATH,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Required screening provenance artifact missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"Screening manifest is not valid JSON: {path}") from error
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"Screening manifest is empty or incompatible: {path}")
        hashes[label] = sha256(path)
        payloads[label] = payload
    return hashes, payloads


def validate_screening_decisions(payloads, data_sha256):
    screening_a = payloads["03a_manifest"]
    screening_b = payloads["03b_manifest"]

    if screening_a.get("screening_build") != "2026-08-11_controlled_training_only_screening_v3":
        raise ValueError("Unexpected 03a algorithm-screening build.")
    if screening_a.get("data", {}).get("sha256") != data_sha256:
        raise ValueError("03a screening used a different prepared dataset.")
    frozen_split = screening_a.get("frozen_split", {})
    if frozen_split.get("holdout_predictions_generated") is not False:
        raise ValueError("03a screening manifest does not confirm hold-out exclusion.")
    if frozen_split.get("holdout_used_for_model_selection") is not False:
        raise ValueError("03a screening manifest indicates hold-out model selection.")
    screening_design = screening_a.get("screening", {})
    if screening_design.get("primary_metric") != "macro_f1":
        raise ValueError("03a screening used an unexpected primary metric.")
    if screening_design.get("shared_cv_folds") != 2:
        raise ValueError("03a screening did not use the expected two shared folds.")
    if screening_a.get("failed_evaluations") not in ([], None):
        raise ValueError("03a screening contains failed evaluations.")
    if not set(MODEL_ORDER).issubset(set(screening_a.get("successful_algorithms", []))):
        raise ValueError("03a screening does not contain all retained final classifiers.")

    if screening_b.get("build") != "2026-08-11_word_char_representation_screening_v1":
        raise ValueError("Unexpected 03b representation-screening build.")
    if screening_b.get("data_sha256") != data_sha256:
        raise ValueError("03b screening used a different prepared dataset.")
    if screening_b.get("holdout_scored") is not False:
        raise ValueError("03b screening manifest does not confirm hold-out exclusion.")
    if screening_b.get("shared_cv_folds") != 2:
        raise ValueError("03b screening did not use the expected two shared folds.")
    if screening_b.get("primary_metric") != "macro_f1":
        raise ValueError("03b screening used an unexpected primary metric.")
    if int(screening_b.get("common_selected_feature_budget", -1)) != SELECTED_TEXT_FEATURES:
        raise ValueError("03b screening used a different selected-feature budget.")
    word_char = screening_b.get("representations", {}).get("word_char", {})
    if word_char.get("components") != ["word", "char"]:
        raise ValueError("03b screening does not document the frozen Word+Character representation.")


def search_diagnostics(grid, params):
    edges, natural_limits = [], []
    for parameter, values in grid.items():
        if len(values) <= 1:
            continue
        selected = params.get(parameter)
        if selected is None and None in values:
            natural_limits.append(f"{parameter.split('__')[-1]}=None")
            continue
        numeric = [v for v in values if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)]
        if selected in numeric and len(numeric) >= 2 and selected in {min(numeric), max(numeric)}:
            edges.append(f"{parameter.split('__')[-1]}={selected}")
    return "; ".join(edges) if edges else "none", "; ".join(natural_limits) if natural_limits else "none"


def distribution_row(name, frame):
    row = {
        "subset": name,
        "records": len(frame),
        "semantic_groups": frame[GROUP].nunique(),
        "exact_text_groups": frame[EXACT_GROUP].nunique(),
    }
    for label in CLASS_ORDER:
        count = int(frame[TARGET].eq(label).sum())
        row[f"{label}_records"] = count
        row[f"{label}_share"] = count / len(frame)
    return row


def get_stopwords():
    try:
        words = set(stopwords.words("english")) | set(stopwords.words("german"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("english")) | set(stopwords.words("german"))
    return sorted(words)


def build_pipeline(feature_set, classifier):
    word = TfidfVectorizer(
        analyzer="word", lowercase=True, ngram_range=WORD_NGRAM_RANGE, min_df=2,
        max_features=WORD_MAX_FEATURES, stop_words=BILINGUAL_STOPWORDS,
        sublinear_tf=True, dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", lowercase=True, ngram_range=CHAR_NGRAM_RANGE, min_df=2,
        max_features=CHAR_MAX_FEATURES, sublinear_tf=True, dtype=np.float32,
    )
    text = Pipeline([
        ("representation", FeatureUnion([("word", word), ("char", char)])),
        ("selector", SelectKBest(chi2, k=SELECTED_TEXT_FEATURES)),
    ])

    spec = FEATURE_SETS[feature_set]
    transformers = [("text", text, TEXT)]
    if spec["categorical"]:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
            spec["categorical"],
        ))
    if spec["sentiment"]:
        transformers.append(("sentiment", "passthrough", SENTIMENT_COLUMNS))

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers, sparse_threshold=1.0)),
        ("classifier", classifier),
    ])


def metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(3), zero_division=0
    )
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
    }
    per_class = pd.DataFrame({
        "priority": CLASS_ORDER, "precision": precision, "recall": recall,
        "f1": f1, "support": support.astype(int),
    })
    return overall, per_class


def valid_sentiment(frame):
    values = frame[SENTIMENT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy()
    ok = (
        not values.isna().any().any() and np.isfinite(array).all()
        and (array >= 0).all() and (array <= 1).all()
        and np.allclose(values.sum(axis=1), 1.0, atol=1e-5)
    )
    return values if ok else None


def load_or_create_sentiment(frame):
    text_hash = stable_hash(frame[SENTIMENT_TEXT].astype(str).tolist())

    if SENTIMENT_PATH.exists() and SENTIMENT_META_PATH.exists():
        meta = json.loads(SENTIMENT_META_PATH.read_text(encoding="utf-8"))
        cached = pd.read_csv(SENTIMENT_PATH)
        legacy_hash = hashlib.sha256(
            pd.util.hash_pandas_object(frame[SENTIMENT_TEXT].astype(str), index=False).to_numpy().tobytes()
        ).hexdigest()
        cached_text_hash = meta.get("text_hash", meta.get("text_fingerprint"))
        if (
            meta.get("records") == len(frame)
            and cached_text_hash in {text_hash, legacy_hash}
            and meta.get("model_id") == SENTIMENT_MODEL_ID
            and meta.get("revision", meta.get("model_revision")) == SENTIMENT_MODEL_REVISION
            and len(cached) == len(frame)
            and "modeling_row_id" in cached
            and cached["modeling_row_id"].astype(int).tolist() == frame["modeling_row_id"].tolist()
        ):
            probabilities = valid_sentiment(cached)
            if probabilities is not None:
                section("Sentiment features")
                print("Validated cached sentiment features reused.")
                return probabilities, text_hash

    try:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except (ImportError, OSError) as error:
        raise RuntimeError("Sentiment extraction requires PyTorch and Transformers.") from error

    config = AutoConfig.from_pretrained(SENTIMENT_MODEL_ID, revision=SENTIMENT_MODEL_REVISION)
    labels = {int(i): str(label).lower() for i, label in config.id2label.items()}
    if labels != SENTIMENT_LABELS:
        raise ValueError(f"Unexpected sentiment label mapping: {labels}")

    tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL_ID, revision=SENTIMENT_MODEL_REVISION, use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_ID, revision=SENTIMENT_MODEL_REVISION
    )
    model.eval()

    batches = []
    texts = frame[SENTIMENT_TEXT].tolist()
    section("Sentiment feature extraction")
    print(f"Records                   : {fmt_int(len(texts))}")

    with torch.inference_mode():
        for start in range(0, len(texts), SENTIMENT_BATCH_SIZE):
            encoded = tokenizer(
                texts[start:start + SENTIMENT_BATCH_SIZE], padding=True,
                truncation=True, max_length=512, return_tensors="pt",
            )
            batches.append(torch.softmax(model(**encoded).logits, dim=1).cpu().numpy())

    probabilities = pd.DataFrame(np.vstack(batches), columns=SENTIMENT_COLUMNS)
    if valid_sentiment(probabilities) is None:
        raise ValueError("Generated sentiment probabilities are invalid.")

    output = pd.concat(
        [frame[["modeling_row_id"]].reset_index(drop=True), probabilities], axis=1
    )
    output.to_csv(SENTIMENT_PATH, index=False)
    save_json({
        "records": len(frame), "text_hash": text_hash,
        "model_id": SENTIMENT_MODEL_ID, "revision": SENTIMENT_MODEL_REVISION,
    }, SENTIMENT_META_PATH)
    return probabilities, text_hash


def load_or_create_design(frame):
    required_split_columns = {
        "modeling_row_id", "source_row_id", GROUP, EXACT_GROUP, TARGET, "split"
    }

    if SPLIT_PATH.exists():
        split_source = "validated existing split"
        split = pd.read_parquet(SPLIT_PATH).sort_values("modeling_row_id").reset_index(drop=True)
        if not required_split_columns.issubset(split.columns):
            raise ValueError("Frozen split artifact has an incompatible schema.")
        if split["modeling_row_id"].duplicated().any() or set(split["split"]) != {"training", "holdout"}:
            raise ValueError("Frozen split contains duplicate row IDs or unexpected split labels.")
        if len(split) != len(frame) or not np.array_equal(
            split["modeling_row_id"].to_numpy(), frame["modeling_row_id"].to_numpy()
        ):
            raise ValueError("Frozen split does not match the prepared dataset.")
        for column in ["source_row_id", GROUP, EXACT_GROUP, TARGET]:
            if not np.array_equal(
                split[column].astype(str).to_numpy(), frame[column].astype(str).to_numpy()
            ):
                raise ValueError(f"Frozen split disagrees on {column}.")
    else:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        train_idx, _ = next(
            splitter.split(
                frame,
                frame["y"],
                groups=frame[GROUP],
            )
        )
        split = frame[
            ["modeling_row_id", "source_row_id", GROUP, EXACT_GROUP, TARGET]
        ].copy()
        split["split"] = "holdout"
        split.loc[train_idx, "split"] = "training"
        save_parquet(split, SPLIT_PATH)
        split_source = "new deterministic split"

    train_ids = set(split.loc[split["split"].eq("training"), "modeling_row_id"].astype(int))
    train = frame.loc[frame["modeling_row_id"].isin(train_ids)].copy().reset_index(drop=True)
    holdout = frame.loc[~frame["modeling_row_id"].isin(train_ids)].copy().reset_index(drop=True)

    if set(train[GROUP]) & set(holdout[GROUP]):
        raise ValueError("Semantic-group leakage between training and hold-out.")
    if set(train[EXACT_GROUP]) & set(holdout[EXACT_GROUP]):
        raise ValueError("Exact-text leakage between training and hold-out.")

    if STRICT_DATA_FREEZE:
        frozen_counts = {
            "training_records": (len(train), EXPECTED_TRAINING_RECORDS),
            "holdout_records": (len(holdout), EXPECTED_HOLDOUT_RECORDS),
            "training_semantic_groups": (train[GROUP].nunique(), EXPECTED_TRAINING_SEMANTIC_GROUPS),
            "holdout_semantic_groups": (holdout[GROUP].nunique(), EXPECTED_HOLDOUT_SEMANTIC_GROUPS),
        }
        for label, (observed, expected) in frozen_counts.items():
            if int(observed) != int(expected):
                raise ValueError(
                    f"Frozen design mismatch for {label}: observed {observed}, expected {expected}."
                )

    required_fold_columns = {
        "modeling_row_id", "source_row_id", GROUP, EXACT_GROUP, "cv_fold"
    }
    if FOLD_PATH.exists():
        fold_source = "validated existing CV folds"
        stored = pd.read_parquet(FOLD_PATH)
        if not required_fold_columns.issubset(stored.columns):
            raise ValueError("Frozen CV-fold artifact has an incompatible schema.")
        if stored["modeling_row_id"].duplicated().any():
            raise ValueError("Frozen CV-fold artifact contains duplicate row IDs.")
        lookup = stored.set_index("modeling_row_id")
        if set(lookup.index.astype(int)) != set(train["modeling_row_id"].astype(int)):
            raise ValueError("Frozen CV folds do not match the frozen training partition.")
        training_lookup = train.set_index("modeling_row_id")
        for column in ["source_row_id", GROUP, EXACT_GROUP]:
            aligned = lookup.loc[training_lookup.index, column].astype(str).to_numpy()
            expected = training_lookup[column].astype(str).to_numpy()
            if not np.array_equal(aligned, expected):
                raise ValueError(f"Frozen CV-fold artifact disagrees on {column}.")
        train["cv_fold"] = train["modeling_row_id"].map(lookup["cv_fold"]).astype(np.int8)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        train["cv_fold"] = 0
        for fold, (_, valid_idx) in enumerate(
            splitter.split(
                train,
                train["y"],
                groups=train[GROUP],
            ),
            start=1,
        ):
            train.loc[valid_idx, "cv_fold"] = fold
        save_parquet(
            train[
                ["modeling_row_id", "source_row_id", GROUP, EXACT_GROUP, "cv_fold"]
            ],
            FOLD_PATH,
        )
        fold_source = "new deterministic CV folds"

    if set(train["cv_fold"].unique()) != set(range(1, N_SPLITS + 1)):
        raise ValueError("Expected five frozen CV folds numbered 1..5.")

    folds = []
    for fold in range(1, N_SPLITS + 1):
        valid_idx = np.flatnonzero(train["cv_fold"].to_numpy() == fold)
        train_idx = np.flatnonzero(train["cv_fold"].to_numpy() != fold)
        if set(train.iloc[train_idx][GROUP]) & set(train.iloc[valid_idx][GROUP]):
            raise ValueError(f"Semantic-group leakage in CV fold {fold}.")
        if set(train.iloc[train_idx][EXACT_GROUP]) & set(train.iloc[valid_idx][EXACT_GROUP]):
            raise ValueError(f"Exact-text leakage in CV fold {fold}.")
        folds.append((train_idx, valid_idx))

    return train, holdout, folds, split_source, fold_source


# %% 03 - Load prepared data and validate the frozen design

for required_path in [
    DATA_PATH,
    DATA_PREPARATION_SUMMARY_PATH,
    DATA_PREPARATION_COHESION_PATH,
    SCREENING_A_MANIFEST_PATH,
    SCREENING_B_MANIFEST_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required Modeling input/provenance artifact not found: {required_path}")

SCREENING_HASHES, SCREENING_MANIFESTS = validate_screening_provenance()

data = pd.read_csv(DATA_PATH, low_memory=False)
data["modeling_row_id"] = np.arange(len(data), dtype=np.int64)

required = {
    "source_row_id", TARGET, GROUP, EXACT_GROUP, "semantic_group_size", "text_group_size",
    TEXT, SENTIMENT_TEXT, "type", "language", "queue",
}
missing = sorted(required - set(data.columns))
if missing:
    raise ValueError(f"Missing modeling columns: {missing}")

for column in [GROUP, EXACT_GROUP, TEXT, SENTIMENT_TEXT, "type", "language", "queue"]:
    data[column] = data[column].fillna("").astype(str).str.strip()
data[TARGET] = data[TARGET].astype(str).str.lower().str.strip()
data["source_row_id"] = pd.to_numeric(data["source_row_id"], errors="raise").astype(np.int64)
for column in ["semantic_group_size", "text_group_size"]:
    data[column] = pd.to_numeric(data[column], errors="raise").astype(np.int64)

if data.empty or data[[GROUP, EXACT_GROUP, TEXT, SENTIMENT_TEXT, "type", "language", "queue"]].eq("").any().any():
    raise ValueError("Prepared modeling data contain empty required values.")
if data["source_row_id"].duplicated().any():
    raise ValueError("Duplicate source_row_id values remain.")
if set(data[TARGET]) != set(CLASS_ORDER):
    raise ValueError(f"Target classes must be exactly {CLASS_ORDER}.")
if set(data["language"].unique()) != {"de", "en"}:
    raise ValueError("The language feature must contain exactly de and en.")
if data.groupby(EXACT_GROUP)[TARGET].nunique().gt(1).any():
    raise ValueError("Conflicting targets occur within an exact text group.")
if data.groupby(EXACT_GROUP)[GROUP].nunique().gt(1).any():
    raise ValueError("An exact text group is split across semantic groups.")

semantic_sizes = data.groupby(GROUP).size()
exact_sizes = data.groupby(EXACT_GROUP).size()
if not np.array_equal(data["semantic_group_size"].to_numpy(), data[GROUP].map(semantic_sizes).to_numpy()):
    raise ValueError("Stored semantic_group_size values are inconsistent.")
if not np.array_equal(data["text_group_size"].to_numpy(), data[EXACT_GROUP].map(exact_sizes).to_numpy()):
    raise ValueError("Stored text_group_size values are inconsistent.")

semantic_conflicts = int(data.groupby(GROUP)[TARGET].nunique().gt(1).sum())
grouping_method, minimum_group_similarity = validate_preparation_artifacts(data)

if STRICT_DATA_FREEZE:
    if len(data) != EXPECTED_PREPARED_RECORDS:
        raise ValueError(
            f"Prepared record count changed: {len(data)} != {EXPECTED_PREPARED_RECORDS}."
        )
    if int(data[GROUP].nunique()) != EXPECTED_SEMANTIC_GROUPS:
        raise ValueError(
            f"Semantic-group count changed: {data[GROUP].nunique()} != {EXPECTED_SEMANTIC_GROUPS}."
        )
    if semantic_conflicts != EXPECTED_SEMANTIC_CONFLICT_GROUPS:
        raise ValueError(
            f"Semantic conflict-group count changed: {semantic_conflicts} "
            f"!= {EXPECTED_SEMANTIC_CONFLICT_GROUPS}."
        )
    observed_target_counts = data[TARGET].value_counts().to_dict()
    if observed_target_counts != EXPECTED_TARGET_COUNTS:
        raise ValueError(
            f"Frozen target distribution changed: {observed_target_counts} "
            f"!= {EXPECTED_TARGET_COUNTS}."
        )
    if grouping_method != EXPECTED_GROUPING_METHOD:
        raise ValueError(
            f"Grouping method changed: {grouping_method!r} != {EXPECTED_GROUPING_METHOD!r}."
        )

data["y"] = data[TARGET].map(CLASS_TO_INT).astype(np.int8)
DATA_SHA256 = sha256(DATA_PATH)
validate_screening_decisions(SCREENING_MANIFESTS, DATA_SHA256)

BILINGUAL_STOPWORDS = get_stopwords()
STOPWORD_PATH.write_text("\n".join(BILINGUAL_STOPWORDS) + "\n", encoding="utf-8")
STOPWORD_SHA256 = sha256(STOPWORD_PATH)

sentiment, SENTIMENT_TEXT_HASH = load_or_create_sentiment(data)
for column in SENTIMENT_COLUMNS:
    data[column] = sentiment[column].to_numpy()

training, holdout, CV_FOLDS, SPLIT_SOURCE, FOLD_SOURCE = load_or_create_design(data)
DESIGN_HASH = stable_hash({
    "training_ids": training["modeling_row_id"].tolist(),
    "training_semantic_groups": training[GROUP].tolist(),
    "cv_folds": training["cv_fold"].tolist(),
    "group": GROUP,
})

design_rows = [
    distribution_row("prepared", data),
    distribution_row("training", training),
    distribution_row("holdout", holdout),
]
design_rows.extend(
    distribution_row(f"cv_fold_{fold}", training.loc[training["cv_fold"].eq(fold)])
    for fold in range(1, N_SPLITS + 1)
)
save_csv(pd.DataFrame(design_rows), DESIGN_SUMMARY_PATH)

section("Frozen modeling design")
print(f"Script build              : {SCRIPT_BUILD}")
print(f"Prepared records          : {fmt_int(len(data))}")
print(f"Semantic groups           : {fmt_int(data[GROUP].nunique())}")
print(f"Semantic conflict groups  : {fmt_int(semantic_conflicts)}")
print(f"Grouping method           : {grouping_method}")
print(f"Minimum group similarity  : {minimum_group_similarity:.6f}")
print(f"Training records          : {fmt_int(len(training))}")
print(f"Hold-out records          : {fmt_int(len(holdout))}")
print(f"Hold-out semantic groups  : {fmt_int(holdout[GROUP].nunique())}")
print("Semantic-group overlap    : 0")
print("Exact-text overlap        : 0")
print("Text representation       : Word + Character TF-IDF")
print(f"Selected text features    : {fmt_int(SELECTED_TEXT_FEATURES)}")
print(f"CV folds                  : {N_SPLITS}")
print(f"Split source              : {SPLIT_SOURCE}")
print(f"CV-fold source            : {FOLD_SOURCE}")
print(f"Search budget/model       : {SEARCH_BUDGET}")

# %% 04 - Tune seven classifiers on FS1 with equal search budget

for model_name, grid in MODEL_GRIDS.items():
    if len(list(ParameterGrid(grid))) != SEARCH_BUDGET:
        raise ValueError(f"{model_name} does not have exactly {SEARCH_BUDGET} candidates.")

TUNING_SIGNATURE = stable_hash({
    "analytical_design_id": ANALYTICAL_DESIGN_ID,
    "data_sha256": DATA_SHA256,
    "design_hash": DESIGN_HASH,
    "feature_set": "FS1",
    "model_order": MODEL_ORDER,
    "model_grids": MODEL_GRIDS,
    "search_budget": SEARCH_BUDGET,
    "scoring": SCORING,
    "cv_folds": N_SPLITS,
    "stopword_sha256": STOPWORD_SHA256,
    "selected_text_features": SELECTED_TEXT_FEATURES,
})

SELECTED_PARAMS = {}
parameter_meta = {}
search_candidates = pd.DataFrame()
search_summary = pd.DataFrame()
completed_models = []
tuning_source = "new tuning run"

if (
    REUSE_VALID_RESULTS
    and PARAMETER_PATH.exists()
    and SEARCH_SUMMARY_PATH.exists()
    and SEARCH_CANDIDATES_PATH.exists()
):
    try:
        parameter_meta = json.loads(PARAMETER_PATH.read_text(encoding="utf-8"))
        cached_summary = pd.read_csv(SEARCH_SUMMARY_PATH)
        cached_candidates = pd.read_csv(SEARCH_CANDIDATES_PATH)
        cached_params = parameter_meta.get("selected_parameters", {})
        cached_parameter_hash = parameter_meta.get("parameter_hash")

        full_cache_valid = (
            parameter_meta.get("data_sha256") == DATA_SHA256
            and parameter_meta.get("design_hash") == DESIGN_HASH
            and int(parameter_meta.get("search_budget", -1)) == SEARCH_BUDGET
            and set(cached_params) == set(MODEL_ORDER)
            and cached_parameter_hash == stable_hash(cached_params)
            and set(cached_summary["model"]) == set(MODEL_ORDER)
            and set(cached_candidates["model"]) == set(MODEL_ORDER)
            and cached_candidates.groupby("model").size().eq(SEARCH_BUDGET).all()
        )
        if full_cache_valid:
            SELECTED_PARAMS = cached_params
            search_summary = cached_summary.copy()
            search_candidates = cached_candidates.copy()
            completed_models = MODEL_ORDER.copy()
            PARAMETER_HASH = cached_parameter_hash
            tuning_source = "validated completed tuning reused"
    except (KeyError, ValueError, json.JSONDecodeError):
        pass

if not completed_models and RESUME_PARTIAL_RESULTS and TUNING_CHECKPOINT_PATH.exists():
    try:
        checkpoint = json.loads(TUNING_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if (
            checkpoint.get("status") == "partial"
            and checkpoint.get("signature") == TUNING_SIGNATURE
        ):
            checkpoint_models = checkpoint.get("completed_models", [])
            checkpoint_params = checkpoint.get("selected_parameters", {})
            if (
                set(checkpoint_models).issubset(MODEL_ORDER)
                and set(checkpoint_params) == set(checkpoint_models)
            ):
                completed_models = [m for m in MODEL_ORDER if m in checkpoint_models]
                SELECTED_PARAMS = checkpoint_params
                if SEARCH_CANDIDATES_PATH.exists() and completed_models:
                    cached_candidates = pd.read_csv(SEARCH_CANDIDATES_PATH)
                    search_candidates = cached_candidates.loc[
                        cached_candidates["model"].isin(completed_models)
                    ].copy()
                if SEARCH_SUMMARY_PATH.exists() and completed_models:
                    cached_summary = pd.read_csv(SEARCH_SUMMARY_PATH)
                    search_summary = cached_summary.loc[
                        cached_summary["model"].isin(completed_models)
                    ].copy()
                valid_partial = (
                    not completed_models
                    or (
                        set(search_summary["model"]) == set(completed_models)
                        and set(search_candidates["model"]) == set(completed_models)
                        and search_candidates.groupby("model").size().eq(SEARCH_BUDGET).all()
                    )
                )
                if valid_partial:
                    tuning_source = "validated partial tuning resumed"
                else:
                    completed_models = []
                    SELECTED_PARAMS = {}
                    search_candidates = pd.DataFrame()
                    search_summary = pd.DataFrame()
    except (KeyError, ValueError, json.JSONDecodeError):
        completed_models = []
        SELECTED_PARAMS = {}
        search_candidates = pd.DataFrame()
        search_summary = pd.DataFrame()

section("FS1 classifier tuning")
print(f"Algorithms                : {len(MODEL_ORDER)}")
print(f"Candidates per algorithm  : {SEARCH_BUDGET}")
print(f"CV fits per algorithm     : {SEARCH_BUDGET * N_SPLITS}")
print(f"Source                    : {tuning_source}")
print(f"Completed algorithms      : {len(completed_models)} / {len(MODEL_ORDER)}")

for model_name in MODEL_ORDER:
    if model_name in completed_models:
        continue

    print(f"Running                   : {MODEL_LABELS[model_name]}")
    started = perf_counter()

    search = GridSearchCV(
        build_pipeline("FS1", clone(BASE_MODELS[model_name])),
        MODEL_GRIDS[model_name],
        scoring=SCORING,
        refit=False,
        cv=CV_FOLDS,
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(training, training["y"])

    raw = pd.DataFrame(search.cv_results_)
    candidates = pd.DataFrame({
        "model": model_name,
        "candidate_index": np.arange(1, len(raw) + 1),
        "mean_train_macro_f1": raw["mean_train_macro_f1"],
        "mean_macro_f1": raw["mean_test_macro_f1"],
        "std_macro_f1": raw["std_test_macro_f1"],
        "mean_weighted_f1": raw["mean_test_weighted_f1"],
        "mean_accuracy": raw["mean_test_accuracy"],
        "mean_fit_seconds": raw["mean_fit_time"],
        "params": raw["params"].map(
            lambda x: json.dumps(x, sort_keys=True, default=str)
        ),
    })
    candidates["train_validation_gap"] = (
        candidates["mean_train_macro_f1"] - candidates["mean_macro_f1"]
    )
    candidates = candidates.sort_values(
        ["mean_macro_f1", "mean_weighted_f1", "mean_accuracy", "candidate_index"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    candidates.insert(1, "rank_within_model", np.arange(1, len(candidates) + 1))

    best, second = candidates.iloc[0], candidates.iloc[1]
    best_params = json.loads(best["params"])
    SELECTED_PARAMS[model_name] = best_params
    edge_signal, natural_limit = search_diagnostics(MODEL_GRIDS[model_name], best_params)

    summary_row = pd.DataFrame([{
        "model": model_name,
        "model_family": MODEL_FAMILIES[model_name],
        "candidate_configurations": SEARCH_BUDGET,
        "mean_train_macro_f1": float(best["mean_train_macro_f1"]),
        "mean_macro_f1": float(best["mean_macro_f1"]),
        "std_macro_f1": float(best["std_macro_f1"]),
        "train_validation_gap": float(best["train_validation_gap"]),
        "gap_to_second_macro_f1": float(best["mean_macro_f1"] - second["mean_macro_f1"]),
        "mean_weighted_f1": float(best["mean_weighted_f1"]),
        "mean_accuracy": float(best["mean_accuracy"]),
        "boundary_signal": edge_signal,
        "natural_limit": natural_limit,
        "best_params": best["params"],
        "elapsed_seconds": perf_counter() - started,
    }])

    search_candidates = pd.concat(
        [search_candidates.loc[search_candidates.get("model", pd.Series(dtype=str)).ne(model_name)]
         if not search_candidates.empty else search_candidates,
         candidates],
        ignore_index=True,
    )
    search_summary = pd.concat(
        [search_summary.loc[search_summary.get("model", pd.Series(dtype=str)).ne(model_name)]
         if not search_summary.empty else search_summary,
         summary_row],
        ignore_index=True,
    )
    completed_models = [m for m in MODEL_ORDER if m in SELECTED_PARAMS]

    save_csv(search_candidates, SEARCH_CANDIDATES_PATH)
    save_csv(search_summary, SEARCH_SUMMARY_PATH)
    save_json({
        "status": "partial",
        "signature": TUNING_SIGNATURE,
        "data_sha256": DATA_SHA256,
        "design_hash": DESIGN_HASH,
        "completed_models": completed_models,
        "selected_parameters": SELECTED_PARAMS,
    }, TUNING_CHECKPOINT_PATH)

if set(SELECTED_PARAMS) != set(MODEL_ORDER):
    raise ValueError("Tuning did not produce selected parameters for all seven classifiers.")

# Canonical ordering and final validation.
search_candidates = search_candidates.loc[search_candidates["model"].isin(MODEL_ORDER)].copy()
search_summary = search_summary.loc[search_summary["model"].isin(MODEL_ORDER)].copy()
if (
    set(search_summary["model"]) != set(MODEL_ORDER)
    or set(search_candidates["model"]) != set(MODEL_ORDER)
    or not search_candidates.groupby("model").size().eq(SEARCH_BUDGET).all()
):
    raise ValueError("Completed tuning artifacts are incomplete or inconsistent.")

search_summary = search_summary.sort_values(
    ["mean_macro_f1", "mean_weighted_f1", "mean_accuracy", "model"],
    ascending=[False, False, False, True],
    kind="mergesort",
).reset_index(drop=True)
search_candidates = search_candidates.sort_values(
    ["model", "rank_within_model", "candidate_index"],
    kind="mergesort",
).reset_index(drop=True)

save_csv(search_candidates, SEARCH_CANDIDATES_PATH)
save_csv(search_summary, SEARCH_SUMMARY_PATH)
PARAMETER_HASH = stable_hash(SELECTED_PARAMS)
save_json({
    "status": "complete",
    "build": SCRIPT_BUILD,
    "analytical_design_id": ANALYTICAL_DESIGN_ID,
    "tuning_signature": TUNING_SIGNATURE,
    "data_sha256": DATA_SHA256,
    "design_hash": DESIGN_HASH,
    "search_budget": SEARCH_BUDGET,
    "parameter_hash": PARAMETER_HASH,
    "selected_parameters": SELECTED_PARAMS,
    "source": tuning_source,
}, PARAMETER_PATH)
save_json({
    "status": "complete",
    "signature": TUNING_SIGNATURE,
    "completed_models": MODEL_ORDER,
    "selected_parameters": SELECTED_PARAMS,
}, TUNING_CHECKPOINT_PATH)

section("Selected FS1 configurations")
for row in search_summary.itertuples(index=False):
    print(
        f"{MODEL_LABELS[row.model]:<24}: Macro-F1 {row.mean_macro_f1:.4f} ± {row.std_macro_f1:.4f} "
        f"| train gap {row.train_validation_gap:.4f} | gap to 2nd {row.gap_to_second_macro_f1:.4f}"
    )
    print(
        f"  Search diagnostics      : edge={row.boundary_signal}; "
        f"natural limit={row.natural_limit}"
    )


# %% 05 - Technical majority baseline

dummy_pred = cross_val_predict(
    DummyClassifier(strategy="most_frequent"),
    np.zeros((len(training), 1), dtype=np.float32), training["y"],
    cv=CV_FOLDS, method="predict", n_jobs=1,
).astype(np.int8)
dummy_metrics, _ = metrics(training["y"].to_numpy(), dummy_pred)
save_csv(pd.DataFrame([{"model": "dummy_majority", **dummy_metrics}]), DUMMY_PATH)

section("Technical baseline")
print(f"Dummy Majority Macro-F1   : {dummy_metrics['macro_f1']:.4f}")
print("H1 baseline               : Logistic Regression")


# %% 06 - Generate restart-safe OOF predictions for FS1-FS5 without retuning

OOF_COLUMNS = {
    (feature_set, model_name): f"pred_{feature_set}_{model_name}"
    for feature_set in FEATURE_ORDER
    for model_name in MODEL_ORDER
}
BASE_OOF_COLUMNS = [
    "modeling_row_id", "source_row_id", GROUP, EXACT_GROUP, "y", "cv_fold"
]

OOF_SIGNATURE = stable_hash({
    "analytical_design_id": ANALYTICAL_DESIGN_ID,
    "data_sha256": DATA_SHA256,
    "design_hash": DESIGN_HASH,
    "parameter_hash": PARAMETER_HASH,
    "feature_sets": FEATURE_SETS,
    "models": MODEL_ORDER,
    "stopword_sha256": STOPWORD_SHA256,
    "selected_text_features": SELECTED_TEXT_FEATURES,
})

oof = training[BASE_OOF_COLUMNS].copy()
reused_oof_columns = 0
oof_source = "new OOF run"

if OOF_PATH.exists() and OOF_META_PATH.exists():
    try:
        metadata = json.loads(OOF_META_PATH.read_text(encoding="utf-8"))
        cached = pd.read_parquet(OOF_PATH)
        core_compatible = (
            metadata.get("data_sha256") == DATA_SHA256
            and metadata.get("design_hash") == DESIGN_HASH
            and metadata.get("parameter_hash") == PARAMETER_HASH
            and len(cached) == len(training)
            and set(BASE_OOF_COLUMNS).issubset(cached.columns)
            and np.array_equal(
                cached["modeling_row_id"].to_numpy(),
                training["modeling_row_id"].to_numpy(),
            )
            and np.array_equal(
                cached["cv_fold"].to_numpy(),
                training["cv_fold"].to_numpy(),
            )
        )
        is_new_partial = (
            RESUME_PARTIAL_RESULTS
            and metadata.get("status") == "partial"
            and metadata.get("signature") == OOF_SIGNATURE
        )
        is_valid_complete = (
            REUSE_VALID_RESULTS
            and core_compatible
            and (
                metadata.get("status") in {None, "complete"}
                or metadata.get("signature") == OOF_SIGNATURE
            )
        )
        if core_compatible and (is_new_partial or is_valid_complete):
            for column in OOF_COLUMNS.values():
                if column in cached.columns and not cached[column].isna().any():
                    oof[column] = cached[column].to_numpy(np.int8)
                    reused_oof_columns += 1
            if reused_oof_columns:
                oof_source = (
                    "validated partial OOF resumed"
                    if is_new_partial and reused_oof_columns < len(OOF_COLUMNS)
                    else "validated completed OOF reused"
                )
    except (KeyError, ValueError, json.JSONDecodeError):
        pass

missing_keys = [
    key for key, column in OOF_COLUMNS.items()
    if column not in oof.columns or oof[column].isna().any()
]

section("OOF prediction generation")
print(f"Required combinations     : {len(OOF_COLUMNS)}")
print(f"Reusable combinations     : {reused_oof_columns}")
print(f"Remaining combinations    : {len(missing_keys)}")
print(f"Parallel CV jobs          : {N_JOBS}")
print(f"Source                    : {oof_source}")
print("Checkpointing             : after every completed combination")

save_parquet(oof, OOF_PATH)
save_json({
    "status": "partial" if missing_keys else "complete",
    "signature": OOF_SIGNATURE,
    "analytical_design_id": ANALYTICAL_DESIGN_ID,
    "data_sha256": DATA_SHA256,
    "design_hash": DESIGN_HASH,
    "parameter_hash": PARAMETER_HASH,
    "completed_prediction_columns": [
        column for column in OOF_COLUMNS.values()
        if column in oof.columns and not oof[column].isna().any()
    ],
    "prediction_columns": list(OOF_COLUMNS.values()),
    "holdout_predictions_included": False,
}, OOF_META_PATH)

for position, (feature_set, model_name) in enumerate(missing_keys, start=1):
    prediction_column = OOF_COLUMNS[(feature_set, model_name)]
    print(
        f"{position:02d}/{len(missing_keys):02d}                     : "
        f"{feature_set} + {MODEL_LABELS[model_name]}"
    )

    model = build_pipeline(feature_set, clone(BASE_MODELS[model_name]))
    model.set_params(**SELECTED_PARAMS[model_name])
    predictions = cross_val_predict(
        model,
        training,
        training["y"],
        cv=CV_FOLDS,
        method="predict",
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
    ).astype(np.int8)
    oof[prediction_column] = predictions

    save_parquet(oof, OOF_PATH)
    save_json({
        "status": "partial",
        "signature": OOF_SIGNATURE,
        "analytical_design_id": ANALYTICAL_DESIGN_ID,
        "data_sha256": DATA_SHA256,
        "design_hash": DESIGN_HASH,
        "parameter_hash": PARAMETER_HASH,
        "completed_prediction_columns": [
            column for column in OOF_COLUMNS.values()
            if column in oof.columns and not oof[column].isna().any()
        ],
        "prediction_columns": list(OOF_COLUMNS.values()),
        "holdout_predictions_included": False,
    }, OOF_META_PATH)

required_prediction_columns = list(OOF_COLUMNS.values())
if not set(required_prediction_columns).issubset(oof.columns):
    raise ValueError("OOF generation finished without all required prediction columns.")
if oof[required_prediction_columns].isna().any().any():
    raise ValueError("At least one OOF prediction column is incomplete.")

save_parquet(oof, OOF_PATH)
save_json({
    "status": "complete",
    "signature": OOF_SIGNATURE,
    "analytical_design_id": ANALYTICAL_DESIGN_ID,
    "data_sha256": DATA_SHA256,
    "design_hash": DESIGN_HASH,
    "parameter_hash": PARAMETER_HASH,
    "completed_prediction_columns": required_prediction_columns,
    "prediction_columns": required_prediction_columns,
    "holdout_predictions_included": False,
}, OOF_META_PATH)


# %% 07 - Summarize cross-validated performance

fold_rows = []
class_rows = []
y_all = training["y"].to_numpy(np.int8)

for feature_set in FEATURE_ORDER:
    for model_name in MODEL_ORDER:
        pred = oof[OOF_COLUMNS[feature_set, model_name]].to_numpy(np.int8)
        for fold in range(1, N_SPLITS + 1):
            mask = training["cv_fold"].eq(fold).to_numpy()
            overall, per_class = metrics(y_all[mask], pred[mask])
            common = {
                "feature_set": feature_set,
                "features": FEATURE_SETS[feature_set]["label"],
                "model": model_name,
                "model_family": MODEL_FAMILIES[model_name],
                "fold": fold,
            }
            fold_rows.append({**common, **overall})
            class_rows.extend({**common, **row} for row in per_class.to_dict("records"))

fold_metrics = pd.DataFrame(fold_rows)
fold_class_metrics = pd.DataFrame(class_rows)
save_parquet(fold_metrics, FOLD_METRICS_PATH)
save_parquet(fold_class_metrics, FOLD_CLASS_METRICS_PATH)

cv_results = (
    fold_metrics.groupby(["feature_set", "features", "model", "model_family"], as_index=False)
    .agg(
        mean_macro_f1=("macro_f1", "mean"), std_macro_f1=("macro_f1", "std"),
        mean_weighted_f1=("weighted_f1", "mean"), std_weighted_f1=("weighted_f1", "std"),
        mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
        mean_macro_precision=("macro_precision", "mean"),
        mean_macro_recall=("macro_recall", "mean"),
    )
)

feature_rank = {fs: i for i, fs in enumerate(FEATURE_ORDER)}
cv_results["feature_rank"] = cv_results["feature_set"].map(feature_rank)
cv_results = cv_results.sort_values(["model", "feature_rank"]).reset_index(drop=True)
cv_results["delta_vs_fs1"] = cv_results.groupby("model")["mean_macro_f1"].transform(lambda s: s - s.iloc[0])
cv_results["delta_vs_previous"] = cv_results.groupby("model")["mean_macro_f1"].diff()
cv_results["best_params"] = cv_results["model"].map(lambda m: json.dumps(SELECTED_PARAMS[m], sort_keys=True))
cv_results = cv_results.sort_values(SORT_METRICS, ascending=False).reset_index(drop=True)
save_csv(cv_results, CV_RESULTS_PATH)

feature_set_selection = (
    cv_results
    .groupby("feature_set", as_index=False)
    .agg(
        mean_macro_f1_across_models=("mean_macro_f1", "mean"),
        mean_weighted_f1_across_models=("mean_weighted_f1", "mean"),
        mean_accuracy_across_models=("mean_accuracy", "mean"),
    )
)
feature_set_selection["feature_rank"] = (
    feature_set_selection["feature_set"].map(feature_rank)
)
if feature_set_selection["feature_rank"].isna().any():
    raise ValueError("Feature-set selection contains an unknown feature set.")

feature_set_selection = (
    feature_set_selection
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
feature_set_selection["selected_for_common_comparison"] = False
feature_set_selection.loc[0, "selected_for_common_comparison"] = True
COMMON_FEATURE_SET = str(feature_set_selection.iloc[0]["feature_set"])

save_csv(feature_set_selection, FEATURE_SET_SELECTION_PATH)

section("Common feature-set selection")
print(
    feature_set_selection[[
        "feature_set",
        "mean_macro_f1_across_models",
        "mean_weighted_f1_across_models",
        "mean_accuracy_across_models",
        "selected_for_common_comparison",
    ]].to_string(
        index=False,
        formatters={
            "mean_macro_f1_across_models": "{:.4f}".format,
            "mean_weighted_f1_across_models": "{:.4f}".format,
            "mean_accuracy_across_models": "{:.4f}".format,
        },
    )
)
print(f"Selected common feature set: {COMMON_FEATURE_SET}")

cv_class_results = (
    fold_class_metrics.groupby(["feature_set", "features", "model", "model_family", "priority"], as_index=False)
    .agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
    )
)
save_csv(cv_class_results, CV_CLASS_RESULTS_PATH)

matrix = cv_results.pivot(index="feature_set", columns="model", values="mean_macro_f1").reindex(
    index=FEATURE_ORDER, columns=MODEL_ORDER
)
section("Cross-validated Macro-F1")
print(matrix.round(4).to_string())

best_by_feature = (
    cv_results.sort_values(SORT_METRICS, ascending=False)
    .groupby("feature_set", as_index=False).first()
    .set_index("feature_set").reindex(FEATURE_ORDER).reset_index()
)
section("Best model by feature set")
print(best_by_feature[["feature_set", "model", "mean_macro_f1", "std_macro_f1"]].to_string(
    index=False,
    formatters={"mean_macro_f1": "{:.4f}".format, "std_macro_f1": "{:.4f}".format},
))


# %% 08 - Create paper-ready grayscale CV figures

values = matrix.to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(9.2, 3.8))
image = ax.imshow(values, cmap="Greys", aspect="auto", interpolation="nearest")
ax.set_xticks(np.arange(len(MODEL_ORDER)), labels=[
    "Logistic\nRegression", "Linear\nSVM", "Complement\nNB", "kNN",
    "Random\nForest", "XGBoost", "LightGBM",
])
ax.set_yticks(np.arange(len(FEATURE_ORDER)), labels=FEATURE_ORDER)
ax.set_xlabel("Classifier")
ax.set_ylabel("Feature set")

threshold = np.nanmean(values)
for r in range(values.shape[0]):
    for c in range(values.shape[1]):
        ax.text(
            c, r, f"{values[r, c]:.3f}",
            ha="center", va="center",
            color="white" if values[r, c] > threshold else "black",
            fontsize=7.5,
        )

bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
bar.set_label("Cross-validated Macro-F1")
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

figure_ranked = cv_results.sort_values(SORT_METRICS, ascending=False)
figure_feature_set = str(figure_ranked.iloc[0]["feature_set"])

class_plot = (
    cv_class_results.loc[cv_class_results["feature_set"].eq(figure_feature_set)]
    .pivot(index="model", columns="priority", values="f1_mean")
    .reindex(index=MODEL_ORDER, columns=CLASS_ORDER)
)

if class_plot.isna().any().any():
    raise ValueError(
        f"Incomplete class-specific F1 results for selected feature set {figure_feature_set}."
    )

x = np.arange(len(CLASS_ORDER))
width = 0.11

gray_levels = np.linspace(0.25, 0.85, len(MODEL_ORDER))
gray_colors = [plt.cm.Greys(level) for level in gray_levels]

fig, ax = plt.subplots(figsize=(8.4, 4.3))

for index, model_name in enumerate(MODEL_ORDER):
    offset = (index - (len(MODEL_ORDER) - 1) / 2) * width
    bars = ax.bar(
        x + offset,
        class_plot.loc[model_name].to_numpy(dtype=float),
        width=width,
        facecolor=gray_colors[index],
        edgecolor="black",
        linewidth=0.6,
        label=MODEL_LABELS[model_name],
    )
    ax.bar_label(bars, fmt="%.2f", padding=1, fontsize=6)

ax.set_xticks(x, labels=[label.capitalize() for label in CLASS_ORDER])
ax.set_xlabel("Priority class")
ax.set_ylabel("Cross-validated F1-score")
ax.set_ylim(0, 1)
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.17),
    ncol=4,
    frameon=False,
    fontsize=7,
)

fig.tight_layout()
fig.savefig(CLASS_F1_FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)


# %% 09 - Fit or reuse required full-training models

ranked = cv_results.sort_values(SORT_METRICS, ascending=False)
roles = {
    "final_overall_model": ranked.iloc[0],
    "text_only_core_model": ranked.loc[ranked["feature_set"].eq("FS1")].iloc[0],
}
for model_name in MODEL_ORDER:
    candidate = cv_results.loc[
        cv_results["feature_set"].eq(COMMON_FEATURE_SET)
        & cv_results["model"].eq(model_name)
    ]
    if len(candidate) != 1:
        raise ValueError(
            f"Expected exactly one {COMMON_FEATURE_SET} CV row for {model_name}; "
            f"found {len(candidate)}."
        )
    roles[f"common_{model_name}"] = candidate.iloc[0]

combinations = {}
for role, row in roles.items():
    combinations.setdefault(
        (str(row["feature_set"]), str(row["model"])), []
    ).append(role)

expected_model_artifacts = {
    f"03_model_{feature_set}_{model_name}.joblib"
    for feature_set, model_name in combinations
}
existing_model_artifacts = {path.name for path in MODEL_DIR.glob("03_model_*.joblib")}
stale_model_artifacts = sorted(existing_model_artifacts - expected_model_artifacts)
if stale_model_artifacts and FAIL_ON_STALE_MODEL_ARTIFACTS:
    raise RuntimeError(
        "Stale model artifacts are present and must be removed manually before the final run: "
        + ", ".join(stale_model_artifacts)
    )

MODEL_FIT_SIGNATURE = stable_hash({
    "analytical_design_id": ANALYTICAL_DESIGN_ID,
    "data_sha256": DATA_SHA256,
    "design_hash": DESIGN_HASH,
    "parameter_hash": PARAMETER_HASH,
    "expected_artifacts": sorted(expected_model_artifacts),
})

model_meta = {
    "status": "partial",
    "signature": MODEL_FIT_SIGNATURE,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "data_sha256": DATA_SHA256,
    "design_hash": DESIGN_HASH,
    "parameter_hash": PARAMETER_HASH,
    "models": {},
}

if MODEL_META_PATH.exists():
    try:
        cached_meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
        core_compatible = (
            cached_meta.get("data_sha256") == DATA_SHA256
            and cached_meta.get("design_hash") == DESIGN_HASH
            and cached_meta.get("parameter_hash") == PARAMETER_HASH
        )
        accept_completed = REUSE_VALID_RESULTS and core_compatible
        accept_partial = (
            RESUME_PARTIAL_RESULTS
            and core_compatible
            and cached_meta.get("status") == "partial"
            and cached_meta.get("signature") == MODEL_FIT_SIGNATURE
        )
        if accept_completed or accept_partial:
            model_meta["models"] = cached_meta.get("models", {})
    except (ValueError, json.JSONDecodeError):
        pass

selection_rows = []
section("Fit selected models on full training partition")

for (feature_set, model_name), model_roles in combinations.items():
    artifact = f"03_model_{feature_set}_{model_name}.joblib"
    path = MODEL_DIR / artifact
    cached_details = model_meta["models"].get(artifact, {})

    artifact_valid = (
        path.exists()
        and cached_details.get("feature_set") == feature_set
        and cached_details.get("model") == model_name
        and cached_details.get("selected_params") == SELECTED_PARAMS[model_name]
        and cached_details.get("sha256") == sha256(path)
    )

    if artifact_valid:
        print(f"Reusing                   : {feature_set} + {MODEL_LABELS[model_name]}")
    else:
        print(f"Fitting                   : {feature_set} + {MODEL_LABELS[model_name]}")
        model = build_pipeline(feature_set, clone(BASE_MODELS[model_name]))
        model.set_params(**SELECTED_PARAMS[model_name])
        model.fit(training, training["y"])
        save_joblib(model, path)

    current_model_sha256 = sha256(path)
    model_meta["models"][artifact] = {
        "sha256": current_model_sha256,
        "file_sha256": current_model_sha256,
        "feature_set": feature_set,
        "model": model_name,
        "selected_params": SELECTED_PARAMS[model_name],
        "roles": sorted(model_roles),
    }
    save_json(model_meta, MODEL_META_PATH)

    for role in model_roles:
        row = roles[role]
        if role == "final_overall_model":
            selection_rule = (
                "Highest CV Macro-F1 across all model-feature combinations; "
                "then Weighted-F1, then Accuracy"
            )
        elif role == "text_only_core_model":
            selection_rule = (
                "Highest CV Macro-F1 within FS1; then Weighted-F1, then Accuracy"
            )
        elif role.startswith("common_"):
            selection_rule = (
                "Classifier evaluated on the CV-selected common feature set; "
                "feature-set rule = mean Macro-F1 across all classifiers, "
                "then mean Weighted-F1, then mean Accuracy"
            )
        else:
            raise ValueError(f"Unknown model-selection role: {role}")

        selection_rows.append({
            "selection": role,
            "feature_set": feature_set,
            "model": model_name,
            "model_family": MODEL_FAMILIES[model_name],
            "macro_f1_cv": row["mean_macro_f1"],
            "macro_f1_cv_std": row["std_macro_f1"],
            "weighted_f1_cv": row["mean_weighted_f1"],
            "accuracy_cv": row["mean_accuracy"],
            "model_artifact": artifact,
            "selection_rule": selection_rule,
        })

if set(model_meta["models"]) != expected_model_artifacts:
    raise ValueError("Fitted-model metadata do not contain exactly the expected current artifacts.")

model_meta["status"] = "complete"
model_meta["completed_utc"] = datetime.now(timezone.utc).isoformat()
save_json(model_meta, MODEL_META_PATH)
model_selection = pd.DataFrame(selection_rows).sort_values("selection").reset_index(drop=True)
save_csv(model_selection, MODEL_SELECTION_PATH)


# %% 10 - Save reproducibility manifest

final_row = model_selection.loc[model_selection["selection"].eq("final_overall_model")]
if len(final_row) != 1:
    raise ValueError("Exactly one final overall model must be selected.")

screening_hashes = SCREENING_HASHES.copy()

artifact_paths = {
    "03a_screening_manifest": SCREENING_A_MANIFEST_PATH,
    "03b_representation_manifest": SCREENING_B_MANIFEST_PATH,
    "data_preparation_summary": DATA_PREPARATION_SUMMARY_PATH,
    "semantic_group_cohesion": DATA_PREPARATION_COHESION_PATH,
    "split": SPLIT_PATH,
    "folds": FOLD_PATH,
    "design_summary": DESIGN_SUMMARY_PATH,
    "search_candidates": SEARCH_CANDIDATES_PATH,
    "search_summary": SEARCH_SUMMARY_PATH,
    "parameters": PARAMETER_PATH,
    "tuning_checkpoint": TUNING_CHECKPOINT_PATH,
    "dummy_baseline": DUMMY_PATH,
    "oof": OOF_PATH,
    "oof_metadata": OOF_META_PATH,
    "cv_fold_metrics": FOLD_METRICS_PATH,
    "cv_fold_class_metrics": FOLD_CLASS_METRICS_PATH,
    "cv_results": CV_RESULTS_PATH,
    "feature_set_selection": FEATURE_SET_SELECTION_PATH,
    "cv_class_results": CV_CLASS_RESULTS_PATH,
    "model_selection": MODEL_SELECTION_PATH,
    "fitted_model_metadata": MODEL_META_PATH,
    "macro_f1_figure": FIGURE_PATH,
    "class_f1_figure": CLASS_F1_FIGURE_PATH,
    "stopwords": STOPWORD_PATH,
    "sentiment_features": SENTIMENT_PATH,
    "sentiment_metadata": SENTIMENT_META_PATH,
}

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "script": SCRIPT_PATH.name,
    "script_sha256": sha256(SCRIPT_PATH),
    "script_build": SCRIPT_BUILD,
    "analytical_design_id": ANALYTICAL_DESIGN_ID,
    "run_id": RUN_ID,
    "data_file": str(DATA_PATH.relative_to(PROJECT_ROOT)),
    "data_sha256": DATA_SHA256,
    "prepared_records": len(data),
    "semantic_groups": int(data[GROUP].nunique()),
    "exact_text_groups": int(data[EXACT_GROUP].nunique()),
    "training_records": len(training),
    "holdout_records": len(holdout),
    "training_semantic_groups": int(training[GROUP].nunique()),
    "holdout_semantic_groups": int(holdout[GROUP].nunique()),
    "semantic_conflict_groups": semantic_conflicts,
    "group_column": GROUP,
    "exact_group_column": EXACT_GROUP,
    "grouping_method": grouping_method,
    "minimum_group_similarity": minimum_group_similarity,
    "random_state": RANDOM_STATE,
    "cv_folds": N_SPLITS,
    "design_hash": DESIGN_HASH,
    "split_source": SPLIT_SOURCE,
    "cv_fold_source": FOLD_SOURCE,
    "primary_metric": "macro_f1",
    "selection_tiebreakers": ["weighted_f1", "accuracy"],
    "holdout_scored_in_modeling": False,
    "holdout_used_for_parameter_tuning": False,
    "holdout_used_for_model_selection": False,
    "holdout_used_for_preprocessing": False,
    "holdout_used_for_fitted_preprocessing": False,
    "sentiment_extracted_before_split_with_fixed_pretrained_model": True,
    "models": MODEL_ORDER,
    "model_families_descriptive_only": MODEL_FAMILIES,
    "feature_sets": FEATURE_SETS,
    "search_budget_per_model": SEARCH_BUDGET,
    "strict_data_freeze": STRICT_DATA_FREEZE,
    "reuse_valid_results": REUSE_VALID_RESULTS,
    "resume_partial_results": RESUME_PARTIAL_RESULTS,
    "parallel_cv_jobs": N_JOBS,
    "tuning_source": tuning_source,
    "tuning_checkpointing": "after_each_completed_classifier",
    "oof_source": oof_source,
    "oof_checkpointing": "after_each_model_feature_combination",
    "model_checkpointing": "after_each_fitted_model",
    "parameter_hash": PARAMETER_HASH,
    "oof_signature": OOF_SIGNATURE,
    "screening_decisions": {
        "retained_models": MODEL_ORDER,
        "excluded_after_training_only_screening": [
            "decision_tree", "adaboost", "catboost", "distilmbert_multilingual"
        ],
        "frozen_text_representation": "word_plus_character_tfidf",
        "screening_holdout_used": False,
        "screening_manifest_hashes": SCREENING_HASHES,
    },
    "selected_parameters": SELECTED_PARAMS,
    "text_representation": {
        "name": "word_plus_character_tfidf",
        "shared_by_all_models": True,
        "word_ngram_range": WORD_NGRAM_RANGE,
        "char_analyzer": "char_wb",
        "char_ngram_range": CHAR_NGRAM_RANGE,
        "word_max_features": WORD_MAX_FEATURES,
        "char_max_features": CHAR_MAX_FEATURES,
        "selected_text_features": SELECTED_TEXT_FEATURES,
        "bilingual_stopwords": True,
        "bilingual_stopword_count": len(BILINGUAL_STOPWORDS),
        "stopword_sha256": STOPWORD_SHA256,
        "stemming": False,
        "lemmatization": False,
    },
    "technical_baseline": {
        "model": "dummy_majority", "macro_f1_cv": dummy_metrics["macro_f1"],
        "hypothesis_baseline": False,
    },
    "common_feature_set_selection": {
        "selected_feature_set": COMMON_FEATURE_SET,
        "primary_metric": "mean cross-validated Macro-F1 across all seven classifiers",
        "tie_breakers": [
            "mean cross-validated Weighted-F1 across all seven classifiers",
            "mean cross-validated Accuracy across all seven classifiers",
            "predefined feature-set order",
        ],
        "source": "training-only cross-validation",
        "holdout_used": False,
        "selection_artifact": FEATURE_SET_SELECTION_PATH.name,
    },
    "hypothesis_operationalization": {
        "H1": {
            "feature_set": COMMON_FEATURE_SET,
            "feature_set_selection": "CV-selected common feature set",
            "baseline": "logistic_regression",
            "alternatives": [m for m in MODEL_ORDER if m != "logistic_regression"],
            "contrast": "Macro-F1(alternative) - Macro-F1(logistic_regression)",
            "pairwise": True,
            "family_mean_comparison": False,
            "multiple_testing": "Holm correction in Evaluation",
        },
        "H2": {
            "feature_set": COMMON_FEATURE_SET,
            "feature_set_selection": "same CV-selected common feature set as H1",
            "baseline": "logistic_regression",
            "comparison": "pairwise SHAP compactness against Logistic Regression",
            "family_mean_comparison": False,
            "eligible_alternatives": "defined in Explainability from technically comparable SHAP outputs",
        },
    },
    "sentiment": {
        "model_id": SENTIMENT_MODEL_ID,
        "revision": SENTIMENT_MODEL_REVISION,
        "text_hash": SENTIMENT_TEXT_HASH,
    },
    "figures": {
        "grayscale_only": True,
        "macro_f1_matrix": FIGURE_PATH.name,
        "class_f1_best_feature_set": CLASS_F1_FIGURE_PATH.name,
    },
    "final_model": {
        "feature_set": str(final_row.iloc[0]["feature_set"]),
        "model": str(final_row.iloc[0]["model"]),
    },
    "selected_models": model_selection.to_dict("records"),
    "artifact_hashes": {name: sha256(path) for name, path in artifact_paths.items()},
    "stale_model_artifacts": stale_model_artifacts,
    "selected_model_hashes": {
        artifact: details["sha256"] for artifact, details in model_meta["models"].items()
    },
    "package_versions": {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__, "joblib": joblib.__version__,
        "nltk": version_of("nltk"), "torch": version_of("torch"),
        "transformers": version_of("transformers"),
    },
}
save_json(manifest, MANIFEST_PATH)

section("CRISP-DM Modeling completed")
print(f"Script build              : {SCRIPT_BUILD}")
print(f"Algorithms                : {len(MODEL_ORDER)}")
print(f"Training records          : {fmt_int(len(training))}")
print(f"Hold-out records          : {fmt_int(len(holdout))}")
print(f"Common comparison set     : {COMMON_FEATURE_SET}")
print(f"Final model from CV       : {final_row.iloc[0]['feature_set']} + {MODEL_LABELS[str(final_row.iloc[0]['model'])]}")
print(f"Macro-F1 figure           : {FIGURE_PATH.name}")
print(f"Class-F1 figure           : {CLASS_F1_FIGURE_PATH.name}")
print(f"Strict data freeze        : {'YES' if STRICT_DATA_FREEZE else 'NO'}")
print(f"Tuning source             : {tuning_source}")
print(f"OOF source                : {oof_source}")
print(f"H1 comparison             : {COMMON_FEATURE_SET}, LR baseline vs each alternative (pairwise)")
print(f"Manifest                  : {MANIFEST_PATH.name}")
