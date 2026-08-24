"""SHAP-based explainability analysis for seven classifiers on the CV-selected common feature set and the final overall model."""

# %% 00 - Load packages

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import gc
import hashlib
import json
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
import shap
import sklearn
import xgboost as xgb
# %% 01 - Configure the explainability analysis

SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name.lower() != 'src':
    raise RuntimeError(f'Place this file in the existing <project>/src folder before running it. Current location: {SCRIPT_PATH}')
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'customer_it_support_prepared.csv'
MODELING_TABLE_DIR = PROJECT_ROOT / 'reports' / 'tables' / '03_tickets_modeling'
MODELING_MODEL_DIR = PROJECT_ROOT / 'models' / '03_tickets_modeling'
MODELING_CACHE_DIR = PROJECT_ROOT / 'cache' / '03_tickets_modeling'
EVALUATION_TABLE_DIR = PROJECT_ROOT / 'reports' / 'tables' / '04_evaluation'
EVALUATION_CACHE_DIR = PROJECT_ROOT / 'cache' / '04_evaluation'
TABLE_DIR = PROJECT_ROOT / 'reports' / 'tables' / '05_explainability'
FIGURE_DIR = PROJECT_ROOT / 'reports' / 'figures' / '05_explainability'
CACHE_DIR = PROJECT_ROOT / 'cache' / '05_explainability'
MODEL_CACHE_DIR = CACHE_DIR / 'models'
for directory in [TABLE_DIR, FIGURE_DIR, CACHE_DIR, MODEL_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
RUN_ID = 'explainability_common_feature_selection_v1'
SCRIPT_BUILD = '2026-08-16_explainability_common_feature_selection_v1'
EXPECTED_MODELING_BUILD = '2026-08-16_final_comparative_common_feature_selection_v1'
EXPECTED_MODELING_RUN_ID = 'final_comparative_common_feature_selection_v1'
EXPECTED_EVALUATION_BUILD = '2026-08-16_evaluation_common_feature_selection_v1'
EXPECTED_EVALUATION_RUN_ID = 'evaluation_common_feature_selection_v1'
REUSE_VALID_RESULTS = True
RANDOM_STATE = 42
ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_BATCH_SIZE = 250
GLOBAL_GROUPS_PER_CLASS = 100
FAITHFULNESS_GROUPS_PER_CLASS = 50
XGB_SENSITIVITY_GROUPS_PER_CLASS = 30
BACKGROUND_GROUPS_PER_CLASS = 50
LOCAL_CASE_TARGET = 12
GLOBAL_TOP_FEATURES = 50
LOCAL_TOP_POSITIVE = 10
LOCAL_TOP_NEGATIVE = 10
TOP_K_VALUES = (5, 10, 20)
PRIMARY_TOP_K = 10
MASS_THRESHOLDS = (0.5, 0.8, 0.9)
FAITHFULNESS_TOP_K_VALUES = (10,)
FAITHFULNESS_RANDOM_REPEATS = 5
SHAP_ZERO_TOLERANCE = 1e-12
EXPECTED_RECORDS = 28551
EXPECTED_SEMANTIC_GROUPS = 20825
EXPECTED_TRAINING_RECORDS = 22842
EXPECTED_TRAINING_GROUPS = 16661
EXPECTED_HOLDOUT_RECORDS = 5709
EXPECTED_HOLDOUT_GROUPS = 4164
TARGET_COLUMN = 'priority'
GROUP_COLUMN = 'semantic_group_id'
EXACT_GROUP_COLUMN = 'text_group_id'
TEXT_COLUMN = 'text'
TFIDF_TEXT_COLUMN = 'text_clean'
CLASS_ORDER = ['low', 'medium', 'high']
CLASS_TO_INT = {label: index for index, label in enumerate(CLASS_ORDER)}
ALL_MODEL_ORDER = ['logistic_regression', 'linear_svm', 'complement_naive_bayes', 'knn_cosine', 'random_forest', 'xgboost', 'lightgbm']
XAI_MODEL_ORDER = ALL_MODEL_ORDER.copy()
MODEL_ORDER = XAI_MODEL_ORDER
H2_BASELINE_MODEL = 'logistic_regression'
H2_ALTERNATIVES = [model for model in XAI_MODEL_ORDER if model != H2_BASELINE_MODEL]
MODEL_FAMILIES = {'logistic_regression': 'linear_baseline', 'linear_svm': 'linear', 'complement_naive_bayes': 'naive_bayes', 'knn_cosine': 'instance_based', 'random_forest': 'bagging_ensemble', 'xgboost': 'gradient_boosting', 'lightgbm': 'gradient_boosting'}
MODEL_LABELS = {'logistic_regression': 'Logistic Regression', 'linear_svm': 'Linear SVM', 'complement_naive_bayes': 'Complement Naive Bayes', 'knn_cosine': 'kNN (cosine)', 'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM'}
LINEAR_MODELS = ['logistic_regression', 'linear_svm']
ADDITIVE_SCORE_MODELS = ['complement_naive_bayes']
MODEL_AGNOSTIC_MODELS = ['knn_cosine']
TREE_MODELS = ['random_forest', 'xgboost', 'lightgbm']
APPROXIMATE_TREE_MODELS = ['random_forest', 'xgboost']
KNN_PERMUTATION_CYCLES = 3
KNN_BACKGROUND_GROUPS_PER_CLASS = 3
PREVIOUS_XAI_BUILD = '2026-08-13_explainability_all7_plus_final_model_v4'
PREVIOUS_XAI_BUILD_V3 = '2026-08-13_explainability_all7_submission_v3'
LEGACY_XAI_BUILD = '2026-08-13_explainability_all7_submission_v2'
LEGACY_KNN_PERMUTATION_CYCLES = 1
LEGACY_KNN_BACKGROUND_GROUPS_PER_CLASS = 1
VALID_FEATURE_SETS = {'FS1', 'FS2', 'FS3', 'FS4', 'FS5'}
FEATURE_GROUP_ORDER = []
FINAL_FEATURE_GROUP_ORDER = []
SENTIMENT_COLUMNS = ['sentiment_negative', 'sentiment_neutral', 'sentiment_positive']
H2_PRIMARY_METRIC = f'predicted-class full-attribution Top-{PRIMARY_TOP_K} absolute SHAP-mass concentration'
MODEL_SELECTION_PATH = MODELING_TABLE_DIR / '03_modeling_model_selection.csv'
FEATURE_SET_SELECTION_PATH = MODELING_TABLE_DIR / '03_modeling_feature_set_selection.csv'
MODELING_MANIFEST_PATH = MODELING_TABLE_DIR / '03_modeling_manifest.json'
FITTED_MODEL_METADATA_PATH = MODELING_MODEL_DIR / '03_fitted_model_metadata.json'
SPLIT_PATH = MODELING_CACHE_DIR / '03_split_assignments.parquet'
EVALUATION_MANIFEST_PATH = EVALUATION_TABLE_DIR / '04_evaluation_manifest.json'
HOLDOUT_PREDICTIONS_PATH = EVALUATION_CACHE_DIR / '04_holdout_predictions.parquet'
SENTIMENT_PATH = MODELING_CACHE_DIR / '03_sentiment_features.csv'
SENTIMENT_META_PATH = MODELING_CACHE_DIR / '03_sentiment_features_metadata.json'
XAI_ELIGIBILITY_PATH = TABLE_DIR / '05_xai_model_eligibility.csv'
XAI_SAMPLE_PATH = TABLE_DIR / '05_xai_sample.csv'
BACKGROUND_SAMPLE_PATH = CACHE_DIR / '05_linear_shap_background_sample.parquet'
KNN_BACKGROUND_SAMPLE_PATH = CACHE_DIR / '05_knn_permutation_background_sample.parquet'
FEATURE_DICTIONARY_SUMMARY_PATH = TABLE_DIR / '05_feature_dictionary_summary.csv'
COMPACTNESS_CASE_PATH = CACHE_DIR / '05_shap_compactness_by_case.parquet'
COMPACTNESS_SUMMARY_PATH = TABLE_DIR / '05_shap_compactness_summary.csv'
H2_RESULTS_PATH = TABLE_DIR / '05_h2_pairwise_results.csv'
H2_SENSITIVITY_PATH = TABLE_DIR / '05_h2_sensitivity_results.csv'
GLOBAL_FEATURES_PATH = TABLE_DIR / '05_global_top_features.csv'
FEATURE_GROUPS_PATH = TABLE_DIR / '05_feature_group_summary.csv'
LOCAL_CASES_PATH = TABLE_DIR / '05_local_explanation_cases.csv'
LOCAL_CONTRIBUTIONS_PATH = TABLE_DIR / '05_local_feature_contributions.csv'
FAITHFULNESS_CASE_PATH = CACHE_DIR / '05_faithfulness_by_case.parquet'
FAITHFULNESS_SUMMARY_PATH = TABLE_DIR / '05_faithfulness_summary.csv'
XGB_SENSITIVITY_CASE_PATH = CACHE_DIR / '05_xgboost_shap_sensitivity_by_case.parquet'
XGB_SENSITIVITY_SUMMARY_PATH = TABLE_DIR / '05_xgboost_shap_sensitivity_summary.csv'
QUALITATIVE_TEMPLATE_PATH = TABLE_DIR / '05_qualitative_plausibility_assessment_template.csv'
VALIDATION_SUMMARY_PATH = TABLE_DIR / '05_shap_validation_summary.csv'
MANIFEST_PATH = TABLE_DIR / '05_explainability_manifest.json'
PACKAGE_VERSIONS = {'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'shap': shap.__version__, 'xgboost': xgb.__version__, 'lightgbm': lgb.__version__, 'joblib': joblib.__version__}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8})

# %% 02 - Define helper functions

def fmt_int(value):
    return f'{int(value):,}'.replace(',', '.')

def print_section(title):
    print(f"\n{title}\n{'-' * len(title)}")

def load_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

def save_json(payload, path):
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def stable_hash(payload):
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def save_png(figure, filename):
    figure.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close(figure)

def ensure_csr(matrix):
    output = matrix.tocsr().astype(np.float32) if sp.issparse(matrix) else sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    if not np.isfinite(output.data).all():
        raise ValueError('The transformed feature matrix contains non-finite values.')
    return output

def standardize_shap_output(raw_values, raw_base, n_samples, n_features, n_classes):
    if isinstance(raw_values, list):
        values = np.stack([np.asarray(item) for item in raw_values], axis=1)
    else:
        values = np.asarray(raw_values)
    if values.ndim == 2:
        if n_classes != 1:
            raise ValueError(f'Unexpected two-dimensional SHAP output: {values.shape}')
        values = values[:, None, :]
    elif values.ndim == 3:
        if values.shape == (n_samples, n_features, n_classes):
            values = np.transpose(values, (0, 2, 1))
        elif values.shape == (n_classes, n_samples, n_features):
            values = np.transpose(values, (1, 0, 2))
        elif values.shape != (n_samples, n_classes, n_features):
            raise ValueError(f'Unsupported SHAP output shape: {values.shape}')
    else:
        raise ValueError(f'Unsupported SHAP output dimensions: {values.shape}')
    base = np.asarray(raw_base)
    if base.ndim == 0:
        base = np.full((n_samples, n_classes), float(base))
    elif base.ndim == 1:
        if len(base) == n_classes:
            base = np.broadcast_to(base, (n_samples, n_classes))
        elif len(base) == n_samples and n_classes == 1:
            base = base[:, None]
        else:
            raise ValueError(f'Unsupported SHAP base-value shape: {base.shape}')
    elif base.ndim == 2:
        if base.shape == (n_samples, n_classes):
            pass
        elif base.shape == (n_classes, n_samples):
            base = base.T
        else:
            raise ValueError(f'Unsupported SHAP base-value shape: {base.shape}')
    else:
        raise ValueError(f'Unsupported SHAP base-value dimensions: {base.shape}')
    values = np.asarray(values, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('SHAP values contain non-finite values before storage conversion.')
    if not np.isfinite(base).all():
        raise ValueError('SHAP base values contain non-finite values.')
    return (values, base)

def score_matrix(estimator, matrix, model_name, shap_scale=False):
    if model_name in LINEAR_MODELS:
        scores = estimator.decision_function(matrix)
    elif model_name == 'complement_naive_bayes':
        scores = matrix @ np.asarray(estimator.feature_log_prob_, dtype=np.float64).T
    elif model_name == 'knn_cosine':
        scores = estimator.predict_proba(matrix)
    elif model_name == 'random_forest':
        scores = estimator.predict_proba(matrix)
    elif model_name == 'xgboost':
        scores = estimator.predict(matrix, output_margin=True) if shap_scale else estimator.predict_proba(matrix)
    elif model_name == 'lightgbm':
        scores = estimator.booster_.predict(matrix, raw_score=True) if shap_scale else estimator.predict_proba(matrix)
    else:
        raise ValueError(f'Unsupported SHAP model: {model_name}')
    scores = np.asarray(scores, dtype=np.float64)
    return np.column_stack([-scores, scores]) if scores.ndim == 1 else scores

def extract_feature_dictionary(preprocessor, model_name, feature_group_order=None):
    feature_group_order = feature_group_order or FEATURE_GROUP_ORDER
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    groups, labels = ([], [])
    for feature_name in names:
        name = str(feature_name)
        if name.startswith('text__'):
            group = 'text'
            label = name.split('text__', 1)[1]
            if label.startswith('word__'):
                label = label.split('word__', 1)[1]
            elif label.startswith('char__'):
                label = '[char] ' + label.split('char__', 1)[1]
        elif name.startswith('categorical__'):
            matches = [group_name for group_name in feature_group_order if group_name != 'text' and group_name != 'sentiment' and name.startswith(f'categorical__{group_name}_')]
            if len(matches) != 1:
                raise ValueError(f'Unable to assign transformed categorical feature: {name}')
            group = matches[0]
            label = name[len(f'categorical__{group}_'):]
        elif name.startswith('sentiment__'):
            group = 'sentiment'
            label = name.split('sentiment__', 1)[1]
        else:
            raise ValueError(f'Unable to assign transformed feature to a feature group: {name}')
        groups.append(group)
        labels.append(label)
    return pd.DataFrame({'model': model_name, 'model_label': MODEL_LABELS[model_name], 'model_family': MODEL_FAMILIES[model_name], 'feature_index': np.arange(len(names), dtype=np.int32), 'feature_name': names, 'feature_label': labels, 'feature_group': groups})

def calculate_shap_values(model_name, estimator, analysis_matrix, background_matrix, knn_background_matrix=None):
    n_samples, n_features = analysis_matrix.shape
    n_classes = len(CLASS_ORDER)
    if not np.array_equal(np.asarray(estimator.classes_, dtype=np.int8), np.arange(n_classes, dtype=np.int8)):
        raise ValueError(f'Unexpected classifier class order for {model_name}: {estimator.classes_}')

    if model_name in LINEAR_MODELS:
        masker = shap.maskers.Independent(background_matrix, max_samples=background_matrix.shape[0])
        explainer = shap.LinearExplainer(estimator, masker)
        try:
            explanation = explainer(analysis_matrix)
            raw_values = explanation.values
            raw_base = explanation.base_values
            explainer_name = 'LinearExplainer'
        except TypeError:
            raw_values = explainer.shap_values(analysis_matrix)
            raw_base = explainer.expected_value
            explainer_name = 'LinearExplainer legacy API'
        values, base_values = standardize_shap_output(raw_values, raw_base, n_samples, n_features, n_classes)
        background_assumption = 'independent_background'
        approximate = False
        output_scale = 'decision_function'

    elif model_name == 'complement_naive_bayes':
        coefficients = np.asarray(estimator.feature_log_prob_, dtype=np.float64)
        intercept = np.zeros(n_classes, dtype=np.float64)
        masker = shap.maskers.Independent(background_matrix, max_samples=background_matrix.shape[0])
        explainer = shap.LinearExplainer((coefficients, intercept), masker)
        try:
            explanation = explainer(analysis_matrix)
            raw_values = explanation.values
            raw_base = explanation.base_values
            explainer_name = 'LinearExplainer on ComplementNB class scores'
        except TypeError:
            raw_values = explainer.shap_values(analysis_matrix)
            raw_base = explainer.expected_value
            explainer_name = 'LinearExplainer on ComplementNB class scores legacy API'
        values, base_values = standardize_shap_output(raw_values, raw_base, n_samples, n_features, n_classes)
        background_assumption = 'independent_background'
        approximate = False
        output_scale = 'ComplementNB additive class score'

    elif model_name == 'knn_cosine':
        if knn_background_matrix is None:
            raise ValueError('kNN requires the fixed class-balanced permutation background.')

        knn_background_dense = np.asarray(
            knn_background_matrix.toarray(),
            dtype=np.float32,
        )
        analysis_dense = np.asarray(
            analysis_matrix.toarray(),
            dtype=np.float32,
        )

        def knn_probability_function(masked_matrix):
            return np.asarray(
                estimator.predict_proba(
                    sp.csr_matrix(masked_matrix)
                ),
                dtype=np.float64,
            )

        explainer = shap.PermutationExplainer(
            knn_probability_function,
            knn_background_dense,
            seed=RANDOM_STATE,
        )
        values = np.zeros(
            (n_samples, n_classes, n_features),
            dtype=np.float64,
        )
        base_values = np.zeros(
            (n_samples, n_classes),
            dtype=np.float64,
        )

        print(
            f'kNN permutation SHAP      : {n_samples} cases, '
            f'{knn_background_dense.shape[0]} background groups, '
            f'{KNN_PERMUTATION_CYCLES} antithetic cycle(s)'
        )

        for row_index in range(n_samples):
            row = analysis_dense[row_index]
            varying = np.any(
                ~np.isclose(
                    knn_background_dense,
                    row[None, :],
                    rtol=0.0,
                    atol=1e-12,
                ),
                axis=0,
            )
            varying_features = int(varying.sum())

            if varying_features == 0:
                base_values[row_index] = knn_probability_function(
                    row.reshape(1, -1)
                )[0]
                continue

            max_evals = (
                KNN_PERMUTATION_CYCLES
                * (2 * varying_features + 1)
            )
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
                    'Unexpected kNN Permutation SHAP value shape: '
                    f'{row_values.shape}'
                )

            values[row_index] = row_values
            base_values[row_index] = np.asarray(
                explanation.base_values[0],
                dtype=np.float64,
            )

            if (
                (row_index + 1) % 25 == 0
                or row_index + 1 == n_samples
            ):
                print(
                    f'kNN SHAP progress         : '
                    f'{row_index + 1}/{n_samples}'
                )

        explainer_name = 'PermutationExplainer'
        background_assumption = (
            f'model-agnostic permutation SHAP with '
            f'{knn_background_matrix.shape[0]} class-balanced training '
            f'background groups'
        )
        approximate = True
        output_scale = 'class_probability'

    elif model_name in {'random_forest', 'xgboost'}:
        explainer = shap.TreeExplainer(estimator, feature_perturbation='tree_path_dependent', model_output='raw')
        call_matrix = np.asarray(analysis_matrix.toarray(), dtype=np.float64) if model_name == 'random_forest' else analysis_matrix
        try:
            explanation = explainer(call_matrix, check_additivity=False, approximate=True)
            raw_values = explanation.values
            raw_base = explanation.base_values
            explainer_name = 'TreeExplainer approximate'
        except TypeError:
            raw_values = explainer.shap_values(call_matrix, check_additivity=False, approximate=True)
            raw_base = explainer.expected_value
            explainer_name = 'TreeExplainer approximate legacy API'
        values, base_values = standardize_shap_output(raw_values, raw_base, n_samples, n_features, n_classes)
        background_assumption = 'tree_path_dependent'
        approximate = True
        output_scale = 'class_probability' if model_name == 'random_forest' else 'raw_margin'

    elif model_name == 'lightgbm':
        raw_contributions = estimator.booster_.predict(analysis_matrix, pred_contrib=True)
        if isinstance(raw_contributions, list):
            if len(raw_contributions) != n_classes:
                raise ValueError('Unexpected LightGBM native SHAP contribution structure.')
            contribution_matrices = raw_contributions
        else:
            raw_array = np.asarray(raw_contributions)
            if raw_array.shape == (n_samples, n_classes * (n_features + 1)):
                reshaped = raw_array.reshape(n_samples, n_classes, n_features + 1)
                contribution_matrices = [reshaped[:, class_index, :] for class_index in range(n_classes)]
            elif raw_array.shape == (n_samples, n_classes, n_features + 1):
                contribution_matrices = [raw_array[:, class_index, :] for class_index in range(n_classes)]
            else:
                raise ValueError(f'Unexpected LightGBM native SHAP contribution shape: {raw_array.shape}')
        values = np.empty((n_samples, n_classes, n_features), dtype=np.float64)
        base_values = np.empty((n_samples, n_classes), dtype=np.float64)
        for class_index, contribution_matrix in enumerate(contribution_matrices):
            contribution_matrix = contribution_matrix.toarray() if sp.issparse(contribution_matrix) else np.asarray(contribution_matrix)
            if contribution_matrix.shape != (n_samples, n_features + 1):
                raise ValueError(f'Unexpected LightGBM native SHAP contribution shape: {contribution_matrix.shape}')
            values[:, class_index, :] = contribution_matrix[:, :-1]
            base_values[:, class_index] = contribution_matrix[:, -1]
        explainer_name = 'LightGBM native TreeSHAP'
        background_assumption = 'tree_path_dependent'
        approximate = False
        output_scale = 'raw_margin'
    else:
        raise ValueError(f'Unsupported SHAP model: {model_name}')

    if not np.isfinite(values).all() or not np.isfinite(base_values).all():
        raise ValueError(f'Non-finite SHAP outputs for {model_name}.')
    target_scores = score_matrix(estimator, analysis_matrix, model_name, shap_scale=True)
    reconstructed = base_values + values.sum(axis=2, dtype=np.float64)
    max_error = float(np.max(np.abs(reconstructed - target_scores)))
    max_abs_value = float(np.max(np.abs(values)))
    tolerance = 0.001 if model_name in {'xgboost', 'lightgbm'} else 0.0001
    if max_error > tolerance:
        raise ValueError(
            f'SHAP additivity check failed for {model_name}: '
            f'max error={max_error:.8f}, tolerance={tolerance:.8f}.'
        )
    predicted = np.asarray(estimator.predict(analysis_matrix), dtype=np.int8)
    if not np.array_equal(predicted, reconstructed.argmax(axis=1).astype(np.int8)):
        raise ValueError(f'Reconstructed SHAP outputs do not reproduce predictions for {model_name}.')
    stored_values = values.astype(np.float32)
    if not np.isfinite(stored_values).all():
        raise ValueError(f'SHAP values for {model_name} overflow during float32 storage.')
    return stored_values, base_values, {
        'explainer': explainer_name,
        'background_assumption': background_assumption,
        'approximate_shap': approximate,
        'max_abs_shap_value_before_storage': max_abs_value,
        'max_additivity_error': max_error,
        'additivity_tolerance': tolerance,
        'prediction_reconstruction_matches': True,
        'shap_output_scale': output_scale,
    }
def metrics_for_vector(values, available_features):
    vector = np.asarray(values, dtype=np.float64)
    available_features = int(available_features)
    total = float(vector.sum())
    result = {'attribution_mass': total, 'available_features': available_features, 'nonzero_shap_features': int(np.sum(vector > SHAP_ZERO_TOLERANCE)), **{f'top_{top_k}_share': np.nan for top_k in TOP_K_VALUES}, **{f'k_{int(threshold * 100)}': np.nan for threshold in MASS_THRESHOLDS}, 'normalized_entropy': np.nan}
    if total <= SHAP_ZERO_TOLERANCE or available_features == 0:
        return result
    shares = np.sort(vector)[::-1] / total
    cumulative = np.cumsum(shares)
    for top_k in TOP_K_VALUES:
        result[f'top_{top_k}_share'] = float(shares[:top_k].sum())
    for threshold in MASS_THRESHOLDS:
        result[f'k_{int(threshold * 100)}'] = int(np.searchsorted(cumulative, threshold) + 1)
    positive = shares[shares > 0]
    if available_features <= 1:
        result['normalized_entropy'] = 0.0
    else:
        result['normalized_entropy'] = -float(np.sum(positive * np.log(positive))) / np.log(available_features)
    return result

def calculate_compactness(model_name, values, analysis_matrix, analysis_data):
    predictions = analysis_data[f'pred_{model_name}'].to_numpy(dtype=np.int8)
    absolute_values = np.abs(values[np.arange(len(analysis_data)), predictions, :])
    rows = []
    for row_index, row in enumerate(analysis_data.itertuples(index=False)):
        common = {'model': model_name, 'model_label': MODEL_LABELS[model_name], 'model_family': MODEL_FAMILIES[model_name], 'modeling_row_id': int(row.modeling_row_id), GROUP_COLUMN: getattr(row, GROUP_COLUMN), 'actual_class': getattr(row, TARGET_COLUMN), 'predicted_class': CLASS_ORDER[predictions[row_index]], 'correct': bool(predictions[row_index] == getattr(row, 'priority_encoded')), 'analysis_role': row.analysis_role}
        rows.append({**common, 'scope': 'full_attribution', **metrics_for_vector(absolute_values[row_index], absolute_values.shape[1])})
        active_indices = analysis_matrix.getrow(row_index).indices
        rows.append({**common, 'scope': 'active_features', **metrics_for_vector(absolute_values[row_index, active_indices], len(active_indices))})
    return pd.DataFrame(rows)

def calculate_explanation_tables(model_name, values, analysis_matrix, analysis_data, feature_dictionary, base_indices):
    predictions = analysis_data[f'pred_{model_name}'].to_numpy(dtype=np.int8)
    feature_groups = feature_dictionary['feature_group'].to_numpy(dtype=object)
    group_masks = {group: feature_groups == group for group in FEATURE_GROUP_ORDER}
    global_rows, group_rows, local_rows = ([], [], [])
    output_vectors = [(label, np.mean(np.abs(values[base_indices, class_index, :]), axis=0)) for class_index, label in enumerate(CLASS_ORDER)]
    predicted_values = values[base_indices, predictions[base_indices], :]
    output_vectors.append(('predicted_class', np.mean(np.abs(predicted_values), axis=0)))
    for class_output, mean_absolute in output_vectors:
        total = float(mean_absolute.sum())
        for rank, feature_index in enumerate(np.argsort(mean_absolute)[::-1][:GLOBAL_TOP_FEATURES], start=1):
            feature = feature_dictionary.iloc[int(feature_index)]
            global_rows.append({'model': model_name, 'model_label': MODEL_LABELS[model_name], 'model_family': MODEL_FAMILIES[model_name], 'class_output': class_output, 'rank': rank, 'feature_index': int(feature_index), 'feature_name': feature.feature_name, 'feature_label': feature.feature_label, 'feature_group': feature.feature_group, 'mean_abs_shap': float(mean_absolute[feature_index]), 'share_of_total_abs_shap': float(mean_absolute[feature_index] / total) if total else np.nan})
    for class_index, class_output in enumerate(CLASS_ORDER):
        mean_absolute = np.mean(np.abs(values[base_indices, class_index, :]), axis=0)
        total = float(mean_absolute.sum())
        for group in FEATURE_GROUP_ORDER:
            mass = float(mean_absolute[group_masks[group]].sum())
            group_rows.append({'model': model_name, 'model_label': MODEL_LABELS[model_name], 'model_family': MODEL_FAMILIES[model_name], 'result_level': 'global_class_output', 'class_output': class_output, 'modeling_row_id': np.nan, GROUP_COLUMN: '', 'feature_group': group, 'absolute_shap_mass': mass, 'share_of_total_abs_shap': mass / total if total else np.nan})
    case_values = np.abs(values[np.arange(len(analysis_data)), predictions, :])
    for row_index in base_indices:
        total = float(case_values[row_index].sum())
        for group in FEATURE_GROUP_ORDER:
            mass = float(case_values[row_index, group_masks[group]].sum())
            group_rows.append({'model': model_name, 'model_label': MODEL_LABELS[model_name], 'model_family': MODEL_FAMILIES[model_name], 'result_level': 'case_predicted_class', 'class_output': CLASS_ORDER[predictions[row_index]], 'modeling_row_id': int(analysis_data.iloc[row_index]['modeling_row_id']), GROUP_COLUMN: analysis_data.iloc[row_index][GROUP_COLUMN], 'feature_group': group, 'absolute_shap_mass': mass, 'share_of_total_abs_shap': mass / total if total else np.nan})
    for row_index in np.flatnonzero(analysis_data['is_local_case'].to_numpy()):
        predicted_class_index = int(predictions[row_index])
        shap_vector = values[row_index, predicted_class_index, :]
        active_indices = analysis_matrix.getrow(row_index).indices
        active_shap = shap_vector[active_indices]
        positive = [index for index in active_indices[np.argsort(active_shap)[::-1]] if shap_vector[index] > 0][:LOCAL_TOP_POSITIVE]
        negative = [index for index in active_indices[np.argsort(active_shap)] if shap_vector[index] < 0][:LOCAL_TOP_NEGATIVE]
        for direction, indices in [('supports_prediction', positive), ('opposes_prediction', negative)]:
            for rank, feature_index in enumerate(indices, start=1):
                feature = feature_dictionary.iloc[int(feature_index)]
                local_rows.append({'case_id': analysis_data.iloc[row_index]['case_id'], 'case_type': analysis_data.iloc[row_index]['case_type'], 'model': model_name, 'model_label': MODEL_LABELS[model_name], 'model_family': MODEL_FAMILIES[model_name], 'modeling_row_id': int(analysis_data.iloc[row_index]['modeling_row_id']), GROUP_COLUMN: analysis_data.iloc[row_index][GROUP_COLUMN], 'actual_class': analysis_data.iloc[row_index][TARGET_COLUMN], 'predicted_class': CLASS_ORDER[predicted_class_index], 'direction': direction, 'rank': rank, 'feature_index': int(feature_index), 'feature_name': feature.feature_name, 'feature_label': feature.feature_label, 'feature_group': feature.feature_group, 'feature_value': float(analysis_matrix[row_index, int(feature_index)]), 'shap_value': float(shap_vector[feature_index]), 'abs_shap_value': float(abs(shap_vector[feature_index]))})
    return (pd.DataFrame(global_rows), pd.DataFrame(group_rows), pd.DataFrame(local_rows))


def calculate_final_model_tables(
    values,
    analysis_matrix,
    analysis_data,
    feature_dictionary,
    base_indices,
    prediction_column,
    feature_set,
    model_name,
):
    predictions = analysis_data[prediction_column].to_numpy(dtype=np.int8)
    feature_groups = feature_dictionary["feature_group"].to_numpy(dtype=object)
    group_masks = {
        group: feature_groups == group
        for group in FINAL_FEATURE_GROUP_ORDER
    }

    global_rows = []
    group_rows = []
    local_rows = []

    output_vectors = [
        (
            label,
            np.mean(
                np.abs(values[base_indices, class_index, :]),
                axis=0,
            ),
        )
        for class_index, label in enumerate(CLASS_ORDER)
    ]
    predicted_values = values[
        base_indices,
        predictions[base_indices],
        :,
    ]
    output_vectors.append(
        (
            "predicted_class",
            np.mean(np.abs(predicted_values), axis=0),
        )
    )

    for class_output, mean_absolute in output_vectors:
        total = float(mean_absolute.sum())

        for rank, feature_index in enumerate(
            np.argsort(mean_absolute)[::-1][:GLOBAL_TOP_FEATURES],
            start=1,
        ):
            feature = feature_dictionary.iloc[int(feature_index)]
            global_rows.append({
                "feature_set": feature_set,
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "class_output": class_output,
                "rank": rank,
                "feature_index": int(feature_index),
                "feature_name": feature.feature_name,
                "feature_label": feature.feature_label,
                "feature_group": feature.feature_group,
                "mean_abs_shap": float(mean_absolute[feature_index]),
                "share_of_total_abs_shap": (
                    float(mean_absolute[feature_index] / total)
                    if total
                    else np.nan
                ),
            })

        for group in FINAL_FEATURE_GROUP_ORDER:
            mass = float(mean_absolute[group_masks[group]].sum())
            group_rows.append({
                "feature_set": feature_set,
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "class_output": class_output,
                "feature_group": group,
                "absolute_shap_mass": mass,
                "share_of_total_abs_shap": (
                    mass / total
                    if total
                    else np.nan
                ),
            })

    for row_index in np.flatnonzero(
        analysis_data["is_local_case"].to_numpy()
    ):
        predicted_class_index = int(predictions[row_index])
        shap_vector = values[
            row_index,
            predicted_class_index,
            :,
        ]
        active_indices = analysis_matrix.getrow(row_index).indices
        active_shap = shap_vector[active_indices]

        positive = [
            index
            for index in active_indices[
                np.argsort(active_shap)[::-1]
            ]
            if shap_vector[index] > 0
        ][:LOCAL_TOP_POSITIVE]

        negative = [
            index
            for index in active_indices[
                np.argsort(active_shap)
            ]
            if shap_vector[index] < 0
        ][:LOCAL_TOP_NEGATIVE]

        for direction, indices in [
            ("supports_prediction", positive),
            ("opposes_prediction", negative),
        ]:
            for rank, feature_index in enumerate(indices, start=1):
                feature = feature_dictionary.iloc[int(feature_index)]
                local_rows.append({
                    "case_id": analysis_data.iloc[row_index]["case_id"],
                    "case_type": analysis_data.iloc[row_index]["case_type"],
                    "feature_set": feature_set,
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "modeling_row_id": int(
                        analysis_data.iloc[row_index]["modeling_row_id"]
                    ),
                    GROUP_COLUMN: analysis_data.iloc[row_index][GROUP_COLUMN],
                    "actual_class": analysis_data.iloc[row_index][TARGET_COLUMN],
                    "predicted_class": CLASS_ORDER[predicted_class_index],
                    "direction": direction,
                    "rank": rank,
                    "feature_index": int(feature_index),
                    "feature_name": feature.feature_name,
                    "feature_label": feature.feature_label,
                    "feature_group": feature.feature_group,
                    "feature_value": float(
                        analysis_matrix[row_index, int(feature_index)]
                    ),
                    "shap_value": float(shap_vector[feature_index]),
                    "abs_shap_value": float(abs(shap_vector[feature_index])),
                })

    return (
        pd.DataFrame(global_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(local_rows),
    )


def zero_selected_features(matrix, selected_indices):
    modified = matrix.tolil(copy=True)
    for row_index, indices in enumerate(selected_indices):
        if len(indices):
            modified[row_index, list(indices)] = 0.0
    output = modified.tocsr()
    output.eliminate_zeros()
    return output

def calculate_faithfulness(model_name, estimator, values, analysis_matrix, analysis_data, feature_groups, base_indices):
    base_matrix = analysis_matrix[base_indices].tocsr()
    base_data = analysis_data.iloc[base_indices].reset_index(drop=True)
    predictions = base_data[f'pred_{model_name}'].to_numpy(dtype=np.int8)
    predicted_values = values[base_indices, predictions, :]
    text_feature_mask = feature_groups == 'text'
    original_scores = score_matrix(estimator, base_matrix, model_name, shap_scale=False)
    original_predicted_scores = original_scores[np.arange(len(base_data)), predictions]
    active_text_indices = [active[text_feature_mask[active]] for active in (base_matrix.getrow(row_index).indices for row_index in range(len(base_data)))]
    rows = []
    model_seed = RANDOM_STATE + MODEL_ORDER.index(model_name) * 10000
    for top_k in FAITHFULNESS_TOP_K_VALUES:
        top_selections = []
        for row_index, active_indices in enumerate(active_text_indices):
            order = np.argsort(np.abs(predicted_values[row_index, active_indices]))[::-1]
            top_selections.append(active_indices[order[:top_k]].astype(np.int32))
        top_scores = score_matrix(estimator, zero_selected_features(base_matrix, top_selections), model_name, shap_scale=False)
        top_drop = original_predicted_scores - top_scores[np.arange(len(base_data)), predictions]
        top_flip = top_scores.argmax(axis=1).astype(np.int8) != predictions
        random_drops = np.zeros((FAITHFULNESS_RANDOM_REPEATS, len(base_data)), dtype=np.float64)
        random_flips = np.zeros((FAITHFULNESS_RANDOM_REPEATS, len(base_data)), dtype=np.float64)
        for repeat in range(FAITHFULNESS_RANDOM_REPEATS):
            rng = np.random.default_rng(model_seed + top_k * 100 + repeat)
            random_selections = [np.asarray(rng.choice(active, size=len(top), replace=False), dtype=np.int32) if len(top) else np.array([], dtype=np.int32) for active, top in zip(active_text_indices, top_selections)]
            random_scores = score_matrix(estimator, zero_selected_features(base_matrix, random_selections), model_name, shap_scale=False)
            random_drops[repeat] = original_predicted_scores - random_scores[np.arange(len(base_data)), predictions]
            random_flips[repeat] = random_scores.argmax(axis=1).astype(np.int8) != predictions
        random_drop_mean = random_drops.mean(axis=0)
        random_flip_mean = random_flips.mean(axis=0)
        for row_index, row in enumerate(base_data.itertuples(index=False)):
            rows.append({'model': model_name, 'model_label': MODEL_LABELS[model_name], 'model_family': MODEL_FAMILIES[model_name], 'modeling_row_id': int(row.modeling_row_id), GROUP_COLUMN: getattr(row, GROUP_COLUMN), 'actual_class': getattr(row, TARGET_COLUMN), 'predicted_class': CLASS_ORDER[predictions[row_index]], 'top_k': top_k, 'perturbed_active_text_features': len(top_selections[row_index]), 'comprehensiveness_top': float(top_drop[row_index]), 'flip_top': bool(top_flip[row_index]), 'comprehensiveness_random_mean': float(random_drop_mean[row_index]), 'flip_random_mean': float(random_flip_mean[row_index]), 'comprehensiveness_advantage': float(top_drop[row_index] - random_drop_mean[row_index]), 'top_stronger_than_random': bool(top_drop[row_index] > random_drop_mean[row_index]), 'random_repeats': FAITHFULNESS_RANDOM_REPEATS, 'score_scale': ('decision_function' if model_name in LINEAR_MODELS else 'ComplementNB class score' if model_name == 'complement_naive_bayes' else 'class_probability')})
    return pd.DataFrame(rows)

def calculate_xgboost_sensitivity(estimator, analysis_matrix, approximate_values, analysis_data, feature_groups, sensitivity_indices):
    matrix = analysis_matrix[sensitivity_indices].tocsr()
    data = analysis_data.iloc[sensitivity_indices].reset_index(drop=True)
    predictions = data['pred_xgboost'].to_numpy(dtype=np.int8)
    n_samples, n_features = matrix.shape
    n_classes = len(CLASS_ORDER)
    contributions = estimator.get_booster().predict(xgb.DMatrix(matrix), pred_contribs=True, approx_contribs=False, strict_shape=True)
    contributions = np.asarray(contributions, dtype=np.float64)
    if contributions.shape != (n_samples, n_classes, n_features + 1):
        raise ValueError(f'Unexpected exact XGBoost contribution shape in the sensitivity analysis: {contributions.shape}')
    exact_values, exact_base = standardize_shap_output(contributions[:, :, :-1], contributions[:, :, -1], n_samples, n_features, n_classes)
    target_scores = score_matrix(estimator, matrix, 'xgboost', shap_scale=True)
    exact_reconstructed = exact_base + exact_values.sum(axis=2, dtype=np.float64)
    exact_error = float(np.max(np.abs(exact_reconstructed - target_scores)))
    exact_predictions = exact_reconstructed.argmax(axis=1).astype(np.int8)
    if exact_error > 0.001 or not np.array_equal(exact_predictions, predictions):
        raise ValueError('Exact XGBoost SHAP sensitivity values failed additivity or prediction reconstruction.')
    approximate_subset = np.asarray(approximate_values[sensitivity_indices], dtype=np.float64)
    case_rows = []
    exact_selected = np.abs(exact_values[np.arange(n_samples), predictions, :])
    approximate_selected = np.abs(approximate_subset[np.arange(n_samples), predictions, :])
    for row_index, row in enumerate(data.itertuples(index=False)):
        exact_metrics = metrics_for_vector(exact_selected[row_index], n_features)
        approximate_metrics = metrics_for_vector(approximate_selected[row_index], n_features)
        case_rows.append({'modeling_row_id': int(row.modeling_row_id), GROUP_COLUMN: getattr(row, GROUP_COLUMN), 'actual_class': getattr(row, TARGET_COLUMN), 'predicted_class': CLASS_ORDER[predictions[row_index]], 'exact_top_10_share': exact_metrics['top_10_share'], 'approximate_top_10_share': approximate_metrics['top_10_share'], 'top_10_share_difference_approximate_minus_exact': approximate_metrics['top_10_share'] - exact_metrics['top_10_share'], 'exact_k_80': exact_metrics['k_80'], 'approximate_k_80': approximate_metrics['k_80'], 'k_80_difference_approximate_minus_exact': approximate_metrics['k_80'] - exact_metrics['k_80'], 'exact_normalized_entropy': exact_metrics['normalized_entropy'], 'approximate_normalized_entropy': approximate_metrics['normalized_entropy'], 'entropy_difference_approximate_minus_exact': approximate_metrics['normalized_entropy'] - exact_metrics['normalized_entropy']})
    case_results = pd.DataFrame(case_rows)
    mean_exact = exact_selected.mean(axis=0)
    mean_approximate = approximate_selected.mean(axis=0)
    rank_mask = (mean_exact > SHAP_ZERO_TOLERANCE) | (mean_approximate > SHAP_ZERO_TOLERANCE)
    if rank_mask.sum() >= 2 and np.ptp(mean_exact[rank_mask]) > 0 and (np.ptp(mean_approximate[rank_mask]) > 0):
        rank_correlation = float(spearmanr(mean_exact[rank_mask], mean_approximate[rank_mask]).statistic)
    else:
        rank_correlation = np.nan
    summary_rows = [{'result_type': 'case_metric', 'metric': 'top_10_share_mean', 'feature_group': '', 'exact_value': float(case_results['exact_top_10_share'].mean()), 'approximate_value': float(case_results['approximate_top_10_share'].mean())}, {'result_type': 'case_metric', 'metric': 'k_80_mean', 'feature_group': '', 'exact_value': float(case_results['exact_k_80'].mean()), 'approximate_value': float(case_results['approximate_k_80'].mean())}, {'result_type': 'case_metric', 'metric': 'normalized_entropy_mean', 'feature_group': '', 'exact_value': float(case_results['exact_normalized_entropy'].mean()), 'approximate_value': float(case_results['approximate_normalized_entropy'].mean())}, {'result_type': 'global_rank', 'metric': 'mean_abs_shap_spearman', 'feature_group': '', 'exact_value': 1.0, 'approximate_value': rank_correlation}]
    for group in FEATURE_GROUP_ORDER:
        mask = feature_groups == group
        exact_mass = float(mean_exact[mask].sum())
        approximate_mass = float(mean_approximate[mask].sum())
        exact_total = float(mean_exact.sum())
        approximate_total = float(mean_approximate.sum())
        summary_rows.append({'result_type': 'feature_group', 'metric': 'share_of_total_abs_shap', 'feature_group': group, 'exact_value': exact_mass / exact_total if exact_total else np.nan, 'approximate_value': approximate_mass / approximate_total if approximate_total else np.nan})
    summary = pd.DataFrame(summary_rows)
    summary['absolute_difference'] = np.abs(summary['approximate_value'] - summary['exact_value'])
    validation = {'groups': n_samples, 'groups_per_class': XGB_SENSITIVITY_GROUPS_PER_CLASS, 'exact_method': 'XGBoost native exact Tree SHAP contributions', 'approximate_method': 'TreeExplainer approximate=True', 'exact_max_additivity_error': exact_error, 'exact_prediction_reconstruction_matches': True, 'mean_abs_shap_rank_spearman': rank_correlation, 'mean_absolute_top_10_difference': float(np.abs(case_results['top_10_share_difference_approximate_minus_exact']).mean()), 'mean_absolute_k_80_difference': float(np.abs(case_results['k_80_difference_approximate_minus_exact']).mean())}
    return (case_results, summary, validation)

def paired_stratified_bootstrap(vectors, strata):
    arrays = {name: np.asarray(vector, dtype=float) for name, vector in vectors.items()}
    if not arrays:
        return {}
    length = len(next(iter(arrays.values())))
    if any((len(vector) != length for vector in arrays.values())) or len(strata) != length:
        raise ValueError('Bootstrap vectors and strata must have identical lengths.')
    strata = np.asarray(strata)
    stratum_indices = [np.flatnonzero(strata == label) for label in CLASS_ORDER]
    if any((len(indices) == 0 for indices in stratum_indices)):
        raise ValueError('Every priority class must be represented in the bootstrap sample.')
    rng = np.random.default_rng(RANDOM_STATE)
    distributions = {name: np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float32) for name in arrays}
    for start in range(0, BOOTSTRAP_ITERATIONS, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_ITERATIONS)
        sampled = np.concatenate([rng.choice(indices, size=(stop - start, len(indices)), replace=True) for indices in stratum_indices], axis=1)
        for name, vector in arrays.items():
            distributions[name][start:stop] = vector[sampled].mean(axis=1)
    return distributions

def select_local_cases(prediction_data):
    data = prediction_data.sort_values('modeling_row_id').drop_duplicates(GROUP_COLUMN).copy()
    prediction_columns = [f'pred_{H2_FEATURE_SET}__{model}' for model in XAI_MODEL_ORDER]
    prediction_matrix = data[prediction_columns].to_numpy(dtype=np.int8)
    actual = data['priority_encoded'].to_numpy(dtype=np.int8)
    data['prediction_diversity'] = [len(set(row)) for row in prediction_matrix]
    data['correct_model_count'] = (prediction_matrix == actual[:, None]).sum(axis=1)
    data['unanimous_correct'] = data['correct_model_count'].eq(len(MODEL_ORDER))
    data['medium_votes'] = (prediction_matrix == CLASS_TO_INT['medium']).sum(axis=1)
    data['low_votes'] = (prediction_matrix == CLASS_TO_INT['low']).sum(axis=1)
    data['high_votes'] = (prediction_matrix == CLASS_TO_INT['high']).sum(axis=1)
    selected, used_groups = ([], set())

    def take_case(case_type, mask, sort_columns=None, ascending=None):
        candidates = data.loc[mask & ~data[GROUP_COLUMN].isin(used_groups)].copy()
        if candidates.empty:
            return False
        sort_columns = sort_columns or ['modeling_row_id']
        ascending = ascending or [True]
        candidate = candidates.sort_values(sort_columns, ascending=ascending).iloc[0]
        selected.append({'case_type': case_type, **candidate.to_dict()})
        used_groups.add(candidate[GROUP_COLUMN])
        return True
    for label in CLASS_ORDER:
        take_case(f'unanimous_correct_{label}', data[TARGET_COLUMN].eq(label) & data['unanimous_correct'])
    take_case('typical_low_to_medium', data[TARGET_COLUMN].eq('low') & data['medium_votes'].ge(2), ['correct_model_count', 'modeling_row_id'], [True, True])
    take_case('typical_high_to_medium', data[TARGET_COLUMN].eq('high') & data['medium_votes'].ge(2), ['correct_model_count', 'modeling_row_id'], [True, True])
    take_case('critical_low_to_high', data[TARGET_COLUMN].eq('low') & data['high_votes'].ge(1), ['high_votes', 'prediction_diversity', 'modeling_row_id'], [False, False, True])
    take_case('critical_high_to_low', data[TARGET_COLUMN].eq('high') & data['low_votes'].ge(1), ['low_votes', 'prediction_diversity', 'modeling_row_id'], [False, False, True])
    case_number = 1
    while len(selected) < LOCAL_CASE_TARGET:
        added = take_case(f'model_disagreement_{case_number}', data['prediction_diversity'].gt(1), ['prediction_diversity', 'correct_model_count', 'modeling_row_id'], [False, True, True])
        if not added:
            added = take_case(f'additional_case_{case_number}', pd.Series(True, index=data.index))
        if not added:
            break
        case_number += 1
    output = pd.DataFrame(selected).head(LOCAL_CASE_TARGET).reset_index(drop=True)
    output.insert(0, 'case_id', [f'case_{index:02d}' for index in range(1, len(output) + 1)])
    return output[['case_id', 'case_type', 'modeling_row_id', GROUP_COLUMN, TARGET_COLUMN, 'priority_encoded', 'prediction_diversity', 'correct_model_count', *prediction_columns]]

# %% 03 - Validate frozen Modeling and Evaluation provenance

required_paths = [DATA_PATH, MODEL_SELECTION_PATH, FEATURE_SET_SELECTION_PATH, MODELING_MANIFEST_PATH, FITTED_MODEL_METADATA_PATH, SPLIT_PATH, EVALUATION_MANIFEST_PATH, HOLDOUT_PREDICTIONS_PATH, SENTIMENT_PATH, SENTIMENT_META_PATH]
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError('Required frozen artifacts are missing:\n' + '\n'.join(missing_paths))
modeling_manifest = load_json(MODELING_MANIFEST_PATH, default={})
evaluation_manifest = load_json(EVALUATION_MANIFEST_PATH, default={})
data_sha256 = sha256_file(DATA_PATH)
expected_modeling_values = {'script_build': EXPECTED_MODELING_BUILD, 'run_id': EXPECTED_MODELING_RUN_ID, 'data_sha256': data_sha256, 'group_column': GROUP_COLUMN, 'exact_group_column': EXACT_GROUP_COLUMN, 'prepared_records': EXPECTED_RECORDS, 'semantic_groups': EXPECTED_SEMANTIC_GROUPS, 'training_records': EXPECTED_TRAINING_RECORDS, 'training_semantic_groups': EXPECTED_TRAINING_GROUPS, 'holdout_records': EXPECTED_HOLDOUT_RECORDS, 'holdout_semantic_groups': EXPECTED_HOLDOUT_GROUPS}
for field, expected in expected_modeling_values.items():
    if modeling_manifest.get(field) != expected:
        raise ValueError(f'Unexpected Modeling manifest value for {field}: {modeling_manifest.get(field)!r} != {expected!r}')
if modeling_manifest.get('strict_data_freeze') is not True:
    raise ValueError('Modeling manifest does not confirm strict data freeze.')
if modeling_manifest.get('models') != ALL_MODEL_ORDER:
    raise ValueError('Unexpected frozen classifier set in Modeling manifest.')
common_selection_manifest = modeling_manifest.get('common_feature_set_selection', {})
H2_FEATURE_SET = str(common_selection_manifest.get('selected_feature_set', '')).strip()
if H2_FEATURE_SET not in VALID_FEATURE_SETS:
    raise ValueError('Modeling manifest does not contain a valid CV-selected common feature set.')

h2_manifest = modeling_manifest.get('hypothesis_operationalization', {}).get('H2', {})
if (
    h2_manifest.get('feature_set') != H2_FEATURE_SET
    or h2_manifest.get('baseline') != H2_BASELINE_MODEL
    or h2_manifest.get('family_mean_comparison') is not False
):
    raise ValueError('Modeling manifest contains an incompatible H2 design.')

feature_set_selection = pd.read_csv(FEATURE_SET_SELECTION_PATH)
required_feature_selection_columns = {
    'feature_set',
    'selected_for_common_comparison',
}
if not required_feature_selection_columns.issubset(feature_set_selection.columns):
    raise ValueError('Feature-set selection table has an incompatible schema.')
selected_mask = (
    feature_set_selection['selected_for_common_comparison']
    .astype(str)
    .str.strip()
    .str.lower()
    .isin({'true', '1'})
)
selected_feature_rows = feature_set_selection.loc[selected_mask]
if len(selected_feature_rows) != 1:
    raise ValueError('Feature-set selection table must contain exactly one selected row.')
if str(selected_feature_rows.iloc[0]['feature_set']) != H2_FEATURE_SET:
    raise ValueError('Feature-set selection artifact and Modeling manifest disagree.')

manifest_final = modeling_manifest.get('final_model', {})
FINAL_OVERALL_MODEL_KEY = (
    str(manifest_final.get('feature_set', '')).strip(),
    str(manifest_final.get('model', '')).strip(),
)
if FINAL_OVERALL_MODEL_KEY[0] not in VALID_FEATURE_SETS or FINAL_OVERALL_MODEL_KEY[1] not in ALL_MODEL_ORDER:
    raise ValueError('Modeling manifest does not contain a valid final overall model.')

feature_sets_manifest = modeling_manifest.get('feature_sets', {})
def feature_groups_for(feature_set):
    spec = feature_sets_manifest.get(feature_set, {})
    groups = ['text', *list(spec.get('categorical', []))]
    if bool(spec.get('sentiment', False)):
        groups.append('sentiment')
    return groups

FEATURE_GROUP_ORDER = feature_groups_for(H2_FEATURE_SET)
FINAL_FEATURE_GROUP_ORDER = feature_groups_for(FINAL_OVERALL_MODEL_KEY[0])
if not FEATURE_GROUP_ORDER or not FINAL_FEATURE_GROUP_ORDER:
    raise ValueError('Unable to derive feature-group order from Modeling manifest.')

FINAL_MODEL_TAG = f'{FINAL_OVERALL_MODEL_KEY[0]}_{FINAL_OVERALL_MODEL_KEY[1]}'
FINAL_MODEL_GLOBAL_FEATURES_PATH = TABLE_DIR / f'05_final_{FINAL_MODEL_TAG}_global_top_features.csv'
FINAL_MODEL_FEATURE_GROUPS_PATH = TABLE_DIR / f'05_final_{FINAL_MODEL_TAG}_feature_group_summary.csv'
FINAL_MODEL_LOCAL_CONTRIBUTIONS_PATH = TABLE_DIR / f'05_final_{FINAL_MODEL_TAG}_local_feature_contributions.csv'
FINAL_MODEL_VALIDATION_PATH = TABLE_DIR / f'05_final_{FINAL_MODEL_TAG}_validation.csv'
FINAL_MODEL_CACHE_PATH = MODEL_CACHE_DIR / f'05_{FINAL_MODEL_TAG}_final_model_results.joblib'

expected_evaluation_values = {
    'script_build': EXPECTED_EVALUATION_BUILD,
    'run_id': EXPECTED_EVALUATION_RUN_ID,
    'modeling_build': EXPECTED_MODELING_BUILD,
    'modeling_run_id': EXPECTED_MODELING_RUN_ID,
    'data_sha256': data_sha256,
    'records': EXPECTED_RECORDS,
    'training_records': EXPECTED_TRAINING_RECORDS,
    'holdout_records': EXPECTED_HOLDOUT_RECORDS,
    'holdout_semantic_groups': EXPECTED_HOLDOUT_GROUPS,
    'group_column': GROUP_COLUMN,
}
for field, expected in expected_evaluation_values.items():
    if evaluation_manifest.get(field) != expected:
        raise ValueError(f'Unexpected Evaluation manifest value for {field}: {evaluation_manifest.get(field)!r} != {expected!r}')
if evaluation_manifest.get('final_model') != {'feature_set': FINAL_OVERALL_MODEL_KEY[0], 'model': FINAL_OVERALL_MODEL_KEY[1]}:
    raise ValueError('Unexpected final model in Evaluation manifest.')
if evaluation_manifest.get('h1', {}).get('feature_set') != H2_FEATURE_SET:
    raise ValueError('Evaluation and Modeling disagree on the common comparison feature set.')
if evaluation_manifest.get('common_feature_set_selection', {}).get('feature_set') != H2_FEATURE_SET:
    raise ValueError('Evaluation manifest does not confirm the CV-selected common feature set.')
if evaluation_manifest.get('output_hashes', {}).get('holdout_predictions') != sha256_file(HOLDOUT_PREDICTIONS_PATH):
    raise ValueError('Hold-out predictions differ from the frozen Evaluation manifest.')
artifact_hashes = modeling_manifest.get('artifact_hashes', {})
for name, path in {
    'split': SPLIT_PATH,
    'feature_set_selection': FEATURE_SET_SELECTION_PATH,
    'model_selection': MODEL_SELECTION_PATH,
    'fitted_model_metadata': FITTED_MODEL_METADATA_PATH,
    'sentiment_features': SENTIMENT_PATH,
    'sentiment_metadata': SENTIMENT_META_PATH,
}.items():
    if artifact_hashes.get(name) != sha256_file(path):
        raise ValueError(f'Frozen Modeling artifact hash mismatch: {name}')
model_selection = pd.read_csv(MODEL_SELECTION_PATH)
common_models_all = model_selection.loc[model_selection['feature_set'].eq(H2_FEATURE_SET), ['feature_set', 'model', 'model_family', 'model_artifact']].drop_duplicates()
if set(common_models_all['model']) != set(ALL_MODEL_ORDER) or len(common_models_all) != len(ALL_MODEL_ORDER):
    raise ValueError('The frozen common-feature classifier set is incomplete.')
xai_models = common_models_all.loc[common_models_all['model'].isin(XAI_MODEL_ORDER)].copy()
if set(xai_models['model']) != set(XAI_MODEL_ORDER):
    raise ValueError('The SHAP-eligible common-feature classifier set is incomplete.')

final_model_rows = model_selection.loc[
    model_selection['feature_set'].eq(FINAL_OVERALL_MODEL_KEY[0])
    & model_selection['model'].eq(FINAL_OVERALL_MODEL_KEY[1]),
    ['feature_set', 'model', 'model_family', 'model_artifact'],
].drop_duplicates()

if len(final_model_rows) != 1:
    raise ValueError(
        'The frozen final overall model artifact is missing or ambiguous.'
    )

eligibility_rows = []
for model in ALL_MODEL_ORDER:
    if model in LINEAR_MODELS:
        method = 'LinearExplainer'
    elif model == 'complement_naive_bayes':
        method = 'LinearExplainer (ComplementNB score)'
    elif model == 'knn_cosine':
        method = 'PermutationExplainer'
    elif model == 'lightgbm':
        method = 'LightGBM native TreeSHAP'
    else:
        method = 'TreeExplainer approximate=True'
    eligibility_rows.append({
        'model': model,
        'included_in_h2': True,
        'shap_method': method,
    })
xai_eligibility = pd.DataFrame(eligibility_rows)
xai_eligibility.to_csv(XAI_ELIGIBILITY_PATH, index=False)
fitted_metadata_all = load_json(FITTED_MODEL_METADATA_PATH, default={})
if fitted_metadata_all.get('status') != 'complete':
    raise ValueError('Fitted-model metadata are not marked complete.')
fitted_metadata = fitted_metadata_all.get('models', {})
manifest_model_hashes = modeling_manifest.get('selected_model_hashes', {})
model_paths, model_hashes = ({}, {})
for row in xai_models.itertuples(index=False):
    path = MODELING_MODEL_DIR / row.model_artifact
    if not path.exists():
        raise FileNotFoundError(f'Frozen common-feature model not found: {path}')
    file_hash = sha256_file(path)
    metadata_entry = fitted_metadata.get(row.model_artifact, {})
    stored_hash = metadata_entry.get('sha256', metadata_entry.get('file_sha256'))
    if stored_hash != file_hash:
        raise ValueError(f'Fitted-model metadata hash mismatch: {row.model_artifact}')
    if manifest_model_hashes.get(row.model_artifact) != file_hash:
        raise ValueError(f'Modeling-manifest hash mismatch: {row.model_artifact}')
    model_paths[row.model] = path
    model_hashes[row.model] = file_hash

final_model_row = final_model_rows.iloc[0]
final_model_path = MODELING_MODEL_DIR / final_model_row['model_artifact']
if not final_model_path.exists():
    raise FileNotFoundError(
        f'Frozen final overall model not found: {final_model_path}'
    )

final_model_hash = sha256_file(final_model_path)
final_metadata_entry = fitted_metadata.get(
    final_model_row['model_artifact'],
    {},
)
final_stored_hash = final_metadata_entry.get(
    'sha256',
    final_metadata_entry.get('file_sha256'),
)

if final_stored_hash != final_model_hash:
    raise ValueError(
        'Fitted-model metadata hash mismatch for final overall model.'
    )
if manifest_model_hashes.get(
    final_model_row['model_artifact']
) != final_model_hash:
    raise ValueError(
        'Modeling-manifest hash mismatch for final overall model.'
    )

print_section('Explainability input')
print(f'Explainability build      : {SCRIPT_BUILD}')
print(f'Modeling build            : {EXPECTED_MODELING_BUILD}')
print(f'Evaluation build          : {EXPECTED_EVALUATION_BUILD}')
print(f'Common comparison set     : {H2_FEATURE_SET}')
print(f'Frozen common classifiers : {len(ALL_MODEL_ORDER)}')
print(f'SHAP-analyzed H2 models   : {len(XAI_MODEL_ORDER)}')
print(f'Final model explained     : {FINAL_OVERALL_MODEL_KEY[0]} + {MODEL_LABELS[FINAL_OVERALL_MODEL_KEY[1]]}')
print(f'H2 pairwise contrasts     : {len(H2_ALTERNATIVES)}')
print(f'H2 baseline               : {MODEL_LABELS[H2_BASELINE_MODEL]}')
print(f'Primary H2 metric         : Top-{PRIMARY_TOP_K} SHAP concentration')
print(f'Bootstrap iterations      : {fmt_int(BOOTSTRAP_ITERATIONS)}')

# %% 04 - Reconstruct fixed training and hold-out partitions

data = pd.read_csv(DATA_PATH, low_memory=False)
data['modeling_row_id'] = np.arange(len(data), dtype=np.int64)
required_columns = {'source_row_id', TARGET_COLUMN, GROUP_COLUMN, EXACT_GROUP_COLUMN, TEXT_COLUMN, TFIDF_TEXT_COLUMN, 'type', 'language', 'queue'}
missing_columns = sorted(required_columns - set(data.columns))
if missing_columns:
    raise ValueError(f'Missing prepared columns: {missing_columns}')
for column in [GROUP_COLUMN, EXACT_GROUP_COLUMN, TEXT_COLUMN, TFIDF_TEXT_COLUMN, 'type', 'language', 'queue']:
    data[column] = data[column].fillna('').astype(str).str.strip()
data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str).str.lower().str.strip()
data['source_row_id'] = pd.to_numeric(data['source_row_id'], errors='raise').astype(np.int64)
if len(data) != EXPECTED_RECORDS:
    raise ValueError('Unexpected prepared record count.')
if data[GROUP_COLUMN].nunique() != EXPECTED_SEMANTIC_GROUPS:
    raise ValueError('Unexpected semantic-group count.')
if set(data[TARGET_COLUMN]) != set(CLASS_ORDER):
    raise ValueError('Unexpected priority classes.')
if data['source_row_id'].duplicated().any():
    raise ValueError('source_row_id must be unique.')
data['priority_encoded'] = data[TARGET_COLUMN].map(CLASS_TO_INT).astype(np.int8)

sentiment_meta = load_json(SENTIMENT_META_PATH, default={})
sentiment_manifest = modeling_manifest.get('sentiment', {})
sentiment_features = pd.read_csv(SENTIMENT_PATH)

if len(sentiment_features) != len(data):
    raise ValueError('Unexpected sentiment-feature record count.')
if 'modeling_row_id' not in sentiment_features.columns:
    raise ValueError('Sentiment cache is missing modeling_row_id.')
if not np.array_equal(
    sentiment_features['modeling_row_id'].to_numpy(dtype=np.int64),
    data['modeling_row_id'].to_numpy(dtype=np.int64),
):
    raise ValueError('Sentiment cache does not match prepared-data row order.')
if not set(SENTIMENT_COLUMNS).issubset(sentiment_features.columns):
    raise ValueError('Sentiment cache is missing probability columns.')

sentiment_values = sentiment_features[SENTIMENT_COLUMNS].apply(
    pd.to_numeric,
    errors='raise',
)
sentiment_array = sentiment_values.to_numpy(dtype=np.float64)

if (
    not np.isfinite(sentiment_array).all()
    or (sentiment_array < 0).any()
    or (sentiment_array > 1).any()
    or not np.allclose(
        sentiment_array.sum(axis=1),
        1.0,
        atol=1e-5,
    )
):
    raise ValueError('Invalid frozen sentiment probabilities.')

if sentiment_meta.get('records') != len(data):
    raise ValueError('Unexpected sentiment metadata record count.')
if sentiment_meta.get('model_id') != sentiment_manifest.get('model_id'):
    raise ValueError('Sentiment model ID mismatch.')
if sentiment_meta.get(
    'revision',
    sentiment_meta.get('model_revision'),
) != sentiment_manifest.get('revision'):
    raise ValueError('Sentiment model revision mismatch.')

for column in SENTIMENT_COLUMNS:
    data[column] = sentiment_values[column].to_numpy(dtype=np.float32)

split_assignments = pd.read_parquet(SPLIT_PATH).sort_values('modeling_row_id').reset_index(drop=True)
data = data.sort_values('modeling_row_id').reset_index(drop=True)
if len(split_assignments) != len(data):
    raise ValueError('Split assignments do not cover all prepared records.')
if not np.array_equal(split_assignments['modeling_row_id'].to_numpy(dtype=np.int64), data['modeling_row_id'].to_numpy(dtype=np.int64)):
    raise ValueError('Split assignments do not match prepared row order.')
for column in ['source_row_id', GROUP_COLUMN, EXACT_GROUP_COLUMN, TARGET_COLUMN]:
    if not np.array_equal(split_assignments[column].astype(str).to_numpy(), data[column].astype(str).to_numpy()):
        raise ValueError(f'Frozen split disagrees with prepared data on {column}.')
if set(split_assignments['split']) != {'training', 'holdout'}:
    raise ValueError('Unexpected split labels.')
data['split'] = split_assignments['split'].to_numpy()
training_data = data.loc[data['split'].eq('training')].reset_index(drop=True)
holdout_data = data.loc[data['split'].eq('holdout')].reset_index(drop=True)
if len(training_data) != EXPECTED_TRAINING_RECORDS:
    raise ValueError('Unexpected training record count.')
if training_data[GROUP_COLUMN].nunique() != EXPECTED_TRAINING_GROUPS:
    raise ValueError('Unexpected training semantic-group count.')
if len(holdout_data) != EXPECTED_HOLDOUT_RECORDS:
    raise ValueError('Unexpected hold-out record count.')
if holdout_data[GROUP_COLUMN].nunique() != EXPECTED_HOLDOUT_GROUPS:
    raise ValueError('Unexpected hold-out semantic-group count.')
if set(training_data[GROUP_COLUMN]).intersection(holdout_data[GROUP_COLUMN]):
    raise ValueError('Semantic-group overlap between training and hold-out.')
if set(training_data[EXACT_GROUP_COLUMN]).intersection(holdout_data[EXACT_GROUP_COLUMN]):
    raise ValueError('Exact-text overlap between training and hold-out.')
holdout_predictions = pd.read_parquet(HOLDOUT_PREDICTIONS_PATH)
if len(holdout_predictions) != len(holdout_data):
    raise ValueError('Evaluation prediction table has an unexpected row count.')
if not np.array_equal(holdout_predictions['modeling_row_id'].to_numpy(dtype=np.int64), holdout_data['modeling_row_id'].to_numpy(dtype=np.int64)):
    raise ValueError('Evaluation predictions do not match fixed hold-out row order.')
prediction_columns = {model: f'pred_{H2_FEATURE_SET}__{model}' for model in XAI_MODEL_ORDER}
missing_prediction_columns = sorted(set(prediction_columns.values()) - set(holdout_predictions.columns))
if missing_prediction_columns:
    raise ValueError(f'Missing common-feature hold-out prediction columns: {missing_prediction_columns}')
for model, column in prediction_columns.items():
    holdout_data[f'pred_{model}'] = holdout_predictions[column].to_numpy(dtype=np.int8)

FINAL_PREDICTION_COLUMN = f'pred_{FINAL_OVERALL_MODEL_KEY[0]}__{FINAL_OVERALL_MODEL_KEY[1]}'
if FINAL_PREDICTION_COLUMN not in holdout_predictions.columns:
    raise ValueError('Missing final overall-model hold-out predictions.')

holdout_data['pred_final_model'] = (
    holdout_predictions[FINAL_PREDICTION_COLUMN]
    .to_numpy(dtype=np.int8)
)

print_section('Fixed explainability population')
print(f'Training records          : {fmt_int(len(training_data))}')
print(f'Training semantic groups  : {fmt_int(EXPECTED_TRAINING_GROUPS)}')
print(f'Hold-out records          : {fmt_int(len(holdout_data))}')
print(f'Hold-out semantic groups  : {fmt_int(EXPECTED_HOLDOUT_GROUPS)}')
print('Semantic-group overlap    : 0')
print('Exact-text overlap        : 0')

# %% 05 - Create shared balanced XAI sample and local cases

holdout_group_representatives = holdout_data.sort_values('modeling_row_id').drop_duplicates(GROUP_COLUMN).reset_index(drop=True)
training_group_representatives = training_data.sort_values('modeling_row_id').drop_duplicates(GROUP_COLUMN).reset_index(drop=True)
rng = np.random.default_rng(RANDOM_STATE)
global_parts, background_parts = ([], [])
for label in CLASS_ORDER:
    holdout_class = holdout_group_representatives.loc[holdout_group_representatives[TARGET_COLUMN].eq(label)]
    training_class = training_group_representatives.loc[training_group_representatives[TARGET_COLUMN].eq(label)]
    if len(holdout_class) < GLOBAL_GROUPS_PER_CLASS:
        raise ValueError(f'Insufficient hold-out groups for class {label}.')
    if len(training_class) < BACKGROUND_GROUPS_PER_CLASS:
        raise ValueError(f'Insufficient training groups for SHAP background class {label}.')
    global_parts.append(holdout_class.iloc[rng.choice(len(holdout_class), size=GLOBAL_GROUPS_PER_CLASS, replace=False)])
    background_parts.append(training_class.iloc[rng.choice(len(training_class), size=BACKGROUND_GROUPS_PER_CLASS, replace=False)])
global_sample = pd.concat(global_parts, ignore_index=True).sort_values([TARGET_COLUMN, 'modeling_row_id']).reset_index(drop=True)
global_sample['analysis_role'] = 'global_h2'
global_sample['is_global_h2_case'] = True
global_sample['is_local_case'] = False
global_sample['case_id'] = ''
global_sample['case_type'] = ''
faithfulness_rng = np.random.default_rng(RANDOM_STATE + 1)
faithfulness_parts = []
for label in CLASS_ORDER:
    class_rows = global_sample.loc[global_sample[TARGET_COLUMN].eq(label)]
    faithfulness_parts.append(class_rows.iloc[faithfulness_rng.choice(len(class_rows), size=FAITHFULNESS_GROUPS_PER_CLASS, replace=False)])
faithfulness_sample = pd.concat(faithfulness_parts, ignore_index=True).sort_values([TARGET_COLUMN, 'modeling_row_id']).reset_index(drop=True)
xgb_rng = np.random.default_rng(RANDOM_STATE + 2000)
xgb_parts = []
for label in CLASS_ORDER:
    class_rows = global_sample.loc[global_sample[TARGET_COLUMN].eq(label)]
    xgb_parts.append(class_rows.iloc[xgb_rng.choice(len(class_rows), size=XGB_SENSITIVITY_GROUPS_PER_CLASS, replace=False)])
xgb_sensitivity_sample = pd.concat(xgb_parts, ignore_index=True).sort_values([TARGET_COLUMN, 'modeling_row_id']).reset_index(drop=True)
faithfulness_group_ids = set(faithfulness_sample[GROUP_COLUMN])
xgb_sensitivity_group_ids = set(xgb_sensitivity_sample[GROUP_COLUMN])
global_sample['is_faithfulness_case'] = global_sample[GROUP_COLUMN].isin(faithfulness_group_ids)
global_sample['is_xgboost_sensitivity_case'] = global_sample[GROUP_COLUMN].isin(xgb_sensitivity_group_ids)
background_sample = pd.concat(background_parts, ignore_index=True).sort_values([TARGET_COLUMN, 'modeling_row_id']).reset_index(drop=True)
background_sample[['modeling_row_id', GROUP_COLUMN, TARGET_COLUMN]].to_parquet(BACKGROUND_SAMPLE_PATH, index=False)
legacy_knn_background_rng = np.random.default_rng(RANDOM_STATE + 3_000)
legacy_knn_background_parts = []
for label in CLASS_ORDER:
    class_background = background_sample.loc[
        background_sample[TARGET_COLUMN].eq(label)
    ].sort_values('modeling_row_id').reset_index(drop=True)
    selected_positions = legacy_knn_background_rng.choice(
        len(class_background),
        size=LEGACY_KNN_BACKGROUND_GROUPS_PER_CLASS,
        replace=False,
    )
    legacy_knn_background_parts.append(
        class_background.iloc[selected_positions]
    )
legacy_knn_background_sample = (
    pd.concat(legacy_knn_background_parts, ignore_index=True)
    .sort_values([TARGET_COLUMN, 'modeling_row_id'])
    .reset_index(drop=True)
)

knn_background_rng = np.random.default_rng(RANDOM_STATE + 3_000)
knn_background_parts = []
for label in CLASS_ORDER:
    class_background = background_sample.loc[
        background_sample[TARGET_COLUMN].eq(label)
    ].sort_values('modeling_row_id').reset_index(drop=True)
    if len(class_background) < KNN_BACKGROUND_GROUPS_PER_CLASS:
        raise ValueError(f'Insufficient kNN SHAP background groups for class {label}.')
    selected_positions = knn_background_rng.choice(
        len(class_background),
        size=KNN_BACKGROUND_GROUPS_PER_CLASS,
        replace=False,
    )
    knn_background_parts.append(
        class_background.iloc[selected_positions]
    )

knn_background_sample = (
    pd.concat(knn_background_parts, ignore_index=True)
    .sort_values([TARGET_COLUMN, 'modeling_row_id'])
    .reset_index(drop=True)
)
knn_background_sample[
    ['modeling_row_id', GROUP_COLUMN, TARGET_COLUMN]
].to_parquet(KNN_BACKGROUND_SAMPLE_PATH, index=False)
local_case_source = holdout_predictions.merge(holdout_data[['modeling_row_id', GROUP_COLUMN, TARGET_COLUMN, 'priority_encoded']], on=['modeling_row_id', GROUP_COLUMN, TARGET_COLUMN, 'priority_encoded'], how='inner', validate='one_to_one')
local_cases = select_local_cases(local_case_source)
local_case_details = local_cases.merge(holdout_data.drop(columns=[f'pred_{model}' for model in XAI_MODEL_ORDER]), on=['modeling_row_id', GROUP_COLUMN, TARGET_COLUMN, 'priority_encoded'], how='left', validate='one_to_one')
for model in XAI_MODEL_ORDER:
    local_case_details[f'pred_{model}'] = local_case_details[prediction_columns[model]].astype(np.int8)
local_case_details['analysis_role'] = 'local_case'
local_case_details['is_global_h2_case'] = False
local_case_details['is_local_case'] = True
local_case_details['is_faithfulness_case'] = False
local_case_details['is_xgboost_sensitivity_case'] = False
analysis_data = pd.concat([global_sample, local_case_details.loc[~local_case_details[GROUP_COLUMN].isin(global_sample[GROUP_COLUMN])]], ignore_index=True, sort=False)
local_mapping = local_case_details.set_index(GROUP_COLUMN)[['case_id', 'case_type']]
local_mask = analysis_data[GROUP_COLUMN].isin(local_mapping.index)
analysis_data.loc[local_mask, 'is_local_case'] = True
analysis_data.loc[local_mask, 'case_id'] = analysis_data.loc[local_mask, GROUP_COLUMN].map(local_mapping['case_id'])
analysis_data.loc[local_mask, 'case_type'] = analysis_data.loc[local_mask, GROUP_COLUMN].map(local_mapping['case_type'])
analysis_data.loc[analysis_data['is_global_h2_case'] & analysis_data['is_local_case'], 'analysis_role'] = 'global_h2_and_local'
analysis_data['is_faithfulness_case'] = analysis_data[GROUP_COLUMN].isin(faithfulness_group_ids)
analysis_data['is_xgboost_sensitivity_case'] = analysis_data[GROUP_COLUMN].isin(xgb_sensitivity_group_ids)
for model in XAI_MODEL_ORDER:
    if f'pred_{model}' not in analysis_data.columns:
        analysis_data = analysis_data.merge(
            holdout_data[['modeling_row_id', f'pred_{model}']],
            on='modeling_row_id',
            how='left',
            validate='one_to_one',
        )
    analysis_data[f'pred_{model}'] = (
        analysis_data[f'pred_{model}'].astype(np.int8)
    )

if 'pred_final_model' not in analysis_data.columns:
    analysis_data = analysis_data.merge(
        holdout_data[
            ['modeling_row_id', 'pred_final_model']
        ],
        on='modeling_row_id',
        how='left',
        validate='one_to_one',
    )
analysis_data['pred_final_model'] = (
    analysis_data['pred_final_model'].astype(np.int8)
)

if analysis_data[GROUP_COLUMN].duplicated().any():
    raise ValueError('XAI sample must contain one record per semantic group.')
expected_balanced = {label: GLOBAL_GROUPS_PER_CLASS for label in CLASS_ORDER}
if global_sample.groupby(TARGET_COLUMN).size().reindex(CLASS_ORDER).to_dict() != expected_balanced:
    raise ValueError('Global XAI sample is not class-balanced.')
if faithfulness_sample.groupby(TARGET_COLUMN).size().reindex(CLASS_ORDER).to_dict() != {label: FAITHFULNESS_GROUPS_PER_CLASS for label in CLASS_ORDER}:
    raise ValueError('Faithfulness sample is not class-balanced.')
if xgb_sensitivity_sample.groupby(TARGET_COLUMN).size().reindex(CLASS_ORDER).to_dict() != {label: XGB_SENSITIVITY_GROUPS_PER_CLASS for label in CLASS_ORDER}:
    raise ValueError('XGBoost sensitivity sample is not class-balanced.')
sample_output_columns = [
    'modeling_row_id',
    GROUP_COLUMN,
    TARGET_COLUMN,
    'priority_encoded',
    'analysis_role',
    'is_global_h2_case',
    'is_faithfulness_case',
    'is_xgboost_sensitivity_case',
    'is_local_case',
    'case_id',
    'case_type',
    'type',
    'language',
    'queue',
    *SENTIMENT_COLUMNS,
    TEXT_COLUMN,
    TFIDF_TEXT_COLUMN,
    *[f'pred_{model}' for model in XAI_MODEL_ORDER],
    'pred_final_model',
]
analysis_data[sample_output_columns].to_csv(XAI_SAMPLE_PATH, index=False)
local_case_details.to_csv(LOCAL_CASES_PATH, index=False)
base_indices = np.flatnonzero(analysis_data['is_global_h2_case'].to_numpy())
faithfulness_indices = np.flatnonzero(analysis_data['is_faithfulness_case'].to_numpy())
xgb_sensitivity_indices = np.flatnonzero(analysis_data[GROUP_COLUMN].isin(xgb_sensitivity_group_ids).to_numpy())
sample_signature = stable_hash({
    'global_modeling_row_ids': global_sample['modeling_row_id'].tolist(),
    'local_modeling_row_ids': local_case_details['modeling_row_id'].tolist(),
    'faithfulness_modeling_row_ids': faithfulness_sample['modeling_row_id'].tolist(),
    'xgboost_sensitivity_modeling_row_ids': xgb_sensitivity_sample['modeling_row_id'].tolist(),
    'background_modeling_row_ids': background_sample['modeling_row_id'].tolist(),
    'knn_background_modeling_row_ids': knn_background_sample['modeling_row_id'].tolist(),
})
legacy_sample_signature = stable_hash({
    'global_modeling_row_ids': global_sample['modeling_row_id'].tolist(),
    'local_modeling_row_ids': local_case_details['modeling_row_id'].tolist(),
    'faithfulness_modeling_row_ids': faithfulness_sample['modeling_row_id'].tolist(),
    'xgboost_sensitivity_modeling_row_ids': xgb_sensitivity_sample['modeling_row_id'].tolist(),
    'background_modeling_row_ids': background_sample['modeling_row_id'].tolist(),
    'knn_background_modeling_row_ids': legacy_knn_background_sample['modeling_row_id'].tolist(),
})
print_section('Shared SHAP design')
print(f'Global groups per class   : {GLOBAL_GROUPS_PER_CLASS}')
print(f'Global H2 groups          : {fmt_int(len(global_sample))}')
print(f'Faithfulness groups/class : {FAITHFULNESS_GROUPS_PER_CLASS}')
print(f'Faithfulness groups       : {fmt_int(len(faithfulness_sample))}')
print(f'XGB sensitivity groups    : {fmt_int(len(xgb_sensitivity_sample))}')
print(f'Local comparison cases    : {len(local_case_details)}')
print(f'Unique explained groups   : {fmt_int(len(analysis_data))}')
print(f'Linear/CNB background     : {fmt_int(len(background_sample))}')
print(f'kNN permutation background: {fmt_int(len(knn_background_sample))}')
print('Shared cases              : identical across all H2 models')
print('H2 comparison             : LR baseline vs all six alternatives')

# %% 06 - Calculate or reuse SHAP analyses

all_feature_dictionaries = []
all_compactness = []
all_global_features = []
all_feature_groups = []
all_local_contributions = []
all_faithfulness = []
validation_rows = []
cache_paths = {}
xgboost_sensitivity_by_case = None
xgboost_sensitivity_summary = None
xgboost_sensitivity_validation = None
for model_name in XAI_MODEL_ORDER:
    print_section(f'SHAP analysis: {MODEL_LABELS[model_name]}')
    cache_path = MODEL_CACHE_DIR / f'05_{H2_FEATURE_SET}_{model_name}_results.joblib'
    cache_paths[model_name] = cache_path
    model_signature = stable_hash({
        'script_build': SCRIPT_BUILD,
        'data_sha256': data_sha256,
        'model_sha256': model_hashes[model_name],
        'sample_signature': sample_signature,
        'feature_set': H2_FEATURE_SET,
        'primary_top_k': PRIMARY_TOP_K,
        'top_k_values': TOP_K_VALUES,
        'mass_thresholds': MASS_THRESHOLDS,
        'faithfulness_groups_per_class': FAITHFULNESS_GROUPS_PER_CLASS,
        'faithfulness_top_k_values': FAITHFULNESS_TOP_K_VALUES,
        'faithfulness_random_repeats': FAITHFULNESS_RANDOM_REPEATS,
        'xgboost_sensitivity_groups_per_class': XGB_SENSITIVITY_GROUPS_PER_CLASS,
        'knn_permutation_cycles': KNN_PERMUTATION_CYCLES,
        'knn_background_modeling_row_ids': knn_background_sample['modeling_row_id'].tolist(),
        'package_versions': PACKAGE_VERSIONS,
    })

    previous_model_signature = stable_hash({
        'script_build': PREVIOUS_XAI_BUILD,
        'data_sha256': data_sha256,
        'model_sha256': model_hashes[model_name],
        'sample_signature': sample_signature,
        'feature_set': H2_FEATURE_SET,
        'primary_top_k': PRIMARY_TOP_K,
        'top_k_values': TOP_K_VALUES,
        'mass_thresholds': MASS_THRESHOLDS,
        'faithfulness_groups_per_class': FAITHFULNESS_GROUPS_PER_CLASS,
        'faithfulness_top_k_values': FAITHFULNESS_TOP_K_VALUES,
        'faithfulness_random_repeats': FAITHFULNESS_RANDOM_REPEATS,
        'xgboost_sensitivity_groups_per_class': XGB_SENSITIVITY_GROUPS_PER_CLASS,
        'knn_permutation_cycles': KNN_PERMUTATION_CYCLES,
        'knn_background_modeling_row_ids': knn_background_sample['modeling_row_id'].tolist(),
        'package_versions': PACKAGE_VERSIONS,
    })

    previous_model_signature_v3 = stable_hash({
        'script_build': PREVIOUS_XAI_BUILD_V3,
        'data_sha256': data_sha256,
        'model_sha256': model_hashes[model_name],
        'sample_signature': sample_signature,
        'feature_set': H2_FEATURE_SET,
        'primary_top_k': PRIMARY_TOP_K,
        'top_k_values': TOP_K_VALUES,
        'mass_thresholds': MASS_THRESHOLDS,
        'faithfulness_groups_per_class': FAITHFULNESS_GROUPS_PER_CLASS,
        'faithfulness_top_k_values': FAITHFULNESS_TOP_K_VALUES,
        'faithfulness_random_repeats': FAITHFULNESS_RANDOM_REPEATS,
        'xgboost_sensitivity_groups_per_class': XGB_SENSITIVITY_GROUPS_PER_CLASS,
        'knn_permutation_cycles': KNN_PERMUTATION_CYCLES,
        'knn_background_modeling_row_ids': knn_background_sample['modeling_row_id'].tolist(),
        'package_versions': PACKAGE_VERSIONS,
    })

    legacy_model_signature = stable_hash({
        'script_build': LEGACY_XAI_BUILD,
        'data_sha256': data_sha256,
        'model_sha256': model_hashes[model_name],
        'sample_signature': legacy_sample_signature,
        'feature_set': H2_FEATURE_SET,
        'primary_top_k': PRIMARY_TOP_K,
        'top_k_values': TOP_K_VALUES,
        'mass_thresholds': MASS_THRESHOLDS,
        'faithfulness_groups_per_class': FAITHFULNESS_GROUPS_PER_CLASS,
        'faithfulness_top_k_values': FAITHFULNESS_TOP_K_VALUES,
        'faithfulness_random_repeats': FAITHFULNESS_RANDOM_REPEATS,
        'xgboost_sensitivity_groups_per_class': XGB_SENSITIVITY_GROUPS_PER_CLASS,
        'knn_permutation_cycles': LEGACY_KNN_PERMUTATION_CYCLES,
        'knn_background_modeling_row_ids': legacy_knn_background_sample['modeling_row_id'].tolist(),
        'package_versions': PACKAGE_VERSIONS,
    })

    cached = None
    if REUSE_VALID_RESULTS and cache_path.exists():
        try:
            candidate = joblib.load(cache_path)
            candidate_signature = candidate.get('signature')
            valid_current = candidate_signature == model_signature
            valid_previous = (
                candidate_signature == previous_model_signature
            )
            valid_previous_v3 = (
                candidate_signature == previous_model_signature_v3
            )
            valid_legacy = (
                model_name != 'knn_cosine'
                and candidate_signature == legacy_model_signature
            )
            if (
                valid_current
                or valid_previous
                or valid_previous_v3
                or valid_legacy
            ):
                cached = candidate
            else:
                print('Cache status              : incompatible cache will be replaced')
        except Exception as error:
            print(f'Cache status              : unreadable cache will be replaced ({error})')
    if cached is not None:
        source = 'reused validated XAI cache'
        feature_dictionary = cached['feature_dictionary']
        compactness = cached['compactness']
        global_features = cached['global_features']
        feature_groups = cached['feature_groups']
        local_contributions = cached['local_contributions']
        faithfulness = cached['faithfulness']
        validation = cached['validation']
        if model_name == 'xgboost':
            xgboost_sensitivity_by_case = cached.get('xgboost_sensitivity_by_case')
            xgboost_sensitivity_summary = cached.get('xgboost_sensitivity_summary')
            xgboost_sensitivity_validation = cached.get('xgboost_sensitivity_validation')
    else:
        source = 'new SHAP calculation'
        pipeline = joblib.load(model_paths[model_name])
        preprocessor = pipeline.named_steps.get('preprocessor')
        estimator = pipeline.named_steps.get('classifier')
        if preprocessor is None or estimator is None:
            raise ValueError(f'Unexpected frozen pipeline structure for {model_name}.')
        analysis_matrix = ensure_csr(preprocessor.transform(analysis_data))
        background_matrix = ensure_csr(preprocessor.transform(background_sample))
        knn_background_matrix = ensure_csr(preprocessor.transform(knn_background_sample)) if model_name == 'knn_cosine' else None
        feature_dictionary = extract_feature_dictionary(preprocessor, model_name)
        if analysis_matrix.shape[1] != len(feature_dictionary):
            raise ValueError(f'Feature-name count mismatch for {model_name}.')
        pipeline_predictions = np.asarray(pipeline.predict(analysis_data), dtype=np.int8)
        stored_predictions = analysis_data[f'pred_{model_name}'].to_numpy(dtype=np.int8)
        if not np.array_equal(pipeline_predictions, stored_predictions):
            raise ValueError(f'Frozen pipeline predictions differ from Evaluation predictions for {model_name}.')
        values, base_values, validation = calculate_shap_values(model_name, estimator, analysis_matrix, background_matrix, knn_background_matrix=knn_background_matrix)
        validation.update({'model': model_name, 'model_label': MODEL_LABELS[model_name], 'model_family': MODEL_FAMILIES[model_name], 'feature_set': H2_FEATURE_SET, 'model_sha256': model_hashes[model_name], 'analysis_records': len(analysis_data), 'global_h2_records': len(base_indices), 'faithfulness_records': len(faithfulness_indices), 'background_records': (len(background_sample) if model_name in (LINEAR_MODELS + ADDITIVE_SCORE_MODELS) else len(knn_background_sample) if model_name == 'knn_cosine' else 0), 'transformed_features': analysis_matrix.shape[1], 'prediction_matches_evaluation': True, 'shap_values_shape': list(values.shape), 'base_values_shape': list(base_values.shape), 'source': source})
        feature_group_array = feature_dictionary['feature_group'].to_numpy(dtype=object)
        compactness = calculate_compactness(model_name, values, analysis_matrix, analysis_data)
        global_features, feature_groups, local_contributions = calculate_explanation_tables(model_name, values, analysis_matrix, analysis_data, feature_dictionary, base_indices)
        faithfulness = calculate_faithfulness(model_name, estimator, values, analysis_matrix, analysis_data, feature_group_array, faithfulness_indices)
        if model_name == 'xgboost':
            xgboost_sensitivity_by_case, xgboost_sensitivity_summary, xgboost_sensitivity_validation = calculate_xgboost_sensitivity(estimator, analysis_matrix, values, analysis_data, feature_group_array, xgb_sensitivity_indices)
            validation['xgboost_exact_approximate_sensitivity'] = xgboost_sensitivity_validation
        joblib.dump({'signature': model_signature, 'created_utc': datetime.now(timezone.utc).isoformat(), 'feature_dictionary': feature_dictionary, 'compactness': compactness, 'global_features': global_features, 'feature_groups': feature_groups, 'local_contributions': local_contributions, 'faithfulness': faithfulness, 'xgboost_sensitivity_by_case': xgboost_sensitivity_by_case if model_name == 'xgboost' else None, 'xgboost_sensitivity_summary': xgboost_sensitivity_summary if model_name == 'xgboost' else None, 'xgboost_sensitivity_validation': xgboost_sensitivity_validation if model_name == 'xgboost' else None, 'validation': validation}, cache_path, compress=3)
        del pipeline
        del preprocessor
        del estimator
        del analysis_matrix
        del background_matrix
        if knn_background_matrix is not None:
            del knn_background_matrix
        del values
        del base_values
        gc.collect()
    validation['source'] = source
    all_feature_dictionaries.append(feature_dictionary)
    all_compactness.append(compactness)
    all_global_features.append(global_features)
    all_feature_groups.append(feature_groups)
    all_local_contributions.append(local_contributions)
    all_faithfulness.append(faithfulness)
    validation_rows.append(validation)
    print(f'Source                    : {source}')
    print(f"Features / explainer      : {fmt_int(validation['transformed_features'])} / {validation['explainer']}")
    print(f"Max additivity error      : {validation['max_additivity_error']:.8f}")
    if model_name == 'xgboost' and xgboost_sensitivity_validation is not None:
        print(f"XGB exact/approx. rho     : {xgboost_sensitivity_validation['mean_abs_shap_rank_spearman']:.4f}")
feature_dictionary = pd.concat(all_feature_dictionaries, ignore_index=True)
compactness_by_case = pd.concat(all_compactness, ignore_index=True)
global_top_features = pd.concat(all_global_features, ignore_index=True)
feature_group_results = pd.concat(all_feature_groups, ignore_index=True)
local_contributions = pd.concat(all_local_contributions, ignore_index=True)
faithfulness_by_case = pd.concat(all_faithfulness, ignore_index=True)
validation_summary = pd.DataFrame(validation_rows)
feature_dictionary_summary = feature_dictionary.groupby(['model', 'model_label', 'model_family', 'feature_group']).size().rename('features').reset_index()
feature_dictionary_summary.to_csv(FEATURE_DICTIONARY_SUMMARY_PATH, index=False)
compactness_by_case.to_parquet(COMPACTNESS_CASE_PATH, index=False)
global_top_features.to_csv(GLOBAL_FEATURES_PATH, index=False)
local_contributions.to_csv(LOCAL_CONTRIBUTIONS_PATH, index=False)
faithfulness_by_case.to_parquet(FAITHFULNESS_CASE_PATH, index=False)
validation_summary.to_csv(VALIDATION_SUMMARY_PATH, index=False)
if xgboost_sensitivity_by_case is None or xgboost_sensitivity_summary is None or xgboost_sensitivity_validation is None:
    raise ValueError('XGBoost exact-versus-approximate sensitivity results are missing.')
xgboost_sensitivity_by_case.to_parquet(XGB_SENSITIVITY_CASE_PATH, index=False)
xgboost_sensitivity_summary.to_csv(XGB_SENSITIVITY_SUMMARY_PATH, index=False)

# %% 07 - Summarize explanation compactness

metric_columns = [*[f'top_{top_k}_share' for top_k in TOP_K_VALUES], *[f'k_{int(threshold * 100)}' for threshold in MASS_THRESHOLDS], 'normalized_entropy', 'nonzero_shap_features', 'available_features']
compactness_global = compactness_by_case.loc[compactness_by_case['analysis_role'].isin(['global_h2', 'global_h2_and_local'])].copy()
compactness_summary = compactness_global.groupby(['model', 'model_label', 'model_family', 'scope'])[metric_columns].agg(['mean', 'median', 'std']).reset_index()
compactness_summary.columns = [column if isinstance(column, str) else '_'.join((part for part in column if part)) for column in compactness_summary.columns]
compactness_summary.to_csv(COMPACTNESS_SUMMARY_PATH, index=False)
print_section('Explanation compactness')
primary_compactness_console = compactness_summary.loc[compactness_summary['scope'].eq('full_attribution'), ['model_label', 'top_10_share_mean', 'top_10_share_median', 'k_80_median', 'normalized_entropy_mean']].set_index('model_label').reindex([MODEL_LABELS[model] for model in XAI_MODEL_ORDER]).reset_index()
print(primary_compactness_console.to_string(index=False, formatters={'top_10_share_mean': '{:.4f}'.format, 'top_10_share_median': '{:.4f}'.format, 'k_80_median': '{:.1f}'.format, 'normalized_entropy_mean': '{:.4f}'.format}))

# %% 08 - Test H2 pairwise against Logistic Regression

full_primary = compactness_global.loc[compactness_global['scope'].eq('full_attribution')].copy()
primary_wide = full_primary.pivot(index=GROUP_COLUMN, columns='model', values=f'top_{PRIMARY_TOP_K}_share').reindex(columns=XAI_MODEL_ORDER)
if primary_wide.isna().any().any() or len(primary_wide) != len(global_sample):
    raise ValueError('Paired H2 compactness matrix is incomplete.')
strata = global_sample.set_index(GROUP_COLUMN)[TARGET_COLUMN].reindex(primary_wide.index).to_numpy()
primary_vectors = {alternative: (primary_wide[H2_BASELINE_MODEL] - primary_wide[alternative]).to_numpy(dtype=float) for alternative in H2_ALTERNATIVES}
primary_bootstrap = paired_stratified_bootstrap(primary_vectors, strata)
h2_rows = []
for alternative in H2_ALTERNATIVES:
    vector = primary_vectors[alternative]
    distribution = primary_bootstrap[alternative]
    observed = float(vector.mean())
    lower, upper = np.quantile(distribution, [ALPHA / 2, 1 - ALPHA / 2])
    null_distribution = distribution - distribution.mean()
    denominator = len(null_distribution) + 1
    p_one_sided = (1 + np.sum(null_distribution >= observed)) / denominator
    p_two_sided = min(1.0, (1 + np.sum(np.abs(null_distribution) >= abs(observed))) / denominator)
    h2_rows.append({'feature_set': H2_FEATURE_SET, 'baseline_model': H2_BASELINE_MODEL, 'baseline_label': MODEL_LABELS[H2_BASELINE_MODEL], 'alternative_model': alternative, 'alternative_label': MODEL_LABELS[alternative], 'primary_metric': H2_PRIMARY_METRIC, 'baseline_mean_top10': float(primary_wide[H2_BASELINE_MODEL].mean()), 'alternative_mean_top10': float(primary_wide[alternative].mean()), 'delta_lr_minus_alternative': observed, 'ci_lower_95': lower, 'ci_upper_95': upper, 'lower_bound_95_one_sided': np.quantile(distribution, ALPHA), 'p_value_one_sided_raw': p_one_sided, 'p_value_two_sided_raw': p_two_sided})
h2_results = pd.DataFrame(h2_rows)

def holm_adjust_local(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(np.minimum(1.0, sorted_values * np.arange(len(sorted_values), 0, -1)))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted
h2_results['p_value_one_sided_holm'] = holm_adjust_local(h2_results['p_value_one_sided_raw'])
h2_results['p_value_two_sided_holm'] = holm_adjust_local(h2_results['p_value_two_sided_raw'])
h2_results['significant_lr_more_compact_holm'] = (h2_results['delta_lr_minus_alternative'] > 0) & (h2_results['p_value_one_sided_holm'] < ALPHA)
h2_supported = bool(h2_results['significant_lr_more_compact_holm'].any())
h2_results = h2_results.sort_values(['p_value_one_sided_holm', 'delta_lr_minus_alternative'], ascending=[True, False], kind='mergesort').reset_index(drop=True)
h2_results.to_csv(H2_RESULTS_PATH, index=False)
print_section('H2 pairwise tests: Logistic Regression vs all six alternatives')
print(h2_results[['alternative_label', 'baseline_mean_top10', 'alternative_mean_top10', 'delta_lr_minus_alternative', 'ci_lower_95', 'ci_upper_95', 'p_value_one_sided_raw', 'p_value_one_sided_holm', 'significant_lr_more_compact_holm']].to_string(index=False, formatters={'baseline_mean_top10': '{:.4f}'.format, 'alternative_mean_top10': '{:.4f}'.format, 'delta_lr_minus_alternative': '{:+.4f}'.format, 'ci_lower_95': '{:+.4f}'.format, 'ci_upper_95': '{:+.4f}'.format, 'p_value_one_sided_raw': '{:.6f}'.format, 'p_value_one_sided_holm': '{:.6f}'.format}))

# %% 09 - Run pairwise compactness sensitivity checks

sensitivity_specs = [('full_top_5_share', 'full_attribution', 'top_5_share', 'baseline_minus_alternative'), ('full_top_20_share', 'full_attribution', 'top_20_share', 'baseline_minus_alternative'), ('full_k_80', 'full_attribution', 'k_80', 'alternative_minus_baseline'), ('full_normalized_entropy', 'full_attribution', 'normalized_entropy', 'alternative_minus_baseline'), ('active_top_10_share', 'active_features', 'top_10_share', 'baseline_minus_alternative')]
sensitivity_vectors = {}
sensitivity_metadata = {}
for name, scope, metric, direction in sensitivity_specs:
    matrix = compactness_global.loc[compactness_global['scope'].eq(scope)].pivot(index=GROUP_COLUMN, columns='model', values=metric).reindex(index=primary_wide.index, columns=XAI_MODEL_ORDER)
    if matrix.isna().any().any():
        raise ValueError(f'Incomplete H2 sensitivity matrix: {name}')
    for alternative in H2_ALTERNATIVES:
        key = f'{name}__{alternative}'
        if direction == 'baseline_minus_alternative':
            vector = matrix[H2_BASELINE_MODEL] - matrix[alternative]
        else:
            vector = matrix[alternative] - matrix[H2_BASELINE_MODEL]
        sensitivity_vectors[key] = vector.to_numpy(dtype=float)
        sensitivity_metadata[key] = {'name': name, 'metric': metric, 'scope': scope, 'alternative': alternative, 'positive_direction': 'supports greater LR compactness'}
sensitivity_bootstrap = paired_stratified_bootstrap(sensitivity_vectors, strata)
sensitivity_rows = []
for key, vector in sensitivity_vectors.items():
    distribution = sensitivity_bootstrap[key]
    lower, upper = np.quantile(distribution, [ALPHA / 2, 1 - ALPHA / 2])
    meta = sensitivity_metadata[key]
    sensitivity_rows.append({'analysis': meta['name'], 'scope': meta['scope'], 'metric': meta['metric'], 'baseline_model': H2_BASELINE_MODEL, 'alternative_model': meta['alternative'], 'alternative_label': MODEL_LABELS[meta['alternative']], 'positive_direction': meta['positive_direction'], 'observed_difference': float(vector.mean()), 'ci_lower_95': lower, 'ci_upper_95': upper, 'iterations': BOOTSTRAP_ITERATIONS})
h2_sensitivity_results = pd.DataFrame(sensitivity_rows)
h2_sensitivity_results.to_csv(H2_SENSITIVITY_PATH, index=False)

# %% 10 - Summarize feature groups and explanation faithfulness

global_feature_groups = feature_group_results.loc[feature_group_results['result_level'].eq('global_class_output')].copy()
case_feature_groups = feature_group_results.loc[feature_group_results['result_level'].eq('case_predicted_class')].copy()
case_feature_group_summary = case_feature_groups.groupby(['model', 'model_label', 'model_family', 'feature_group'])['share_of_total_abs_shap'].agg(['mean', 'median', 'std']).reset_index().rename(columns={'mean': 'mean_share', 'median': 'median_share', 'std': 'std_share'})
global_feature_groups = global_feature_groups.merge(case_feature_group_summary, on=['model', 'model_label', 'model_family', 'feature_group'], how='left', validate='many_to_one')
global_feature_groups.to_csv(FEATURE_GROUPS_PATH, index=False)
faithfulness_summary = faithfulness_by_case.groupby(['model', 'model_label', 'model_family', 'top_k']).agg(mean_comprehensiveness_top=('comprehensiveness_top', 'mean'), median_comprehensiveness_top=('comprehensiveness_top', 'median'), flip_rate_top=('flip_top', 'mean'), mean_comprehensiveness_random=('comprehensiveness_random_mean', 'mean'), mean_flip_rate_random=('flip_random_mean', 'mean'), mean_comprehensiveness_advantage=('comprehensiveness_advantage', 'mean'), top_stronger_than_random_share=('top_stronger_than_random', 'mean'), cases=(GROUP_COLUMN, 'nunique')).reset_index()
faithfulness_summary.to_csv(FAITHFULNESS_SUMMARY_PATH, index=False)
print_section('Explanation faithfulness at k=10')
faithfulness_console = faithfulness_summary.loc[faithfulness_summary['top_k'].eq(PRIMARY_TOP_K), ['model_label', 'top_stronger_than_random_share', 'flip_rate_top', 'mean_flip_rate_random']].set_index('model_label').reindex([MODEL_LABELS[model] for model in XAI_MODEL_ORDER]).reset_index()
print(faithfulness_console.to_string(index=False, formatters={'top_stronger_than_random_share': '{:.4f}'.format, 'flip_rate_top': '{:.4f}'.format, 'mean_flip_rate_random': '{:.4f}'.format}))

# %% 11 - Create qualitative plausibility-assessment template

local_compactness = compactness_by_case.loc[compactness_by_case['scope'].eq('full_attribution') & compactness_by_case[GROUP_COLUMN].isin(local_case_details[GROUP_COLUMN])].copy()
assessment_template = local_case_details[['case_id', 'case_type', 'modeling_row_id', GROUP_COLUMN, TARGET_COLUMN, 'type', 'language', 'queue', TEXT_COLUMN]].merge(local_compactness[['model', 'model_label', 'model_family', 'modeling_row_id', 'predicted_class', 'correct', 'top_10_share', 'k_80', 'normalized_entropy']], on='modeling_row_id', how='left', validate='one_to_many')
for direction, output_column in [('supports_prediction', 'top_supporting_features'), ('opposes_prediction', 'top_opposing_features')]:
    feature_text = local_contributions.loc[local_contributions['direction'].eq(direction)].sort_values(['case_id', 'model', 'rank']).groupby(['case_id', 'model'])['feature_label'].apply(lambda values: ' | '.join(values.astype(str).head(5))).rename(output_column).reset_index()
    assessment_template = assessment_template.merge(feature_text, on=['case_id', 'model'], how='left', validate='one_to_one')
for column in ['text_relation', 'urgency_plausibility', 'impact_plausibility', 'context_relevance', 'contradictory_signals', 'overall_rating', 'brief_rationale']:
    assessment_template[column] = ''
assessment_template['evaluator_role'] = 'author'
assessment_template.to_csv(QUALITATIVE_TEMPLATE_PATH, index=False)


# %% 12 - Explain final overall model

print_section(f'Final model SHAP analysis: {FINAL_OVERALL_MODEL_KEY[0]} + {MODEL_LABELS[FINAL_OVERALL_MODEL_KEY[1]]}')

final_model_signature = stable_hash({
    'script_build': SCRIPT_BUILD,
    'data_sha256': data_sha256,
    'model_sha256': final_model_hash,
    'sample_signature': sample_signature,
    'feature_set': FINAL_OVERALL_MODEL_KEY[0],
    'model': FINAL_OVERALL_MODEL_KEY[1],
    'sentiment_sha256': sha256_file(SENTIMENT_PATH),
    'package_versions': PACKAGE_VERSIONS,
})

previous_final_model_signature = stable_hash({
    'script_build': PREVIOUS_XAI_BUILD,
    'data_sha256': data_sha256,
    'model_sha256': final_model_hash,
    'sample_signature': sample_signature,
    'feature_set': FINAL_OVERALL_MODEL_KEY[0],
    'model': FINAL_OVERALL_MODEL_KEY[1],
    'sentiment_sha256': sha256_file(SENTIMENT_PATH),
    'package_versions': PACKAGE_VERSIONS,
})

final_model_cached = None
if REUSE_VALID_RESULTS and FINAL_MODEL_CACHE_PATH.exists():
    try:
        candidate = joblib.load(FINAL_MODEL_CACHE_PATH)
        if candidate.get('signature') in {final_model_signature, previous_final_model_signature}:
            final_model_cached = candidate
    except Exception:
        final_model_cached = None

if final_model_cached is None:
    final_pipeline = joblib.load(final_model_path)
    final_preprocessor = final_pipeline.named_steps.get('preprocessor')
    final_estimator = final_pipeline.named_steps.get('classifier')

    if final_preprocessor is None or final_estimator is None:
        raise ValueError(
            'Unexpected final overall-model pipeline structure.'
        )

    final_analysis_matrix = ensure_csr(
        final_preprocessor.transform(analysis_data)
    )
    final_feature_dictionary = extract_feature_dictionary(
        final_preprocessor,
        FINAL_OVERALL_MODEL_KEY[1],
        FINAL_FEATURE_GROUP_ORDER,
    )

    if final_analysis_matrix.shape[1] != len(final_feature_dictionary):
        raise ValueError('Final-model feature-name count mismatch.')

    observed_groups = set(
        final_feature_dictionary['feature_group'].astype(str)
    )
    if observed_groups != set(FINAL_FEATURE_GROUP_ORDER):
        raise ValueError(
            f'Unexpected final-model feature groups: {sorted(observed_groups)}'
        )

    final_pipeline_predictions = np.asarray(
        final_pipeline.predict(analysis_data),
        dtype=np.int8,
    )
    final_stored_predictions = analysis_data[
        'pred_final_model'
    ].to_numpy(dtype=np.int8)

    if not np.array_equal(
        final_pipeline_predictions,
        final_stored_predictions,
    ):
        raise ValueError(
            'Final overall-model predictions differ from Evaluation predictions.'
        )

    (
        final_values,
        final_base_values,
        final_validation,
    ) = calculate_shap_values(
        FINAL_OVERALL_MODEL_KEY[1],
        final_estimator,
        final_analysis_matrix,
        ensure_csr(final_preprocessor.transform(background_sample)) if FINAL_OVERALL_MODEL_KEY[1] in (LINEAR_MODELS + ADDITIVE_SCORE_MODELS) else None,
        knn_background_matrix=(ensure_csr(final_preprocessor.transform(knn_background_sample)) if FINAL_OVERALL_MODEL_KEY[1] == 'knn_cosine' else None),
    )

    final_validation.update({
        'feature_set': FINAL_OVERALL_MODEL_KEY[0],
        'model': FINAL_OVERALL_MODEL_KEY[1],
        'model_label': MODEL_LABELS[FINAL_OVERALL_MODEL_KEY[1]],
        'model_sha256': final_model_hash,
        'analysis_records': len(analysis_data),
        'global_records': len(base_indices),
        'local_cases': int(analysis_data['is_local_case'].sum()),
        'transformed_features': final_analysis_matrix.shape[1],
        'prediction_matches_evaluation': True,
        'shap_values_shape': list(final_values.shape),
        'base_values_shape': list(final_base_values.shape),
        'source': 'new SHAP calculation',
    })

    (
        final_global_top_features,
        final_feature_group_summary,
        final_local_contributions,
    ) = calculate_final_model_tables(
        final_values,
        final_analysis_matrix,
        analysis_data,
        final_feature_dictionary,
        base_indices,
        'pred_final_model',
        FINAL_OVERALL_MODEL_KEY[0],
        FINAL_OVERALL_MODEL_KEY[1],
    )

    joblib.dump(
        {
            'signature': final_model_signature,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'feature_dictionary': final_feature_dictionary,
            'global_top_features': final_global_top_features,
            'feature_group_summary': final_feature_group_summary,
            'local_contributions': final_local_contributions,
            'validation': final_validation,
        },
        FINAL_MODEL_CACHE_PATH,
        compress=3,
    )

    del final_pipeline
    del final_preprocessor
    del final_estimator
    del final_analysis_matrix
    del final_values
    del final_base_values
    gc.collect()

else:
    final_global_top_features = final_model_cached[
        'global_top_features'
    ]
    final_feature_group_summary = final_model_cached[
        'feature_group_summary'
    ]
    final_local_contributions = final_model_cached[
        'local_contributions'
    ]
    final_validation = final_model_cached['validation']
    final_validation['source'] = 'reused validated XAI cache'

final_global_top_features.to_csv(
    FINAL_MODEL_GLOBAL_FEATURES_PATH,
    index=False,
)
final_feature_group_summary.to_csv(
    FINAL_MODEL_FEATURE_GROUPS_PATH,
    index=False,
)
final_local_contributions.to_csv(
    FINAL_MODEL_LOCAL_CONTRIBUTIONS_PATH,
    index=False,
)
pd.DataFrame([final_validation]).to_csv(
    FINAL_MODEL_VALIDATION_PATH,
    index=False,
)

print(f"Source                    : {final_validation['source']}")
print(
    f"Features / explainer      : "
    f"{fmt_int(final_validation['transformed_features'])} / "
    f"{final_validation['explainer']}"
)
print(
    f"Max additivity error      : "
    f"{final_validation['max_additivity_error']:.8f}"
)


# %% 13 - Create grayscale paper-ready figures

plot_values = [full_primary.loc[full_primary['model'].eq(model), f'top_{PRIMARY_TOP_K}_share'].to_numpy() for model in XAI_MODEL_ORDER]
figure, axis = plt.subplots(figsize=(8.6, 4.2))
axis.boxplot(plot_values, showfliers=False, medianprops={'color': 'black', 'linewidth': 1.2}, boxprops={'color': 'black'}, whiskerprops={'color': 'black'}, capprops={'color': 'black'})
axis.set_xticks(np.arange(1, len(XAI_MODEL_ORDER) + 1), labels=[MODEL_LABELS[model] for model in XAI_MODEL_ORDER])
axis.set_ylabel(f'Top-{PRIMARY_TOP_K} share of absolute SHAP mass')
axis.tick_params(axis='x', rotation=20)
axis.grid(axis='y', color='0.80', linestyle=':', linewidth=0.5)
axis.set_axisbelow(True)
for spine in ['top', 'right']:
    axis.spines[spine].set_visible(False)
figure.tight_layout()
save_png(figure, '05_h2_compactness_by_model.png')
h2_plot = h2_results.set_index('alternative_model').reindex(H2_ALTERNATIVES).reset_index()
positions = np.arange(len(h2_plot))
figure, axis = plt.subplots(figsize=(6.8, 3.4))
axis.errorbar(h2_plot['delta_lr_minus_alternative'], positions, xerr=np.vstack([h2_plot['delta_lr_minus_alternative'] - h2_plot['ci_lower_95'], h2_plot['ci_upper_95'] - h2_plot['delta_lr_minus_alternative']]), fmt='o', color='black', ecolor='black', capsize=4, linewidth=1.0)
axis.axvline(0, color='0.40', linestyle='--', linewidth=0.8)
axis.set_yticks(positions, labels=[MODEL_LABELS[model] for model in H2_ALTERNATIVES])
axis.invert_yaxis()
axis.set_xlabel(f'Logistic Regression minus alternative Top-{PRIMARY_TOP_K} SHAP concentration (95% bootstrap interval)')
axis.grid(axis='x', color='0.80', linestyle=':', linewidth=0.5)
axis.set_axisbelow(True)
for spine in ['top', 'right', 'left']:
    axis.spines[spine].set_visible(False)
axis.tick_params(axis='y', length=0)
figure.tight_layout()
save_png(figure, '05_h2_pairwise_differences.png')
predicted_group_plot = case_feature_group_summary.pivot(index='model', columns='feature_group', values='mean_share').reindex(index=XAI_MODEL_ORDER, columns=FEATURE_GROUP_ORDER)
x_positions = np.arange(len(XAI_MODEL_ORDER))
bar_width = 0.18
gray_levels = np.linspace(0.3, 0.78, len(FEATURE_GROUP_ORDER))
gray_colors = [plt.cm.Greys(level) for level in gray_levels]
figure, axis = plt.subplots(figsize=(8.8, 4.4))
for group_index, group in enumerate(FEATURE_GROUP_ORDER):
    offset = (group_index - (len(FEATURE_GROUP_ORDER) - 1) / 2) * bar_width
    axis.bar(x_positions + offset, predicted_group_plot[group].to_numpy(), width=bar_width, label=group.capitalize(), facecolor=gray_colors[group_index], edgecolor='black', linewidth=0.4)
axis.set_xticks(x_positions, labels=[MODEL_LABELS[model] for model in XAI_MODEL_ORDER], rotation=20)
axis.set_ylabel('Mean share of absolute SHAP mass')
axis.grid(axis='y', color='0.80', linestyle=':', linewidth=0.5)
axis.set_axisbelow(True)
for spine in ['top', 'right']:
    axis.spines[spine].set_visible(False)
axis.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.18))
figure.tight_layout()
save_png(figure, '05_feature_group_shares_by_model.png')
for old_path in FIGURE_DIR.glob('05_top_features_*'):
    if old_path.name.startswith('05_top_features_') and FINAL_MODEL_TAG not in old_path.name:
        old_path.unlink()

for class_output in CLASS_ORDER:
    plot_data = (
        final_global_top_features.loc[
            final_global_top_features['class_output'].eq(class_output)
        ]
        .head(15)
        .sort_values('mean_abs_shap')
    )
    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    axis.barh(
        plot_data['feature_label'],
        plot_data['mean_abs_shap'],
        color='0.70',
        edgecolor='black',
        linewidth=0.4,
    )
    axis.set_xlabel('Mean absolute SHAP value')
    axis.set_ylabel('Feature')
    axis.grid(axis='x', color='0.80', linestyle=':', linewidth=0.5)
    axis.set_axisbelow(True)
    for spine in ['top', 'right']:
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    save_png(
        figure,
        f'05_top_features_{FINAL_MODEL_TAG}_{class_output}.png',
    )

# %% 14 - Save Explainability manifest

output_paths = {
    'xai_model_eligibility': XAI_ELIGIBILITY_PATH,
    'xai_sample': XAI_SAMPLE_PATH,
    'linear_shap_background_sample': BACKGROUND_SAMPLE_PATH,
    'knn_permutation_background_sample': KNN_BACKGROUND_SAMPLE_PATH,
    'feature_dictionary_summary': FEATURE_DICTIONARY_SUMMARY_PATH,
    'compactness_by_case': COMPACTNESS_CASE_PATH,
    'compactness_summary': COMPACTNESS_SUMMARY_PATH,
    'h2_pairwise_results': H2_RESULTS_PATH,
    'h2_sensitivity_results': H2_SENSITIVITY_PATH,
    'global_top_features': GLOBAL_FEATURES_PATH,
    'feature_group_summary': FEATURE_GROUPS_PATH,
    'local_explanation_cases': LOCAL_CASES_PATH,
    'local_feature_contributions': LOCAL_CONTRIBUTIONS_PATH,
    'faithfulness_by_case': FAITHFULNESS_CASE_PATH,
    'faithfulness_summary': FAITHFULNESS_SUMMARY_PATH,
    'xgboost_sensitivity_by_case': XGB_SENSITIVITY_CASE_PATH,
    'xgboost_sensitivity_summary': XGB_SENSITIVITY_SUMMARY_PATH,
    'qualitative_assessment_template': QUALITATIVE_TEMPLATE_PATH,
    'validation_summary': VALIDATION_SUMMARY_PATH,
    'final_model_global_top_features': FINAL_MODEL_GLOBAL_FEATURES_PATH,
    'final_model_feature_group_summary': FINAL_MODEL_FEATURE_GROUPS_PATH,
    'final_model_local_contributions': FINAL_MODEL_LOCAL_CONTRIBUTIONS_PATH,
    'final_model_validation': FINAL_MODEL_VALIDATION_PATH,
}
expected_figure_names = ['05_h2_compactness_by_model.png', '05_h2_pairwise_differences.png', '05_feature_group_shares_by_model.png', *[f'05_top_features_{FINAL_MODEL_TAG}_{class_output}.png' for class_output in CLASS_ORDER]]
figure_paths = [FIGURE_DIR / name for name in expected_figure_names]
missing_figures = [path.name for path in figure_paths if not path.exists()]
if missing_figures:
    raise FileNotFoundError(f'Expected Explainability figures are missing: {missing_figures}')
save_json({
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'script_build': SCRIPT_BUILD,
    'run_id': RUN_ID,
    'script': SCRIPT_PATH.name,
    'script_sha256': sha256_file(SCRIPT_PATH),
    'modeling_build': EXPECTED_MODELING_BUILD,
    'modeling_run_id': EXPECTED_MODELING_RUN_ID,
    'evaluation_build': EXPECTED_EVALUATION_BUILD,
    'data_file': str(DATA_PATH.relative_to(PROJECT_ROOT)),
    'data_sha256': data_sha256,
    'records': len(data),
    'training_records': len(training_data),
    'training_semantic_groups': training_data[GROUP_COLUMN].nunique(),
    'holdout_records': len(holdout_data),
    'holdout_semantic_groups': holdout_data[GROUP_COLUMN].nunique(),
    'feature_set': H2_FEATURE_SET,
    'all_frozen_models': ALL_MODEL_ORDER,
    'h2_models': XAI_MODEL_ORDER,
    'h2_excluded_models': {},
    'model_hashes': model_hashes,
    'global_groups_per_class': GLOBAL_GROUPS_PER_CLASS,
    'global_h2_groups': len(global_sample),
    'faithfulness_groups_per_class': FAITHFULNESS_GROUPS_PER_CLASS,
    'faithfulness_groups': len(faithfulness_sample),
    'xgboost_sensitivity_groups_per_class': XGB_SENSITIVITY_GROUPS_PER_CLASS,
    'xgboost_sensitivity_groups': len(xgb_sensitivity_sample),
    'local_cases': len(local_case_details),
    'unique_explained_groups': len(analysis_data),
    'linear_background_groups_per_class': BACKGROUND_GROUPS_PER_CLASS,
    'knn_permutation_background_groups_per_class': KNN_BACKGROUND_GROUPS_PER_CLASS,
    'knn_permutation_cycles': KNN_PERMUTATION_CYCLES,
    'h2': {
        'feature_set': H2_FEATURE_SET,
        'baseline': H2_BASELINE_MODEL,
        'alternatives': H2_ALTERNATIVES,
        'primary_metric': H2_PRIMARY_METRIC,
        'contrast': 'Top10(Logistic Regression) - Top10(alternative)',
        'pairwise': True,
        'family_mean_comparison': False,
        'number_of_comparisons': len(H2_ALTERNATIVES),
        'multiple_testing': 'holm_one_sided_6',
        'alpha': ALPHA,
        'supported': h2_supported,
        'supporting_alternatives': h2_results.loc[h2_results['significant_lr_more_compact_holm'], 'alternative_model'].tolist(),
    },
    'compactness_sensitivity_metrics': ['Top-5 concentration', 'Top-20 concentration', 'K80', 'normalized entropy', 'active-feature Top-10 concentration'],
    'faithfulness': {
        'metrics': ['comprehensiveness', 'flip rate'],
        'groups_per_class': FAITHFULNESS_GROUPS_PER_CLASS,
        'groups': len(faithfulness_sample),
        'top_k_values': FAITHFULNESS_TOP_K_VALUES,
        'primary_reporting_k': PRIMARY_TOP_K,
        'perturbation_scope': 'active transformed text features',
        'random_baseline_repeats': FAITHFULNESS_RANDOM_REPEATS,
    },
    'bootstrap': {
        'sampling_unit': GROUP_COLUMN,
        'stratification': TARGET_COLUMN,
        'iterations': BOOTSTRAP_ITERATIONS,
        'random_state': RANDOM_STATE,
        'paired_across_models': True,
    },
    'explainer_assignment': {
        'logistic_regression': 'LinearExplainer',
        'linear_svm': 'LinearExplainer',
        'complement_naive_bayes': 'LinearExplainer (ComplementNB score)',
        'knn_cosine': 'PermutationExplainer',
        'random_forest': 'TreeExplainer approximate=True',
        'xgboost': 'TreeExplainer approximate=True',
        'lightgbm': 'LightGBM native TreeSHAP',
    },
    'tree_shap_approximation': {
        'approximate_models': APPROXIMATE_TREE_MODELS,
        'exact_tree_shap_models': ['lightgbm'],
        'xgboost_exact_approximate_sensitivity': xgboost_sensitivity_validation,
    },
    'model_training_performed': False,
    'hyperparameter_tuning_performed': False,
    'model_selection_performed': False,
    'h2_model_eligibility_based_on_performance': False,
    'common_feature_set_selection': {
        'feature_set': H2_FEATURE_SET,
        'source': 'Modeling training-only cross-validation',
        'selection_performed_in_explainability': False,
        'selection_artifact': FEATURE_SET_SELECTION_PATH.name,
    },
    'report_model': {
        'feature_set': FINAL_OVERALL_MODEL_KEY[0],
        'model': FINAL_OVERALL_MODEL_KEY[1],
    },
    'final_overall_model_from_evaluation': {
        'feature_set': FINAL_OVERALL_MODEL_KEY[0],
        'model': FINAL_OVERALL_MODEL_KEY[1],
    },
    'final_model_explainability': {
        'feature_set': FINAL_OVERALL_MODEL_KEY[0],
        'model': FINAL_OVERALL_MODEL_KEY[1],
        'model_sha256': final_model_hash,
        'sentiment_sha256': sha256_file(SENTIMENT_PATH),
        'feature_groups': FINAL_FEATURE_GROUP_ORDER,
        'global_groups': len(base_indices),
        'local_cases': int(analysis_data['is_local_case'].sum()),
        'validation': final_validation,
    },
    'local_case_selection': {
        'target_cases': LOCAL_CASE_TARGET,
    },
    'qualitative_assessment': {
        'template_file': str(QUALITATIVE_TEMPLATE_PATH.relative_to(PROJECT_ROOT)),
        'evaluator_role': 'author',
    },
    'package_versions': PACKAGE_VERSIONS,
    'cache_files': {
        model: {
            'path': str(path.relative_to(PROJECT_ROOT)),
            'sha256': sha256_file(path),
        }
        for model, path in cache_paths.items()
    },
    'final_model_cache': {
        'path': str(FINAL_MODEL_CACHE_PATH.relative_to(PROJECT_ROOT)),
        'sha256': sha256_file(FINAL_MODEL_CACHE_PATH),
    },
    'output_hashes': {
        name: sha256_file(path)
        for name, path in output_paths.items()
    },
    'figure_hashes': {
        path.name: sha256_file(path)
        for path in figure_paths
    },
}, MANIFEST_PATH)

# %% 15 - Finish

print_section('CRISP-DM Explainability completed')
print(f'Common comparison set     : {H2_FEATURE_SET}')
print(f'Frozen common classifiers : {len(ALL_MODEL_ORDER)}')
print(f'SHAP-analyzed H2 models   : {len(XAI_MODEL_ORDER)}')
print(f'H2 pairwise contrasts     : {len(H2_ALTERNATIVES)}')
print(f'Global H2 groups          : {fmt_int(len(global_sample))}')
print(f'Local comparison cases    : {len(local_case_details)}')
print(f'Final model explained     : {FINAL_OVERALL_MODEL_KEY[0]} + {MODEL_LABELS[FINAL_OVERALL_MODEL_KEY[1]]}')
print(f'Figures created           : {len(figure_paths)}')
print(f"XGB sensitivity rho       : {xgboost_sensitivity_validation['mean_abs_shap_rank_spearman']:.4f}")
print(f'Assessment template       : {QUALITATIVE_TEMPLATE_PATH.name}')
print(f'Manifest                  : {MANIFEST_PATH.name}')
