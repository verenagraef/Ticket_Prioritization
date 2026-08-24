# Reproducibility Input/Output File Map

All paths are relative to the project root. The project-root folder name itself is arbitrary.

The final repository intentionally preserves the validated raw dataset and the frozen generated artifacts from the final analysis run. The raw dataset is redistributed under CC BY 4.0 as documented in `DATASET_LICENSE.md`. Automatically generated artifacts can nevertheless be regenerated from the listed inputs for a clean reproduction test.

## Repository-level reproducibility files

| Role | Relative path | Purpose |
|---|---|---|
| ENVIRONMENT | `environment.yml` | Pinned final Windows x86-64 Conda/Python environment. |
| CODE LICENSE | `LICENSE` | MIT license for original repository code. |
| DATASET LICENSE | `DATASET_LICENSE.md` | CC BY 4.0 attribution and redistribution notice for the source dataset and derived-data distinction. |
| CITATION | `CITATION.cff` | Citation metadata for the repository/software release. |
| DOCUMENTATION | `README.md` | Reproduction instructions and repository overview. |

## Pipeline input/output map

| Stage | Role | Relative path | Purpose |
|---|---|---|---|
| 01 | INPUT | `data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv` | Required external raw dataset; exact SHA-256 validated. |
| 01 | OUTPUT | `cache/01_tickets_data_understanding/language_detection_detailed.parquet` | Deterministic language-detection artifact. |
| 01 | OUTPUT | `cache/01_tickets_data_understanding/multilingual_embeddings.npy` | Normalized multilingual sentence embeddings. |
| 01 | OUTPUT | `cache/01_tickets_data_understanding/multilingual_embedding_source_rows.npy` | Source-row IDs aligned to embedding matrix. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/01_data_understanding_manifest.json` | Data Understanding provenance manifest. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/variable_profile.csv` | Variable profile. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/categorical_distributions.csv` | Categorical distributions. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/text_summary.csv` | Text summary. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/text_quality_indicators.csv` | Text-quality indicators. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/short_text_screening_candidates.csv` | Short-text screening audit. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/body_answer_threshold_sensitivity.csv` | Body/answer leakage-threshold sensitivity. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/data_quality_summary.csv` | Data-quality summary. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/language_label_audit_summary.csv` | Language audit summary. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/semantic_redundancy_summary.csv` | Semantic-neighbour/redundancy summary. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/priority_associations.csv` | Priority association tests. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/semantic_pair_review_sample.csv` | Deterministic semantic review template. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/language_label_manual_review_sample.csv` | Deterministic language review template. |
| 01 | OUTPUT | `reports/tables/01_tickets_data_understanding/priority_label_plausibility_review_sample.csv` | Deterministic priority review template. |
| 01 | OUTPUT | `reports/figures/01_tickets_data_understanding/priority_distribution.png` | Figure. |
| 01 | OUTPUT | `reports/figures/01_tickets_data_understanding/priority_within_queue.png` | Figure. |
| 01 | OUTPUT | `reports/figures/01_tickets_data_understanding/cramers_v_priority_associations.png` | Figure. |
| 01a | INPUT | `reports/tables/01_tickets_data_understanding/01_data_understanding_manifest.json` | Must come from current 01 run. |
| 01a | INPUT | `reports/tables/01_tickets_data_understanding/semantic_pair_review_sample.csv` | Generated template from 01. |
| 01a | INPUT | `reports/tables/01_tickets_data_understanding/language_label_manual_review_sample.csv` | Generated template from 01. |
| 01a | INPUT | `reports/tables/01_tickets_data_understanding/priority_label_plausibility_review_sample.csv` | Generated template from 01. |
| 01a | FROZEN HUMAN INPUT | `reports/tables/01_tickets_data_understanding/semantic_pair_review_sample_AUTHOR_CONFIRMED.csv` | Must be retained/provided. |
| 01a | FROZEN HUMAN INPUT | `reports/tables/01_tickets_data_understanding/language_label_manual_review_sample_AUTHOR_CONFIRMED.csv` | Must be retained/provided. |
| 01a | FROZEN HUMAN INPUT | `reports/tables/01_tickets_data_understanding/priority_label_plausibility_review_sample_AUTHOR_CONFIRMED.csv` | Must be retained/provided. |
| 01a | OUTPUT | `reports/tables/01a_manual_review_validation/01a_validated_language_decisions.csv` | Validated language decisions consumed by 02. |
| 01a | OUTPUT | `reports/tables/01a_manual_review_validation/01a_manual_review_summary.csv` | Review summary. |
| 01a | OUTPUT | `reports/tables/01a_manual_review_validation/01a_manual_review_manifest.json` | Review provenance manifest. |
| 02 | INPUT | `data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv` | Raw source. |
| 02 | INPUT | `reports/tables/01_tickets_data_understanding/01_data_understanding_manifest.json` | 01 provenance. |
| 02 | INPUT | `reports/tables/01a_manual_review_validation/01a_manual_review_manifest.json` | 01a provenance. |
| 02 | INPUT | `reports/tables/01a_manual_review_validation/01a_validated_language_decisions.csv` | Validated human language decisions. |
| 02 | INPUT | `cache/01_tickets_data_understanding/language_detection_detailed.parquet` | Language audit artifact. |
| 02 | INPUT | `cache/01_tickets_data_understanding/multilingual_embeddings.npy` | Semantic embeddings. |
| 02 | INPUT | `cache/01_tickets_data_understanding/multilingual_embedding_source_rows.npy` | Embedding row IDs. |
| 02 | OUTPUT | `data/processed/customer_it_support_prepared.csv` | Final prepared modeling dataset. |
| 02 | OUTPUT | `reports/tables/02_tickets_data_preparation/data_preparation_summary.csv` | Preparation summary. |
| 02 | OUTPUT | `reports/tables/02_tickets_data_preparation/semantic_group_cohesion.csv` | Within-group cohesion audit. |
| 02 | OUTPUT | `reports/tables/02_tickets_data_preparation/semantic_group_summary.csv` | Semantic grouping summary. |
| 02 | OUTPUT | `reports/tables/02_tickets_data_preparation/semantic_priority_conflict_groups.csv` | Target-conflict audit. |
| 02 | OUTPUT | `reports/tables/02_tickets_data_preparation/excluded_record_audit.csv` | Excluded-record audit. |
| 02 | OUTPUT | `reports/tables/02_tickets_data_preparation/language_correction_audit.csv` | Language-correction audit. |
| 02 | OUTPUT | `reports/tables/02_tickets_data_preparation/language_correction_summary.csv` | Language-correction summary. |
| 02 | OUTPUT | `reports/tables/02_tickets_data_preparation/language_scope_summary.csv` | Language-scope summary. |
| 02 | OUTPUT | `reports/tables/02_tickets_data_preparation/text_normalization_audit.csv` | Text-normalization audit. |
| 02 | OUTPUT | `reports/tables/02_tickets_data_preparation/final_priority_distribution.csv` | Final target distribution. |
| 03a | INPUT | `data/processed/customer_it_support_prepared.csv` | Prepared dataset. |
| 03a | GENERATED/REUSED | `cache/03_tickets_modeling/03_split_assignments.parquet` | Canonical deterministic train/hold-out split; created if missing. |
| 03a | OUTPUT | `reports/tables/03a_algorithm_screening/03a_algorithm_screening_pool.csv` | Shared training-only screening pool consumed by 03b. |
| 03a | OUTPUT | `reports/tables/03a_algorithm_screening/03a_algorithm_screening_summary.csv` | Screening summary. |
| 03a | OUTPUT | `reports/tables/03a_algorithm_screening/03a_algorithm_screening_candidate_results.csv` | Candidate results. |
| 03a | OUTPUT | `reports/tables/03a_algorithm_screening/03a_algorithm_screening_fold_results.csv` | Fold-level screening results. |
| 03a | OUTPUT | `reports/tables/03a_algorithm_screening/03a_algorithm_screening_class_results.csv` | Class-level screening results. |
| 03a | OUTPUT | `reports/tables/03a_algorithm_screening/03a_algorithm_screening_failed_evaluations.csv` | Failure log. |
| 03a | OUTPUT | `reports/tables/03a_algorithm_screening/03a_algorithm_screening_manifest.json` | Screening manifest. |
| 03a | OUTPUT | `reports/figures/03a_algorithm_screening/03a_algorithm_screening_macro_f1.png` | Screening figure. |
| 03b | INPUT | `data/processed/customer_it_support_prepared.csv` | Prepared dataset. |
| 03b | INPUT | `cache/03_tickets_modeling/03_split_assignments.parquet` | Canonical split from 03a. |
| 03b | INPUT | `reports/tables/03a_algorithm_screening/03a_algorithm_screening_pool.csv` | Exact 03a screening pool. |
| 03b | OUTPUT | `reports/tables/03b_representation_screening/03b_representation_screening_results.csv` | Representation results. |
| 03b | OUTPUT | `reports/tables/03b_representation_screening/03b_representation_screening_fold_results.csv` | Fold results. |
| 03b | OUTPUT | `reports/tables/03b_representation_screening/03b_representation_screening_class_results.csv` | Class results. |
| 03b | OUTPUT | `reports/tables/03b_representation_screening/03b_representation_screening_manifest.json` | Representation-screening manifest. |
| 03b | OUTPUT | `reports/figures/03b_representation_screening/03b_representation_macro_f1.png` | Representation figure. |
| 03 | INPUT | `data/processed/customer_it_support_prepared.csv` | Prepared dataset. |
| 03 | INPUT | `reports/tables/02_tickets_data_preparation/data_preparation_summary.csv` | Preparation provenance. |
| 03 | INPUT | `reports/tables/02_tickets_data_preparation/semantic_group_cohesion.csv` | Semantic cohesion audit. |
| 03 | INPUT | `reports/tables/03a_algorithm_screening/03a_algorithm_screening_manifest.json` | Algorithm-screening provenance. |
| 03 | INPUT | `reports/tables/03b_representation_screening/03b_representation_screening_manifest.json` | Representation-screening provenance. |
| 03 | GENERATED/REUSED | `cache/03_tickets_modeling/03_split_assignments.parquet` | Validated/recreated canonical split. |
| 03 | GENERATED/REUSED | `cache/03_tickets_modeling/03_cv_fold_assignments.parquet` | Deterministic five-fold assignments. |
| 03 | GENERATED/REUSED | `cache/03_tickets_modeling/03_sentiment_features.csv` | Fixed-pretrained sentiment probabilities. |
| 03 | GENERATED/REUSED | `cache/03_tickets_modeling/03_sentiment_features_metadata.json` | Sentiment provenance. |
| 03 | GENERATED/REUSED | `cache/03_tickets_modeling/03_model_parameters.json` | Selected parameter set. |
| 03 | GENERATED/REUSED | `cache/03_tickets_modeling/03_tuning_checkpoint.json` | Restart-safe tuning checkpoint. |
| 03 | GENERATED/REUSED | `cache/03_tickets_modeling/03_final_oof_predictions.parquet` | OOF predictions. |
| 03 | GENERATED/REUSED | `cache/03_tickets_modeling/03_final_oof_metadata.json` | OOF provenance. |
| 03 | OUTPUT | `reports/tables/03_tickets_modeling/03_modeling_design_summary.csv` | Design summary. |
| 03 | OUTPUT | `reports/tables/03_tickets_modeling/03_modeling_search_candidates.csv` | Tuning candidates. |
| 03 | OUTPUT | `reports/tables/03_tickets_modeling/03_modeling_search_summary.csv` | Tuning summary. |
| 03 | OUTPUT | `reports/tables/03_tickets_modeling/03_modeling_cv_results.csv` | Final CV results. |
| 03 | OUTPUT | `reports/tables/03_tickets_modeling/03_modeling_cv_class_results.csv` | Class-level CV results. |
| 03 | OUTPUT | `reports/tables/03_tickets_modeling/03_modeling_feature_set_selection.csv` | Common feature-set selection. |
| 03 | OUTPUT | `reports/tables/03_tickets_modeling/03_modeling_dummy_baseline.csv` | Technical dummy baseline. |
| 03 | OUTPUT | `reports/tables/03_tickets_modeling/03_modeling_model_selection.csv` | Selected model roles/artifacts. |
| 03 | OUTPUT | `reports/tables/03_tickets_modeling/03_bilingual_stopwords.txt` | Exact stopword list used by final Modeling. |
| 03 | OUTPUT | `reports/tables/03_tickets_modeling/03_modeling_manifest.json` | Central Modeling manifest. |
| 03 | OUTPUT | `models/03_tickets_modeling/03_fitted_model_metadata.json` | Fitted-model metadata. |
| 03 | OUTPUT | `models/03_tickets_modeling/03_model_<feature_set>_<model>.joblib` | Required fitted model artifacts for selected roles. |
| 03 | OUTPUT | `reports/figures/03_tickets_modeling/03_cv_macro_f1_matrix.png` | CV figure. |
| 03 | OUTPUT | `reports/figures/03_tickets_modeling/03_cv_class_f1_best_feature_set.png` | Class-F1 figure. |
| 04 | INPUT | `data/processed/customer_it_support_prepared.csv` | Prepared dataset. |
| 04 | INPUT | `reports/tables/03_tickets_modeling/03_modeling_manifest.json` | Modeling provenance. |
| 04 | INPUT | `reports/tables/03_tickets_modeling/03_modeling_model_selection.csv` | Selected pipelines. |
| 04 | INPUT | `reports/tables/03_tickets_modeling/03_modeling_feature_set_selection.csv` | Common feature set. |
| 04 | INPUT | `models/03_tickets_modeling/03_fitted_model_metadata.json` | Model hashes/metadata. |
| 04 | INPUT | `models/03_tickets_modeling/03_model_<feature_set>_<model>.joblib` | Fitted models referenced by selection table. |
| 04 | INPUT | `cache/03_tickets_modeling/03_split_assignments.parquet` | Fixed hold-out split. |
| 04 | INPUT | `cache/03_tickets_modeling/03_sentiment_features.csv` | Sentiment features when required. |
| 04 | INPUT | `cache/03_tickets_modeling/03_sentiment_features_metadata.json` | Sentiment provenance. |
| 04 | OUTPUT | `cache/04_evaluation/04_holdout_predictions.parquet` | Frozen hold-out predictions. |
| 04 | OUTPUT | `cache/04_evaluation/04_holdout_predictions_metadata.json` | Prediction provenance. |
| 04 | OUTPUT | `cache/04_evaluation/04_bootstrap_distributions.parquet` | Paired group-aware bootstrap distributions. |
| 04 | OUTPUT | `cache/04_evaluation/04_bootstrap_metadata.json` | Bootstrap provenance. |
| 04 | OUTPUT | `reports/tables/04_evaluation/04_evaluation_holdout_results.csv` | Hold-out metrics. |
| 04 | OUTPUT | `reports/tables/04_evaluation/04_evaluation_holdout_class_results.csv` | Class-level hold-out metrics. |
| 04 | OUTPUT | `reports/tables/04_evaluation/04_evaluation_h1_pairwise_results.csv` | H1 pairwise tests. |
| 04 | OUTPUT | `reports/tables/04_evaluation/04_evaluation_final_model_intervals.csv` | Final-model intervals. |
| 04 | OUTPUT | `reports/tables/04_evaluation/04_evaluation_manifest.json` | Evaluation manifest. |
| 04 | OUTPUT | `reports/figures/04_evaluation/04_common_feature_set_holdout_macro_f1.png` | Hold-out comparison figure. |
| 04 | OUTPUT | `reports/figures/04_evaluation/04_H1_pairwise_macro_f1_differences.png` | H1 effect figure. |
| 04 | OUTPUT | `reports/figures/04_evaluation/04_final_model_confusion_matrix.png` | Confusion matrix. |
| 05 | INPUT | `data/processed/customer_it_support_prepared.csv` | Prepared dataset. |
| 05 | INPUT | `reports/tables/03_tickets_modeling/03_modeling_manifest.json` | Modeling provenance. |
| 05 | INPUT | `reports/tables/03_tickets_modeling/03_modeling_model_selection.csv` | Model artifacts. |
| 05 | INPUT | `reports/tables/03_tickets_modeling/03_modeling_feature_set_selection.csv` | Common feature set. |
| 05 | INPUT | `models/03_tickets_modeling/03_fitted_model_metadata.json` | Model metadata/hashes. |
| 05 | INPUT | `models/03_tickets_modeling/03_model_<feature_set>_<model>.joblib` | Frozen models. |
| 05 | INPUT | `cache/03_tickets_modeling/03_split_assignments.parquet` | Fixed training/hold-out split. |
| 05 | INPUT | `cache/03_tickets_modeling/03_sentiment_features.csv` | Sentiment features. |
| 05 | INPUT | `cache/03_tickets_modeling/03_sentiment_features_metadata.json` | Sentiment provenance. |
| 05 | INPUT | `reports/tables/04_evaluation/04_evaluation_manifest.json` | Evaluation provenance. |
| 05 | INPUT | `cache/04_evaluation/04_holdout_predictions.parquet` | Frozen hold-out predictions. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_xai_sample.csv` | Shared H2/local XAI sample. |
| 05 | OUTPUT | `cache/05_explainability/05_linear_shap_background_sample.parquet` | Linear/CNB background. |
| 05 | OUTPUT | `cache/05_explainability/05_knn_permutation_background_sample.parquet` | kNN background. |
| 05 | OUTPUT | `cache/05_explainability/05_shap_compactness_by_case.parquet` | Case-level compactness. |
| 05 | OUTPUT | `cache/05_explainability/05_faithfulness_by_case.parquet` | Case-level faithfulness. |
| 05 | OUTPUT | `cache/05_explainability/05_xgboost_shap_sensitivity_by_case.parquet` | XGBoost exact/approx sensitivity cases. |
| 05 | OUTPUT | `cache/05_explainability/models/05_<feature_set>_<model>_results.joblib` | Per-model validated SHAP caches. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_shap_compactness_summary.csv` | Compactness summary. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_h2_pairwise_results.csv` | H2 primary tests. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_h2_sensitivity_results.csv` | H2 metric sensitivity checks. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_faithfulness_summary.csv` | Faithfulness summary. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_xgboost_shap_sensitivity_summary.csv` | XGBoost approximation sensitivity. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_global_top_features.csv` | Global feature explanations. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_feature_group_summary.csv` | Feature-group explanation shares. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_local_explanation_cases.csv` | Selected local cases. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_local_feature_contributions.csv` | Local SHAP contributions. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_qualitative_plausibility_assessment_template.csv` | Author plausibility template. |
| 05 | FROZEN HUMAN ASSESSMENT | `reports/tables/05_explainability/05_qualitative_plausibility_assessment_completed.csv` | Author-completed qualitative plausibility assessment for the 12 selected local cases across seven classifiers; retained if reported in the thesis. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_shap_validation_summary.csv` | Explainer/additivity validation. |
| 05 | OUTPUT | `reports/tables/05_explainability/05_explainability_manifest.json` | Explainability manifest. |
| 05 | OUTPUT | `reports/figures/05_explainability/05_h2_compactness_by_model.png` | H2 compactness figure. |
| 05 | OUTPUT | `reports/figures/05_explainability/05_h2_pairwise_differences.png` | H2 effect figure. |
| 05 | OUTPUT | `reports/figures/05_explainability/05_feature_group_shares_by_model.png` | Feature-group figure. |
| 05 | OUTPUT | `reports/figures/05_explainability/05_top_features_<final-model>_<class>.png` | Final-model class feature figures. |
| 05a | INPUT | `data/processed/customer_it_support_prepared.csv` | Prepared dataset. |
| 05a | INPUT | `reports/tables/03_tickets_modeling/03_modeling_model_selection.csv` | Locates the frozen kNN artifact. |
| 05a | INPUT | `models/03_tickets_modeling/03_model_<common-feature-set>_knn_cosine.joblib` | Frozen kNN model. |
| 05a | INPUT | `reports/tables/05_explainability/05_xai_sample.csv` | Shared 05 global H2 population. |
| 05a | INPUT | `reports/tables/05_explainability/05_explainability_manifest.json` | 05 provenance and expected hashes. |
| 05a | INPUT | `cache/05_explainability/05_linear_shap_background_sample.parquet` | Training background pool. |
| 05a | INPUT | `cache/05_explainability/05_knn_permutation_background_sample.parquet` | Current 05 kNN reference background. |
| 05a | GENERATED/REUSED | `cache/05_explainability/05a_knn_sensitivity/05a_reference.joblib` | 3/3 reference sensitivity cache. |
| 05a | GENERATED/REUSED | `cache/05_explainability/05a_knn_sensitivity/05a_reduced_background.joblib` | 1/3 sensitivity cache. |
| 05a | GENERATED/REUSED | `cache/05_explainability/05a_knn_sensitivity/05a_reduced_cycles.joblib` | 3/1 sensitivity cache. |
| 05a | OUTPUT | `reports/tables/05_explainability/05a_knn_shap_sensitivity_sample.csv` | Balanced 90-case sensitivity sample. |
| 05a | OUTPUT | `reports/tables/05_explainability/05a_knn_shap_sensitivity_summary.csv` | Configuration summary. |
| 05a | OUTPUT | `reports/tables/05_explainability/05a_knn_shap_sensitivity_comparisons.csv` | Sensitivity comparisons. |
| 05a | OUTPUT | `reports/tables/05_explainability/05a_knn_shap_sensitivity_manifest.json` | Sensitivity provenance manifest. |
| 06 | INPUT | `data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv` | Raw source identity check. |
| 06 | INPUT | `data/processed/customer_it_support_prepared.csv` | Prepared-data integrity check. |
| 06 | INPUT | `reports/tables/03_tickets_modeling/* required final Modeling artifacts` | Modeling regression/integrity checks. |
| 06 | INPUT | `models/03_tickets_modeling/03_fitted_model_metadata.json` | Model metadata. |
| 06 | INPUT | `cache/03_tickets_modeling/03_split_assignments.parquet` | Split integrity. |
| 06 | INPUT | `reports/tables/04_evaluation/* required Evaluation artifacts` | Evaluation/H1 checks. |
| 06 | INPUT | `cache/04_evaluation/04_holdout_predictions.parquet` | Prediction integrity. |
| 06 | INPUT | `reports/tables/05_explainability/* required 05 and 05a artifacts` | Explainability/H2 and kNN sensitivity checks. |
| 06 | INPUT | `reports/figures/04_evaluation/* central figures` | Figure existence checks. |
| 06 | INPUT | `reports/figures/05_explainability/* central figures` | Figure existence checks. |
| 06 | OUTPUT | `reports/tables/06_final_integrity_audit/06_integrity_audit.csv` | PASS/FAIL audit table. |
| 06 | OUTPUT | `reports/tables/06_final_integrity_audit/06_integrity_manifest.json` | Final audit manifest. |
