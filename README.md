# Automatic Prioritization of IT Support Tickets Using Natural Language Processing and Explainable AI: A Data-Driven Evaluation of Machine Learning Classifiers Based on the CRISP-DM Model

**Author:** Verena Gräf

This repository contains the analysis pipeline and final artifacts for the associated paper.

## Repository structure

```text
<project-root>/
├─ src/                         # analysis scripts
├─ data/
│  ├─ raw/                     # source dataset (CC BY 4.0)
│  └─ processed/               # prepared dataset
├─ reports/
│  ├─ tables/                  # results, manifests, review files
│  └─ figures/                 # figures
├─ cache/                       # reproducibility/intermediate artifacts
├─ models/                      # fitted models
├─ paper/                       # LaTeX manuscript files
├─ environment.yml             # Python environment
├─ REPRODUCIBILITY_FILE_MAP.md # input/output map
├─ CITATION.cff                # citation metadata
├─ LICENSE                     # MIT license for repository code and generated analysis artifacts
├─ DATASET_LICENSE.md          # dataset attribution and License
├─ .gitattributes		        # files managed via Git LFS
└─ README.md
```

## Dataset

The source dataset is included at:

```text
data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv
```

Dataset:

**Tobias Bueck (2025). Customer IT Support - Ticket Dataset [Dataset]. Kaggle.**  
DOI: `10.34740/KAGGLE/DSV/13959267`

The dataset is redistributed under **CC BY 4.0**. See `DATASET_LICENSE.md`.

## Environment

The analysis was validated on **Windows x86-64** with **Python 3.11.15**.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate masterthesis_final
```

## Execution order

Run the scripts in this order:

```text
01_data_understanding.py
01a_manual_review_validation.py
02_data_preparation.py
03a_algorithm_screening_explorativ.py
03b_representation_screening_explorativ.py
03_modeling.py
04_evaluation.py
05_explainability.py
05a_knn_shap_sensitivity.py
06_integrity_audit.py
```
## Required files

These three author-confirmed files are required for a clean reproduction:

```text
reports/tables/01_tickets_data_understanding/semantic_pair_review_sample_AUTHOR_CONFIRMED.csv
reports/tables/01_tickets_data_understanding/language_label_manual_review_sample_AUTHOR_CONFIRMED.csv
reports/tables/01_tickets_data_understanding/priority_label_plausibility_review_sample_AUTHOR_CONFIRMED.csv
```

The completed qualitative XAI assessment is retained as a separate author assessment:

```text
reports/tables/05_explainability/05_qualitative_plausibility_assessment_completed.csv
```

It is not required to execute the automated pipeline.

## Reproduction

For a clean reproduction, keep:

1. `src/`
2. `environment.yml`
3. the raw dataset in `data/raw/`
4. the three `*_AUTHOR_CONFIRMED.csv` files

Then run the scripts in the order shown above.

## Git LFS

Large model and cache artifacts are stored using Git Large File Storage (Git LFS).

For a complete Git clone including these artifacts, install Git LFS before cloning.

## Licensing

- **Original repository code and generated analysis artifacts:** MIT License (`LICENSE`)
- **Customer IT Support - Ticket Dataset by Tobias Bueck:** CC BY 4.0 (`DATASET_LICENSE.md`)
- **Paper/manuscript:** not covered by the MIT software license. If published by JAIR, the article will be distributed under JAIR's Creative Commons Attribution (CC BY) publication license.

