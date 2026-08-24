# Dataset License and Attribution

This repository uses and may redistribute the following dataset:

**Customer IT Support - Ticket Dataset**  
Creator: **Tobias Bueck**  
Source: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets  
License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**  
License information: https://creativecommons.org/licenses/by/4.0/

## Attribution

The dataset is attributed to Tobias Bueck and is redistributed under the
terms of CC BY 4.0.

## Modifications

The file in `data/raw/` represents the source dataset used for this thesis.

The file in `data/processed/` is a derived/modified version created by the
analysis pipeline in this repository. Transformations include documented
record exclusions, text normalization, language-label corrections, exact-text
grouping, semantic grouping, and derived metadata. The complete transformation
logic is contained in the scripts under `src/`, primarily
`01\\\_data\\\_understanding.py`, `01a\\\_manual\\\_review\\\_validation.py`, and
`02\\\_data\\\_preparation.py`.

The original dataset license remains applicable to the source material.
The repository's MIT code license does not replace or override CC BY 4.0 for
the dataset or its licensed source content.

