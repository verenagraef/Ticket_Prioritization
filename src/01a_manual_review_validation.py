"""
Validate and consolidate the three author-completed Data Understanding reviews.

Run this script only after:
1. 01_data_understanding.py has completed successfully and
2. the three generated review templates have been saved as
   *_AUTHOR_CONFIRMED.csv in the same Data Understanding table directory.
"""

# %% 00 - Load packages

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# %% 01 - Configure manual-review validation

SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name.lower() != "src":
    raise RuntimeError(
        "Place 01a_manual_review_validation.py in <project>/src before running it."
    )

PROJECT_ROOT = SCRIPT_PATH.parents[1]

DU_TABLES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "01_tickets_data_understanding"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "01a_manual_review_validation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DU_MANIFEST_PATH = DU_TABLES_DIR / "01_data_understanding_manifest.json"

SEMANTIC_TEMPLATE_PATH = DU_TABLES_DIR / "semantic_pair_review_sample.csv"
LANGUAGE_TEMPLATE_PATH = DU_TABLES_DIR / "language_label_manual_review_sample.csv"
PRIORITY_TEMPLATE_PATH = DU_TABLES_DIR / "priority_label_plausibility_review_sample.csv"

SEMANTIC_CONFIRMED_PATH = (
    DU_TABLES_DIR / "semantic_pair_review_sample_AUTHOR_CONFIRMED.csv"
)
LANGUAGE_CONFIRMED_PATH = (
    DU_TABLES_DIR / "language_label_manual_review_sample_AUTHOR_CONFIRMED.csv"
)
PRIORITY_CONFIRMED_PATH = (
    DU_TABLES_DIR / "priority_label_plausibility_review_sample_AUTHOR_CONFIRMED.csv"
)

VALIDATED_LANGUAGE_PATH = OUTPUT_DIR / "01a_validated_language_decisions.csv"
SUMMARY_PATH = OUTPUT_DIR / "01a_manual_review_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "01a_manual_review_manifest.json"

REVIEW_SAMPLE_SIZES = {
    "semantic": 60,
    "language": 60,
    "priority": 30,
}
SEMANTIC_SIMILARITY_THRESHOLD = 0.95
TARGET_LANGUAGES = {"de", "en"}


# %% 02 - Define validation helpers

def fmt_int(value):
    return f"{int(value):,}".replace(",", ".")


def sha256_file(path, block_size=1_048_576):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload, path):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_confirmed_review(path, required_decisions, expected_records):
    review = pd.read_csv(path, encoding="utf-8-sig")

    required = [
        *required_decisions,
        "evaluator_role",
        "author_confirmation",
    ]
    missing = sorted(set(required) - set(review.columns))
    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")

    if len(review) != expected_records:
        raise ValueError(
            f"{path.name} contains {len(review)} records; "
            f"expected {expected_records}."
        )

    if review[required].isna().any().any():
        raise ValueError(f"{path.name} contains incomplete review decisions.")

    if not review["evaluator_role"].astype(str).str.strip().eq("Author").all():
        raise ValueError(f"{path.name} is not marked as Author for every record.")

    if not (
        review["author_confirmation"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("confirmed")
        .all()
    ):
        raise ValueError(f"{path.name} is not fully author-confirmed.")

    return review


def validate_template_hash(manifest, review_name, template_path):
    expected = (
        manifest
        .get("review_templates", {})
        .get(review_name, {})
        .get("sha256")
    )
    observed = sha256_file(template_path)
    if expected != observed:
        raise ValueError(
            f"{template_path.name} differs from the template generated "
            "by the validated 01 run."
        )


def semantic_similarity_band(values):
    numeric = pd.to_numeric(values, errors="raise")
    return pd.cut(
        numeric,
        bins=[0.90, 0.95, 0.98, 1.01],
        right=False,
        labels=["0.90-<0.95", "0.95-<0.98", "0.98-1.00"],
        include_lowest=True,
    )


def validate_provenance(
    template,
    confirmed,
    key_columns,
    label,
    float_columns=(),
):
    missing_template = sorted(set(key_columns) - set(template.columns))
    missing_confirmed = sorted(set(key_columns) - set(confirmed.columns))
    if missing_template or missing_confirmed:
        raise KeyError(
            f"{label} provenance columns missing. "
            f"Template: {missing_template}; confirmed: {missing_confirmed}."
        )

    if len(template) != len(confirmed):
        raise ValueError(f"{label} confirmed file has a different sample size.")

    left = template[key_columns].reset_index(drop=True).copy()
    right = confirmed[key_columns].reset_index(drop=True).copy()

    for column in key_columns:
        if column in float_columns:
            a = pd.to_numeric(left[column], errors="raise").to_numpy(dtype=float)
            b = pd.to_numeric(right[column], errors="raise").to_numpy(dtype=float)
            if not np.allclose(a, b, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"{label} differs from the generated sample in {column}."
                )
        elif column in {"review_id", "pair_id", "source_row_id"}:
            a = pd.to_numeric(left[column], errors="raise").astype("int64")
            b = pd.to_numeric(right[column], errors="raise").astype("int64")
            if not a.equals(b):
                raise ValueError(
                    f"{label} differs from the generated sample in {column}."
                )
        else:
            a = left[column].fillna("").astype(str)
            b = right[column].fillna("").astype(str)
            if not a.equals(b):
                raise ValueError(
                    f"{label} differs from the generated sample in {column}."
                )


# %% 03 - Validate required inputs and 01 provenance

required_paths = [
    DU_MANIFEST_PATH,
    SEMANTIC_TEMPLATE_PATH,
    LANGUAGE_TEMPLATE_PATH,
    PRIORITY_TEMPLATE_PATH,
    SEMANTIC_CONFIRMED_PATH,
    LANGUAGE_CONFIRMED_PATH,
    PRIORITY_CONFIRMED_PATH,
]
missing_paths = [path for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Manual-review validation inputs are missing:\n"
        + "\n".join(str(path) for path in missing_paths)
    )

du_manifest = load_json(DU_MANIFEST_PATH)

validate_template_hash(
    du_manifest,
    "semantic",
    SEMANTIC_TEMPLATE_PATH,
)
validate_template_hash(
    du_manifest,
    "language",
    LANGUAGE_TEMPLATE_PATH,
)
validate_template_hash(
    du_manifest,
    "priority",
    PRIORITY_TEMPLATE_PATH,
)

semantic_template = pd.read_csv(
    SEMANTIC_TEMPLATE_PATH,
    encoding="utf-8-sig",
)
language_template = pd.read_csv(
    LANGUAGE_TEMPLATE_PATH,
    encoding="utf-8-sig",
)
priority_template = pd.read_csv(
    PRIORITY_TEMPLATE_PATH,
    encoding="utf-8-sig",
)

semantic_review = load_confirmed_review(
    SEMANTIC_CONFIRMED_PATH,
    ["review_decision"],
    REVIEW_SAMPLE_SIZES["semantic"],
)
language_review = load_confirmed_review(
    LANGUAGE_CONFIRMED_PATH,
    ["manual_language", "manual_assessment"],
    REVIEW_SAMPLE_SIZES["language"],
)
priority_review = load_confirmed_review(
    PRIORITY_CONFIRMED_PATH,
    ["priority_assessment", "missing_context_required"],
    REVIEW_SAMPLE_SIZES["priority"],
)

validate_provenance(
    semantic_template,
    semantic_review,
    ["pair_id", "text_a", "text_b"],
    "Semantic review",
)
validate_provenance(
    language_template,
    language_review,
    [
        "review_id",
        "language_review_group",
        "source_row_id",
        "version",
        "declared_language",
        "detected_language",
        "text",
    ],
    "Language review",
)
validate_provenance(
    priority_template,
    priority_review,
    [
        "review_id",
        "source_row_id",
        "priority",
        "declared_language",
        "version",
        "type",
        "queue",
        "text",
    ],
    "Priority review",
)

template_semantic_bands = semantic_similarity_band(
    semantic_template["cosine_similarity"]
)
confirmed_semantic_bands = semantic_similarity_band(
    semantic_review["cosine_similarity"]
)

if template_semantic_bands.isna().any() or confirmed_semantic_bands.isna().any():
    raise ValueError(
        "At least one semantic similarity lies outside the deterministic "
        "review range 0.90-1.00."
    )

if not template_semantic_bands.astype(str).equals(
    confirmed_semantic_bands.astype(str)
):
    raise ValueError(
        "At least one semantic review pair moved to a different similarity "
        "band. The review is therefore not provenance-compatible."
    )

semantic_similarity_difference = np.abs(
    pd.to_numeric(
        semantic_template["cosine_similarity"],
        errors="raise",
    ).to_numpy(dtype=float)
    - pd.to_numeric(
        semantic_review["cosine_similarity"],
        errors="raise",
    ).to_numpy(dtype=float)
)
max_semantic_similarity_difference = float(
    semantic_similarity_difference.max()
)

print("\nManual review provenance")
print("------------------------")
print("01 template hashes        : validated")
print("Semantic pair identity    : validated")
print("Semantic similarity bands : validated")
print(
    f"Max similarity difference : "
    f"{max_semantic_similarity_difference:.8f}"
)
print("Semantic confirmed file   : validated")
print("Language confirmed file   : validated")
print("Priority confirmed file   : validated")


# %% 04 - Evaluate the three manual reviews

semantic_above_threshold = semantic_review.loc[
    pd.to_numeric(
        semantic_review["cosine_similarity"],
        errors="raise",
    ).ge(SEMANTIC_SIMILARITY_THRESHOLD)
].copy()

semantic_equivalent = (
    semantic_above_threshold["review_decision"]
    .astype(str)
    .str.strip()
    .eq("equivalent_template_variant")
)
semantic_equivalent_share = float(semantic_equivalent.mean())

priority_not_observable = int(
    priority_review["priority_assessment"]
    .astype(str)
    .str.strip()
    .eq("not_observable_from_text")
    .sum()
)
priority_implausible = int(
    priority_review["priority_assessment"]
    .astype(str)
    .str.strip()
    .eq("implausible")
    .sum()
)

manual_language = (
    language_review["manual_language"]
    .astype(str)
    .str.strip()
    .str.lower()
)

language_review = language_review.copy()
language_review["manual_language"] = manual_language

validated_language = language_review[
    [
        "source_row_id",
        "manual_language",
        "manual_assessment",
        "review_comment",
    ]
].copy()
validated_language["source_row_id"] = pd.to_numeric(
    validated_language["source_row_id"],
    errors="raise",
).astype("int64")
validated_language.to_csv(
    VALIDATED_LANGUAGE_PATH,
    index=False,
    encoding="utf-8-sig",
)

manual_target_count = int(manual_language.isin(TARGET_LANGUAGES).sum())
manual_non_target_count = int(
    (~manual_language.isin(TARGET_LANGUAGES | {"uncertain"})).sum()
)
manual_uncertain_count = int(manual_language.eq("uncertain").sum())

summary_rows = [
    {
        "review": "semantic",
        "indicator": "review_records",
        "value": len(semantic_review),
    },
    {
        "review": "semantic",
        "indicator": "pairs_at_or_ab_0_95",
        "value": len(semantic_above_threshold),
    },
    {
        "review": "semantic",
        "indicator": "equivalent_pairs_at_or_ab_0_95",
        "value": int(semantic_equivalent.sum()),
    },
    {
        "review": "semantic",
        "indicator": "equivalent_share_at_or_ab_0_95",
        "value": semantic_equivalent_share,
    },
    {
        "review": "language",
        "indicator": "review_records",
        "value": len(language_review),
    },
    {
        "review": "language",
        "indicator": "manual_target_language",
        "value": manual_target_count,
    },
    {
        "review": "language",
        "indicator": "manual_non_target_language",
        "value": manual_non_target_count,
    },
    {
        "review": "language",
        "indicator": "manual_uncertain_language",
        "value": manual_uncertain_count,
    },
    {
        "review": "priority",
        "indicator": "review_records",
        "value": len(priority_review),
    },
    {
        "review": "priority",
        "indicator": "not_observable_from_text",
        "value": priority_not_observable,
    },
    {
        "review": "priority",
        "indicator": "implausible",
        "value": priority_implausible,
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\nManual review results")
print("---------------------")
print(f"Semantic review records   : {fmt_int(len(semantic_review))}")
print(
    f"Equivalent share >=0.95   : "
    f"{semantic_equivalent_share:.4f}"
)
print(f"Language review records   : {fmt_int(len(language_review))}")
print(f"Manual DE/EN decisions    : {fmt_int(manual_target_count)}")
print(f"Manual non-DE/EN decisions: {fmt_int(manual_non_target_count)}")
print(f"Priority review records   : {fmt_int(len(priority_review))}")
print(f"Priority not observable   : {fmt_int(priority_not_observable)}")
print(f"Priority implausible       : {fmt_int(priority_implausible)}")


# %% 05 - Save validation manifest

output_hashes = {
    "validated_language_decisions": sha256_file(VALIDATED_LANGUAGE_PATH),
    "manual_review_summary": sha256_file(SUMMARY_PATH),
}

save_json(
    {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": SCRIPT_PATH.name,
        "data_understanding_manifest_sha256": sha256_file(DU_MANIFEST_PATH),
        "template_hashes": {
            "semantic": sha256_file(SEMANTIC_TEMPLATE_PATH),
            "language": sha256_file(LANGUAGE_TEMPLATE_PATH),
            "priority": sha256_file(PRIORITY_TEMPLATE_PATH),
        },
        "confirmed_review_hashes": {
            "semantic": sha256_file(SEMANTIC_CONFIRMED_PATH),
            "language": sha256_file(LANGUAGE_CONFIRMED_PATH),
            "priority": sha256_file(PRIORITY_CONFIRMED_PATH),
        },
        "review_counts": {
            "semantic": len(semantic_review),
            "language": len(language_review),
            "priority": len(priority_review),
        },
        "semantic_similarity_threshold": SEMANTIC_SIMILARITY_THRESHOLD,
        "semantic_similarity_bands_validated": True,
        "max_semantic_similarity_difference": max_semantic_similarity_difference,
        "semantic_equivalent_share": semantic_equivalent_share,
        "priority_not_observable": priority_not_observable,
        "priority_implausible": priority_implausible,
        "output_hashes": output_hashes,
        "all_reviews_validated": True,
    },
    MANIFEST_PATH,
)

print("\nManual review validation completed")
print("----------------------------------")
