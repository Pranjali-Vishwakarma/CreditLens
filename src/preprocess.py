"""
preprocess.py — Data ingestion and preprocessing for German Credit Data (Statlog).

Source: UCI ML Repository — Statlog German Credit Data
URL: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
Format: space-separated, no header, 20 attributes + 1 target (1=good, 2=bad risk)

Expected raw file: dataset/raw/german_credit.csv
Expected processed: dataset/processed/train.csv, dataset/processed/test.csv

Column mapping (original coded names → readable names):
  A1  = checking_account_status  (categorical, ordinal)
  A2  = duration_months           (numeric)
  A3  = credit_history            (categorical)
  A4  = purpose                   (categorical)
  A5  = credit_amount             (numeric)
  A6  = savings_account           (categorical, ordinal)
  A7  = employment_since          (categorical, ordinal)
  A8  = installment_rate_pct      (numeric)
  A9  = personal_status_sex       (categorical)
  A10 = other_debtors             (categorical)
  A11 = residence_since           (numeric)
  A12 = property                  (categorical, ordinal)
  A13 = age                       (numeric)
  A14 = other_installment_plans   (categorical)
  A15 = housing                   (categorical)
  A16 = existing_credits_count    (numeric)
  A17 = job                       (categorical, ordinal)
  A18 = num_dependents            (numeric)
  A19 = telephone                 (binary categorical)
  A20 = foreign_worker            (binary categorical)
  A21 = target                    (1=good, 2=bad → remapped to 0=bad, 1=good)

Encoding choice: ordinal encoding for ordered categoricals (checking_account,
savings_account, employment_since, property, job); one-hot for nominal categoricals.
age_band is kept as a plain string column (not encoded) for downstream fairness audit.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
RAW_PATH = _ROOT / "dataset" / "raw" / "german_credit.csv"
TRAIN_PATH = _ROOT / "dataset" / "processed" / "train.csv"
TEST_PATH = _ROOT / "dataset" / "processed" / "test.csv"

_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
)

COLUMN_NAMES = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "target",
]

# Ordinal maps: higher value = more favourable/stable
_ORDINAL_MAPS: dict[str, dict] = {
    "checking_account_status": {"A11": 0, "A12": 1, "A13": 2, "A14": 3},
    "savings_account":         {"A61": 0, "A62": 1, "A63": 2, "A64": 3, "A65": 4},
    "employment_since":        {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4},
    "property":                {"A121": 0, "A122": 1, "A123": 2, "A124": 3},
    "job":                     {"A171": 0, "A172": 1, "A173": 2, "A174": 3},
}

_NOMINAL_COLS = [
    "credit_history",
    "purpose",
    "personal_status_sex",
    "other_debtors",
    "other_installment_plans",
    "housing",
    "telephone",
    "foreign_worker",
]


def load_raw_data(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Load raw German Credit data into a DataFrame.

    Downloads from UCI if the file is missing. The source file is space-separated
    with no header; column names and target remapping are applied here.

    Args:
        path: Path to the raw CSV/data file.

    Returns:
        DataFrame with readable column names. target: 1=good risk, 0=bad risk.
    """
    path = Path(path)
    if not path.exists():
        print(f"Downloading German Credit Data to {path} …")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_DATA_URL, path)

    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    # Remap target: original 1=good, 2=bad → new 1=good, 0=bad
    df["target"] = (df["target"] == 1).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering and encoding.

    - Derives debt_to_income_proxy = credit_amount * installment_rate_pct / 100
    - Buckets age into age_band (kept as plain string for fairness audit)
    - Ordinal-encodes ordered categoricals (see _ORDINAL_MAPS)
    - One-hot-encodes nominal categoricals (see _NOMINAL_COLS)

    Args:
        df: Raw DataFrame from load_raw_data().

    Returns:
        Processed DataFrame ready for splitting.
    """
    df = df.copy()

    # Derived feature
    df["debt_to_income_proxy"] = df["credit_amount"] * df["installment_rate_pct"] / 100.0

    # Age bands — kept as string; DO NOT encode; fairness module reads this column
    bins = [0, 25, 40, 60, np.inf]
    labels = ["<25", "25-40", "40-60", "60+"]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=False).astype(str)

    # Ordinal encoding
    for col, mapping in _ORDINAL_MAPS.items():
        df[col] = df[col].map(mapping).fillna(-1).astype(int)

    # One-hot encoding for nominal columns (drop_first avoids perfect multicollinearity)
    df = pd.get_dummies(df, columns=_NOMINAL_COLS, drop_first=True)

    return df


def split_and_save(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Stratified train/test split; saves results to dataset/processed/.

    age_band is preserved as-is in both output files.

    Args:
        df: Engineered DataFrame from engineer_features().
        test_size: Fraction reserved for the test set (default 0.2).
        random_state: RNG seed for reproducibility (default 42).
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"Saved train ({len(train)}) -> {TRAIN_PATH}")
    print(f"Saved test  ({len(test)})  -> {TEST_PATH}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df), running the full pipeline if needed.

    If processed files already exist, loads them directly without re-running
    the pipeline. This is the entry point used by train.py.

    Returns:
        Tuple of (train DataFrame, test DataFrame).
    """
    if TRAIN_PATH.exists() and TEST_PATH.exists():
        return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)

    raw = load_raw_data()
    engineered = engineer_features(raw)
    split_and_save(engineered)
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


# Exact feature column order the model was trained with (target + age_band excluded).
# Keep in sync if preprocess logic ever changes.
FEATURE_COLUMNS = [
    "checking_account_status", "duration_months", "credit_amount",
    "savings_account", "employment_since", "installment_rate_pct",
    "residence_since", "property", "age", "existing_credits_count",
    "job", "num_dependents", "debt_to_income_proxy",
    "credit_history_A31", "credit_history_A32", "credit_history_A33", "credit_history_A34",
    "purpose_A41", "purpose_A410", "purpose_A42", "purpose_A43", "purpose_A44",
    "purpose_A45", "purpose_A46", "purpose_A48", "purpose_A49",
    "personal_status_sex_A92", "personal_status_sex_A93", "personal_status_sex_A94",
    "other_debtors_A102", "other_debtors_A103",
    "other_installment_plans_A142", "other_installment_plans_A143",
    "housing_A152", "housing_A153",
    "telephone_A192", "foreign_worker_A202",
]

# One-hot columns grouped by source field (reference category dropped during training)
_ONE_HOT_GROUPS: dict[str, list[str]] = {
    "credit_history":          ["A31", "A32", "A33", "A34"],        # ref: A30
    "purpose":                 ["A41", "A410", "A42", "A43", "A44", "A45", "A46", "A48", "A49"],  # ref: A40
    "personal_status_sex":     ["A92", "A93", "A94"],               # ref: A91
    "other_debtors":           ["A102", "A103"],                    # ref: A101
    "other_installment_plans": ["A142", "A143"],                    # ref: A141
    "housing":                 ["A152", "A153"],                    # ref: A151
    "telephone":               ["A192"],                            # ref: A191
    "foreign_worker":          ["A202"],                            # ref: A201
}


def encode_single_applicant(raw: dict) -> pd.DataFrame:
    """Encode one applicant dict into the model's feature DataFrame.

    Applies the same ordinal and one-hot encoding as engineer_features().
    Returns a single-row DataFrame with columns matching FEATURE_COLUMNS exactly.

    Args:
        raw: Dict with keys:
            Numeric : duration_months, credit_amount, installment_rate_pct,
                      residence_since, age, existing_credits_count, num_dependents
            Ordinal : checking_account_status (A11-A14), savings_account (A61-A65),
                      employment_since (A71-A75), property (A121-A124), job (A171-A174)
            One-hot : credit_history (A30-A34), purpose (A40-A410),
                      personal_status_sex (A91-A94), other_debtors (A101-A103),
                      other_installment_plans (A141-A143), housing (A151-A153),
                      telephone (A191-A192), foreign_worker (A201-A202)

    Returns:
        Single-row DataFrame aligned to FEATURE_COLUMNS.
    """
    row: dict = {}

    # Ordinal features
    for col, mapping in _ORDINAL_MAPS.items():
        row[col] = mapping.get(raw[col], -1)

    # Numeric features (pass through directly)
    for col in ["duration_months", "credit_amount", "installment_rate_pct",
                "residence_since", "age", "existing_credits_count", "num_dependents"]:
        row[col] = float(raw[col])

    # Derived feature
    row["debt_to_income_proxy"] = float(raw["credit_amount"]) * float(raw["installment_rate_pct"]) / 100.0

    # One-hot features
    for field, categories in _ONE_HOT_GROUPS.items():
        selected = raw.get(field, "")
        for cat in categories:
            row[f"{field}_{cat}"] = int(selected == cat)

    return pd.DataFrame([row])[FEATURE_COLUMNS]


if __name__ == "__main__":
    train_df, test_df = load_data()
    print("\n=== Train set ===")
    print(f"Shape : {train_df.shape}")
    print(f"Target: {train_df['target'].value_counts().to_dict()}")
    print("\n=== Test set ===")
    print(f"Shape : {test_df.shape}")
    print(f"Target: {test_df['target'].value_counts().to_dict()}")
    print("\nColumn preview:", list(train_df.columns[:8]), "...")

