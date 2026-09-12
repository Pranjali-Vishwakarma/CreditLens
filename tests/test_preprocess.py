"""tests/test_preprocess.py — preprocess module unit tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocess import (
    FEATURE_COLUMNS,
    COLUMN_NAMES,
    RAW_PATH,
    encode_single_applicant,
    load_raw_data,
)

_SAMPLE_RAW = {
    "checking_account_status": "A13",
    "duration_months": 18,
    "credit_amount": 3200.0,
    "savings_account": "A63",
    "employment_since": "A75",
    "installment_rate_pct": 2,
    "residence_since": 3,
    "property": "A121",
    "age": 42,
    "existing_credits_count": 1,
    "job": "A173",
    "num_dependents": 1,
    "credit_history": "A32",
    "purpose": "A43",
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "other_installment_plans": "A143",
    "housing": "A152",
    "telephone": "A192",
    "foreign_worker": "A201",
}


def test_load_raw_data_columns():
    df = load_raw_data()
    assert list(df.columns) == COLUMN_NAMES
    assert len(df) == 1000


def test_load_raw_data_target_remapped():
    df = load_raw_data()
    assert set(df["target"].unique()) == {0, 1}


def test_load_raw_data_no_nulls():
    df = load_raw_data()
    assert df.isnull().sum().sum() == 0


def test_encode_single_applicant_columns():
    row = encode_single_applicant(_SAMPLE_RAW)
    assert list(row.columns) == FEATURE_COLUMNS


def test_encode_single_applicant_shape():
    row = encode_single_applicant(_SAMPLE_RAW)
    assert row.shape == (1, len(FEATURE_COLUMNS))


def test_encode_single_applicant_numeric():
    row = encode_single_applicant(_SAMPLE_RAW)
    assert row.select_dtypes(exclude="number").empty


def test_encode_single_applicant_debt_proxy():
    row = encode_single_applicant(_SAMPLE_RAW)
    expected = 3200.0 * 2 / 100.0
    assert abs(row["debt_to_income_proxy"].iloc[0] - expected) < 1e-6
