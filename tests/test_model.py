"""tests/test_model.py — model load, predict, and decision threshold tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from explain import load_model
from train import decision_from_probability

_ROOT = Path(__file__).parent.parent
_TEST_CSV = _ROOT / "dataset" / "processed" / "test.csv"
_NON_FEATURE = {"target", "age_band"}


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def test_row(model):
    df = pd.read_csv(_TEST_CSV)
    X = df.drop(columns=[c for c in _NON_FEATURE if c in df.columns])
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype("int8")
    return X.iloc[[0]]


def test_model_loads(model):
    assert model is not None


def test_predict_proba_range(model, test_row):
    proba = model.predict_proba(test_row)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 0] <= 1.0
    assert 0.0 <= proba[0, 1] <= 1.0
    assert abs(proba[0].sum() - 1.0) < 1e-6


def test_decision_approve():
    assert decision_from_probability(0.1) == "Approve"


def test_decision_refer():
    assert decision_from_probability(0.4) == "Refer"


def test_decision_reject():
    assert decision_from_probability(0.7) == "Reject"


def test_decision_boundary_refer_low():
    # Exactly at threshold_refer=0.3 -> Refer (not Approve)
    assert decision_from_probability(0.3) == "Refer"


def test_decision_boundary_reject():
    # Exactly at threshold_reject=0.5 -> Reject (not Refer)
    assert decision_from_probability(0.5) == "Reject"
