"""tests/test_fairness.py — fairness audit structure and threshold logic tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fairness import check_disparate_impact, run_fairness_audit

_EXPECTED_BANDS = {"<25", "25-40", "40-60", "60+"}


@pytest.fixture(scope="module")
def audit():
    return run_fairness_audit()


def test_audit_keys(audit):
    assert {"group_metrics", "disparate_impact_results", "summary_text"} <= audit.keys()


def test_audit_group_metrics_type(audit):
    assert isinstance(audit["group_metrics"], pd.DataFrame)


def test_audit_age_bands_present(audit):
    bands = set(audit["group_metrics"].index.astype(str))
    assert _EXPECTED_BANDS == bands


def test_audit_summary_is_string(audit):
    assert isinstance(audit["summary_text"], str)
    assert len(audit["summary_text"]) > 0


def test_audit_disparate_impact_keys(audit):
    di = audit["disparate_impact_results"]
    for group, info in di.items():
        assert {"age_band", "approval_rate", "ratio_to_reference", "disparate_impact_flag"} <= info.keys()


# Synthetic test: isolate check_disparate_impact threshold logic from real noisy data
def test_check_disparate_impact_flags_correctly():
    # Construct a synthetic group_metrics with obvious 0.5 ratio
    synthetic = pd.DataFrame(
        {"selection_rate": [0.8, 0.4, 0.6]},
        index=["group_A", "group_B", "group_C"],
    )
    result = check_disparate_impact(synthetic, threshold=0.8)

    # group_A is reference (highest at 0.8)
    assert result["group_A"]["ratio_to_reference"] == pytest.approx(1.0)
    assert result["group_A"]["disparate_impact_flag"] is False

    # group_B: 0.4 / 0.8 = 0.5 < 0.8 -> flagged
    assert result["group_B"]["ratio_to_reference"] == pytest.approx(0.5)
    assert result["group_B"]["disparate_impact_flag"] is True

    # group_C: 0.6 / 0.8 = 0.75 < 0.8 -> flagged
    assert result["group_C"]["ratio_to_reference"] == pytest.approx(0.75)
    assert result["group_C"]["disparate_impact_flag"] is True


def test_check_disparate_impact_no_flags():
    # All groups within 80% of reference
    synthetic = pd.DataFrame(
        {"selection_rate": [0.9, 0.85, 0.8]},
        index=["A", "B", "C"],
    )
    result = check_disparate_impact(synthetic, threshold=0.8)
    assert not any(v["disparate_impact_flag"] for v in result.values())
