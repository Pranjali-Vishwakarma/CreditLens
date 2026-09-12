"""
fairness.py — Bias audit for CreditLens using fairlearn.

Sensitive feature: age_band ("<25", "25-40", "40-60", "60+")
Positive/favorable outcome: "Approve"

Simplifying assumption (documented):
  Refer + Reject -> "not approved" (binary favorable/unfavorable for fairness metrics).
  This follows common practice in consumer credit audits.

Fairness convention:
  Four-fifths (80%) rule: a group has disparate impact if its approval rate
  is less than 0.8 * the highest-approval-rate group's rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate

sys.path.insert(0, str(Path(__file__).parent))
from train import decision_from_probability  # noqa: E402

_ROOT = Path(__file__).parent.parent
_MODEL_PATH = _ROOT / "models" / "credit_model.pkl"
_TEST_PATH = _ROOT / "dataset" / "processed" / "test_with_agebands.csv"

_NON_FEATURE_COLS = {"target", "age_band"}


def _load_test_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load test set; return (X_test, y_true, age_band_series)."""
    df = pd.read_csv(_TEST_PATH)
    y_true = df["target"]
    age_band = df["age_band"]
    X = df.drop(columns=[c for c in _NON_FEATURE_COLS if c in df.columns])
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype("int8")
    return X, y_true, age_band


def _predict_favorable(model, X: pd.DataFrame) -> pd.Series:
    """Return 1 (Approve) / 0 (Refer or Reject) for each row.

    Uses decision_from_probability so thresholds match the rest of the app.
    """
    prob_default = model.predict_proba(X)[:, 0]
    decisions = [decision_from_probability(p) for p in prob_default]
    return pd.Series([1 if d == "Approve" else 0 for d in decisions], index=X.index)


def _false_rejection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rate of 'not approved' among actually-good-credit applicants (target==1)."""
    good = y_true == 1
    if good.sum() == 0:
        return float("nan")
    return float((y_pred[good] == 0).mean())


def compute_group_metrics(
    y_true: pd.Series,
    y_pred_favorable: pd.Series,
    sensitive_feature: pd.Series,
) -> pd.DataFrame:
    """Compute per-age_band approval rate, false rejection rate, and group size.

    Args:
        y_true: Ground-truth labels (1=good, 0=bad).
        y_pred_favorable: Binary series — 1 if Approved, 0 otherwise.
        sensitive_feature: age_band series aligned with y_true.

    Returns:
        DataFrame indexed by age_band with columns:
        selection_rate, false_rejection_rate, group_size.
    """
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "false_rejection_rate": _false_rejection_rate,
        },
        y_true=y_true.values,
        y_pred=y_pred_favorable.values,
        sensitive_features=sensitive_feature.values,
    )
    result = mf.by_group.copy()

    # Add group sizes
    sizes = sensitive_feature.value_counts()
    result["group_size"] = sizes
    return result


def check_disparate_impact(
    group_metrics: pd.DataFrame,
    approval_col: str = "selection_rate",
    threshold: float = 0.8,
) -> dict:
    """Apply the four-fifths rule; return per-group impact flags.

    The group with the highest approval rate is the reference.

    Args:
        group_metrics: Output of compute_group_metrics().
        approval_col: Column name holding approval/selection rates.
        threshold: Four-fifths cutoff (default 0.8).

    Returns:
        Dict keyed by age_band with keys:
        approval_rate, ratio_to_reference, disparate_impact_flag.
    """
    rates = group_metrics[approval_col].dropna()
    ref_group = rates.idxmax()
    ref_rate = rates[ref_group]

    results = {}
    for group, rate in rates.items():
        ratio = rate / ref_rate if ref_rate > 0 else float("nan")
        results[group] = {
            "age_band": group,
            "approval_rate": round(float(rate), 4),
            "ratio_to_reference": round(float(ratio), 4),
            "disparate_impact_flag": bool(ratio < threshold),
        }
    return results


def generate_fairness_summary(
    group_metrics: pd.DataFrame,
    disparate_impact_results: dict,
) -> str:
    """Produce a human-readable fairness summary paragraph.

    Args:
        group_metrics: Output of compute_group_metrics().
        disparate_impact_results: Output of check_disparate_impact().

    Returns:
        One-paragraph string suitable for display in the Streamlit app.
    """
    rates = {g: v["approval_rate"] for g, v in disparate_impact_results.items()}
    flagged = [g for g, v in disparate_impact_results.items() if v["disparate_impact_flag"]]
    min_rate = min(rates.values())
    max_rate = max(rates.values())

    if not flagged:
        return (
            f"No age group shows disparate impact under the four-fifths rule. "
            f"Approval rates range from {min_rate:.0%} to {max_rate:.0%} across age bands."
        )

    flagged_str = ", ".join(
        f"{g} ({rates[g]:.0%}, ratio={disparate_impact_results[g]['ratio_to_reference']:.2f})"
        for g in flagged
    )
    ref = max(rates, key=rates.get)
    return (
        f"Disparate impact detected under the four-fifths rule for: {flagged_str}. "
        f"The reference group is '{ref}' with an approval rate of {rates[ref]:.0%}. "
        f"Approval rates range from {min_rate:.0%} to {max_rate:.0%}. "
        f"This may reflect underlying data bias rather than model bias."
    )


def run_fairness_audit() -> dict:
    """Orchestrate the full fairness audit; return results bundle.

    Returns:
        Dict with keys: group_metrics (DataFrame), disparate_impact_results (dict),
        summary_text (str).
    """
    model = joblib.load(_MODEL_PATH)
    X_test, y_true, age_band = _load_test_data()

    y_pred_favorable = _predict_favorable(model, X_test)
    group_metrics = compute_group_metrics(y_true, y_pred_favorable, age_band)
    disparate_impact_results = check_disparate_impact(group_metrics)
    summary_text = generate_fairness_summary(group_metrics, disparate_impact_results)

    return {
        "group_metrics": group_metrics,
        "disparate_impact_results": disparate_impact_results,
        "summary_text": summary_text,
    }


if __name__ == "__main__":
    import json

    audit = run_fairness_audit()

    print("\n=== Group Metrics by Age Band ===")
    print(audit["group_metrics"].to_string())

    print("\n=== Disparate Impact Flags ===")
    print(json.dumps(audit["disparate_impact_results"], indent=2))

    print("\n=== Summary ===")
    print(audit["summary_text"])
