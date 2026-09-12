"""
explain.py — SHAP-based explainability for CreditLens.

SHAP convention used here:
  TreeExplainer returns shap_values with shape (n_samples, n_features).
  For XGBoost binary classification these are SHAP values for class 1 (good credit).
  Positive SHAP  ->  pushes toward good credit  (protective factor)
  Negative SHAP  ->  pushes toward default / bad credit (risk factor)

  prob_default = model.predict_proba(X)[:, 0]   (consistent with train.py)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).parent))
from train import decision_from_probability  # noqa: E402

_ROOT = Path(__file__).parent.parent
_MODEL_PATH = _ROOT / "models" / "credit_model.pkl"
_DOCS_PATH = _ROOT / "docs"

# Module-level explainer cache (rebuilt only when model changes)
_explainer_cache: dict = {}

# Human-readable label map for known base feature names
_LABEL_MAP: dict[str, str] = {
    "checking_account_status": "Checking Account Status",
    "duration_months": "Loan Duration (Months)",
    "credit_amount": "Credit Amount",
    "savings_account": "Savings Account Balance",
    "employment_since": "Employment Tenure",
    "installment_rate_pct": "Installment Rate (%)",
    "residence_since": "Years at Residence",
    "property": "Property Ownership",
    "age": "Age",
    "existing_credits_count": "Existing Credit Count",
    "job": "Job Type",
    "num_dependents": "Number of Dependents",
    "debt_to_income_proxy": "Debt-to-Income (Proxy)",
    "credit_history": "Credit History",
    "purpose": "Loan Purpose",
    "personal_status_sex": "Personal Status / Sex",
    "other_debtors": "Other Debtors / Guarantors",
    "other_installment_plans": "Other Installment Plans",
    "housing": "Housing Type",
    "telephone": "Has Telephone",
    "foreign_worker": "Foreign Worker",
}


def _readable(col: str) -> str:
    """Map a column name (possibly one-hot suffix) to a human-readable label."""
    base = re.sub(r"_A\d+$", "", col)  # strip _A41, _A102, etc.
    return _LABEL_MAP.get(base, base.replace("_", " ").title())


def load_model(path: str | Path = _MODEL_PATH):
    """Load and return the saved XGBClassifier from disk.

    Args:
        path: Path to the .pkl model file.

    Returns:
        Fitted XGBClassifier.
    """
    return joblib.load(path)


def get_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Return SHAP values (class-1 / good-credit axis) for X.

    Uses a module-level cached TreeExplainer — safe to call in a loop.

    Args:
        model: Fitted XGBClassifier.
        X: Feature DataFrame (no target, no age_band).

    Returns:
        ndarray of shape (n_samples, n_features) — SHAP values for class 1.
    """
    model_id = id(model)
    if model_id not in _explainer_cache:
        _explainer_cache[model_id] = shap.TreeExplainer(model)
    explainer = _explainer_cache[model_id]
    sv = explainer.shap_values(X)
    # XGBoost binary: sv may be a list [class0, class1] or a single array (class1)
    if isinstance(sv, list):
        sv = sv[1]
    return sv


def explain_applicant(
    model,
    applicant_row: pd.DataFrame,
    top_n: int = 3,
) -> dict:
    """Explain a single applicant's credit decision via SHAP.

    Args:
        model: Fitted XGBClassifier.
        applicant_row: Single-row DataFrame with training features
                       (no target, no age_band).
        top_n: Number of top risk/protective factors to return.

    Returns:
        Dict with keys: decision, probability_of_default,
        top_risk_factors, top_protective_factors.
    """
    sv = get_shap_values(model, applicant_row)  # (1, n_features)
    shap_row = sv[0]  # (n_features,) — positive = toward good credit

    prob_default = float(model.predict_proba(applicant_row)[0, 0])
    decision = decision_from_probability(prob_default)

    cols = list(applicant_row.columns)

    # Risk factors: most negative SHAP (pushes toward default)
    risk_idx = np.argsort(shap_row)[:top_n]
    # Protective factors: most positive SHAP (pushes toward good credit)
    protect_idx = np.argsort(shap_row)[-top_n:][::-1]

    def _factor(idx, direction):
        return {
            "feature": _readable(cols[idx]),
            "impact": round(float(abs(shap_row[idx])), 4),
            "direction": direction,
        }

    return {
        "decision": decision,
        "probability_of_default": round(prob_default, 4),
        "top_risk_factors": [
            _factor(i, "increases risk") for i in risk_idx if shap_row[i] < 0
        ],
        "top_protective_factors": [
            _factor(i, "reduces risk") for i in protect_idx if shap_row[i] > 0
        ],
    }


def generate_reason_string(explanation_dict: dict) -> str:
    """Convert an explain_applicant dict into a readable one-sentence summary.

    Args:
        explanation_dict: Output of explain_applicant().

    Returns:
        Human-readable decision rationale string.
    """
    decision = explanation_dict["decision"]
    risk = [f["feature"] for f in explanation_dict.get("top_risk_factors", [])]
    protect = [f["feature"] for f in explanation_dict.get("top_protective_factors", [])]

    def _join(items: list[str]) -> str:
        if not items:
            return "unknown factors"
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]

    if decision == "Approve":
        base = f"Approved based on {_join(protect)}"
        suffix = f", despite elevated {_join(risk)}." if risk else "."
        return base + suffix

    if decision == "Refer":
        base = f"Referred for manual review due to {_join(risk)}"
        suffix = f", partially offset by {_join(protect)}." if protect else "."
        return base + suffix

    # Reject
    base = f"Rejected primarily due to {_join(risk)}"
    suffix = f", partially offset by {_join(protect)}." if protect else "."
    return base + suffix


def global_feature_importance(
    model,
    X: pd.DataFrame,
    save_path: str | Path = _DOCS_PATH / "global_shap_summary.png",
) -> None:
    """Compute mean-|SHAP| importance; save bar plot; print top 10.

    Args:
        model: Fitted XGBClassifier.
        X: Feature DataFrame for the full test set (no target, no age_band).
        save_path: Output path for the PNG plot.
    """
    sv = get_shap_values(model, X)  # (n_samples, n_features)
    mean_abs = np.abs(sv).mean(axis=0)
    cols = list(X.columns)

    order = np.argsort(mean_abs)[::-1]
    top10_names = [_readable(cols[i]) for i in order[:10]]
    top10_vals = mean_abs[order[:10]]

    print("\n=== Top 10 Global Feature Importances (mean |SHAP|) ===")
    for name, val in zip(top10_names, top10_vals):
        print(f"  {name:<40} {val:.4f}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10_names[::-1], top10_vals[::-1], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved -> {save_path}")


if __name__ == "__main__":
    import json

    model = load_model()
    test_df = pd.read_csv(_ROOT / "dataset" / "processed" / "test_with_agebands.csv")

    # Drop non-feature cols; cast bools to int (same as train._get_X)
    X_test = test_df.drop(columns=["target", "age_band"], errors="ignore")
    bool_cols = X_test.select_dtypes(include="bool").columns
    X_test[bool_cols] = X_test[bool_cols].astype("int8")

    demo_row = X_test.iloc[[0]]

    print("=== Demo Applicant Explanation ===")
    result = explain_applicant(model, demo_row)
    print(json.dumps(result, indent=2))

    print("\n=== Reason String ===")
    print(generate_reason_string(result))

    global_feature_importance(model, X_test)
