"""
train.py — Train and evaluate the XGBoost credit risk model.

Probability convention (IMPORTANT — read before modifying):
  model.predict_proba(X)[:, 0]  =  P(target=0)  =  P(bad credit)  =  prob_default
  model.predict_proba(X)[:, 1]  =  P(target=1)  =  P(good credit)

  All downstream modules (explain.py, app.py) must use predict_proba[:, 0]
  as the probability of default.  Higher value = higher risk.

ROC-AUC convention:
  sklearn.roc_auc_score expects (y_true, y_score) where y_score is the
  probability of the *positive* class for the label being evaluated.
  Here we treat default (class 0) as the "event of interest", so:
    y_true  = 1 - y_test   (flip so default=1 for sklearn)
    y_score = predict_proba[:, 0]
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Ensure src/ imports work when run directly
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import load_data  # noqa: E402

_ROOT = Path(__file__).parent.parent
MODEL_PATH = _ROOT / "models" / "credit_model.pkl"
TEST_WITH_BANDS_PATH = _ROOT / "dataset" / "processed" / "test_with_agebands.csv"

# Columns that must never enter the feature matrix
_NON_FEATURE_COLS = {"target", "age_band"}


def _get_X(df: pd.DataFrame) -> pd.DataFrame:
    """Drop target + age_band; assert all remaining columns are numeric."""
    X = df.drop(columns=[c for c in _NON_FEATURE_COLS if c in df.columns])
    # bool columns (from pd.get_dummies) survive CSV round-trip as bool dtype;
    # cast to int8 so they pass the numeric guard below.
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype("int8")
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns in feature matrix: {non_numeric}")
    return X


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Train an XGBClassifier for binary credit-risk classification.

    Hyperparameters are conservative for an 800-row dataset (avoid overfitting).

    Args:
        X_train: Numeric feature matrix.
        y_train: Binary labels (1=good, 0=bad).

    Returns:
        Fitted XGBClassifier.
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate model; print confusion matrix; return metrics dict.

    Positive class for all metrics = default (class 0).
    prob_default = predict_proba[:, 0].

    Args:
        model: Fitted XGBClassifier.
        X_test: Numeric feature matrix (test set).
        y_test: True labels (1=good, 0=bad).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc.
    """
    y_pred = model.predict(X_test)
    prob_default = model.predict_proba(X_test)[:, 0]

    # Flip labels so sklearn treats "default" as the positive class
    y_default = 1 - y_test
    y_pred_default = 1 - y_pred

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_default, y_pred_default, zero_division=0),
        "recall":    recall_score(y_default, y_pred_default, zero_division=0),
        "f1":        f1_score(y_default, y_pred_default, zero_division=0),
        "roc_auc":   roc_auc_score(y_default, prob_default),
    }

    cm = confusion_matrix(y_test, y_pred)
    print("\n=== Confusion Matrix (rows=actual, cols=predicted) ===")
    print(f"Labels: [0=bad, 1=good]")
    print(cm)
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    return metrics


def save_model(model: XGBClassifier, path: str | Path = MODEL_PATH) -> None:
    """Persist model to disk with joblib.

    Args:
        model: Fitted XGBClassifier.
        path: Destination path (default: models/credit_model.pkl).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved -> {path}")


def decision_from_probability(
    prob_default: float,
    threshold_reject: float = 0.5,
    threshold_refer: float = 0.3,
) -> str:
    """Map a probability-of-default to a loan decision.

    Bands:
      prob_default >= threshold_reject            -> "Reject"
      threshold_refer <= prob_default < threshold_reject -> "Refer"
      prob_default < threshold_refer              -> "Approve"

    Args:
        prob_default: P(target=0), i.e., probability of default. Range [0, 1].
        threshold_reject: Cutoff above which the application is rejected.
        threshold_refer:  Cutoff above which the application is referred for review.

    Returns:
        One of "Approve", "Refer", or "Reject".
    """
    if prob_default >= threshold_reject:
        return "Reject"
    if prob_default >= threshold_refer:
        return "Refer"
    return "Approve"


if __name__ == "__main__":
    train_df, test_df = load_data()

    X_train = _get_X(train_df)
    y_train = train_df["target"]
    X_test = _get_X(test_df)
    y_test = test_df["target"]

    print(f"Training on {X_train.shape[0]} rows, {X_train.shape[1]} features.")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    save_model(model)

    # Save test set with age_band intact for Chunk 5 (fairness audit)
    test_df.to_csv(TEST_WITH_BANDS_PATH, index=False)
    print(f"Test set (with age_band) saved -> {TEST_WITH_BANDS_PATH}")
