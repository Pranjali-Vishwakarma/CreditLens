# CreditLens — Architecture

## Overview

CreditLens follows a linear pipeline from raw applicant data to an explainable, audited credit decision. The diagram below describes the data flow:

```
Applicant Input (Streamlit form)
        |
        v
  encode_single_applicant()   [preprocess.py]
        |
        v
  XGBClassifier.predict_proba()   [train.py / models/credit_model.pkl]
        |
        v
  prob_default = predict_proba[:, 0]
        |
   _____|_____
  |           |
  v           v
decision   SHAP values    Fairlearn MetricFrame
threshold  [explain.py]   [fairness.py]
[train.py]      |               |
  |         explain_applicant() run_fairness_audit()
  v             |               |
Approve /   reason string   group metrics +
Refer /     + top factors   disparate impact flags
Reject          |               |
  |_____________|_______________|
                v
        Streamlit Results Panel   [app.py]
```

---

## Module Responsibilities

### `src/preprocess.py`

**Responsibility**: Data ingestion, feature engineering, encoding, and train/test splitting.

**Inputs**: Raw German Credit Data (Statlog, UCI ML Repository — 1000 records, 20 attributes + binary target). Downloaded automatically from the UCI URL on first run; cached to `dataset/raw/german_credit.csv`.

**Outputs**: `dataset/processed/train.csv` and `dataset/processed/test.csv` (800/200 stratified split). Also exposes `FEATURE_COLUMNS` and `encode_single_applicant(raw: dict) -> pd.DataFrame` for single-applicant encoding at inference time.

**Key decisions**:
- Ordinal encoding for naturally ordered categoricals (`checking_account_status`, `savings_account`, `employment_since`, `property`, `job`) using documented integer mappings.
- One-hot encoding with `drop_first=True` for nominal categoricals to avoid perfect multicollinearity.
- `age_band` (`<25`, `25-40`, `40-60`, `60+`) kept as a plain string column in all output files — never fed to the model — for use by the fairness module.
- `debt_to_income_proxy = credit_amount × installment_rate_pct / 100` derived as a domain-informed feature.

---

### `src/train.py`

**Responsibility**: Model training, evaluation, serialization, and decision-threshold logic.

**Inputs**: `train.csv` and `test.csv` from `preprocess.py` (via `load_data()`).

**Outputs**: `models/credit_model.pkl` (serialized `XGBClassifier`). Prints accuracy, precision, recall, F1, ROC-AUC, and confusion matrix to stdout. Also saves `dataset/processed/test_with_agebands.csv` for the fairness audit.

**Probability convention** (critical — all downstream modules must follow this):
```
prob_default = model.predict_proba(X)[:, 0]   # P(target=0) = P(bad credit)
```
Higher `prob_default` = higher risk. Decision bands: `prob_default >= 0.5` → Reject, `>= 0.3` → Refer, else → Approve. These thresholds are exposed via `decision_from_probability()` and imported by both `app.py` and `fairness.py` to ensure consistency.

---

### `src/explain.py`

**Responsibility**: SHAP-based local and global explainability.

**Inputs**: Trained model (loaded via `load_model()`), single-row or batch feature DataFrames (no `target`, no `age_band`).

**Outputs**:
- `explain_applicant()` → dict with `decision`, `probability_of_default`, `top_risk_factors`, `top_protective_factors` (human-readable feature names).
- `generate_reason_string()` → single English sentence summarizing the decision.
- `global_feature_importance()` → prints top-10 features by mean |SHAP| and saves `docs/global_shap_summary.png`.

**Implementation**: Uses `shap.TreeExplainer` (exact, fast for tree models — no approximation). The explainer is cached at module level to avoid rebuilding it per call. SHAP values here are for class 1 (good credit): positive SHAP = protective, negative SHAP = risk factor.

See the visual companion: [`docs/global_shap_summary.png`](./global_shap_summary.png)

---

### `src/fairness.py`

**Responsibility**: Disparate impact audit across age bands using fairlearn.

**Inputs**: `test_with_agebands.csv` (includes `age_band` and `target`); trained model loaded from `models/credit_model.pkl`.

**Outputs**: Dict containing `group_metrics` (DataFrame), `disparate_impact_results` (per-group dict), and `summary_text` (human-readable paragraph). The single entry point `run_fairness_audit()` is imported by `app.py`.

**Methodology**:
- Sensitive attribute: `age_band` (`<25`, `25-40`, `40-60`, `60+`).
- Favorable outcome: `Approve` (via `decision_from_probability()` — same thresholds as the live app).
- Refer + Reject collapsed to "not approved" (documented simplification).
- Four-fifths rule: a group is flagged if its approval rate < 0.8 × the highest-approval-rate group's rate.
- Metrics per group: approval rate (selection rate), false rejection rate, group size.

---

### `src/app.py`

**Responsibility**: Interactive Streamlit UI tying all modules together.

**Inputs**: User form inputs (sidebar); sample applicants from `examples/sample_applicants.csv`.

**Data flow**:
1. Form values → `encode_single_applicant()` → 37-column feature row.
2. Feature row → `explain_applicant()` → decision + SHAP explanation dict.
3. `generate_reason_string()` → human-readable sentence.
4. `run_fairness_audit()` (cached, model/test-set level, not per-applicant) → fairness table.
5. Results rendered: color-coded decision banner, risk/protective factor lists, collapsible fairness expander.

**Caching**: Model loaded once with `@st.cache_resource`; fairness audit computed once with `@st.cache_data` — neither recomputes on applicant re-submission.
