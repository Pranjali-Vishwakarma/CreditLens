"""
app.py — CreditLens Streamlit UI.

Encoding: uses encode_single_applicant() from preprocess.py so form inputs
are encoded identically to training data — no logic duplicated here.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from explain import explain_applicant, generate_reason_string, load_model
from fairness import run_fairness_audit
from preprocess import encode_single_applicant

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CreditLens", layout="wide", page_icon="🔍")

# ---------------------------------------------------------------------------
# Category option maps  (display label -> coded value)
# ---------------------------------------------------------------------------
CHECKING = {
    "< 0 DM (negative balance)":       "A11",
    "0 – 200 DM":                       "A12",
    ">= 200 DM":                        "A13",
    "No checking account":              "A14",
}
SAVINGS = {
    "< 100 DM":                         "A61",
    "100 – 500 DM":                     "A62",
    "500 – 1000 DM":                    "A63",
    ">= 1000 DM":                       "A64",
    "No savings / unknown":             "A65",
}
EMPLOYMENT = {
    "Unemployed":                       "A71",
    "< 1 year":                         "A72",
    "1 – 4 years":                      "A73",
    "4 – 7 years":                      "A74",
    ">= 7 years":                       "A75",
}
CREDIT_HISTORY = {
    "No credits / all paid duly":                   "A30",
    "All credits at this bank paid duly":           "A31",
    "Existing credits paid duly till now":          "A32",
    "Delay in past":                                "A33",
    "Critical / other credits elsewhere":           "A34",
}
PURPOSE = {
    "Car (new)":            "A40",
    "Car (used)":           "A41",
    "Furniture/equipment":  "A42",
    "Radio/television":     "A43",
    "Domestic appliances":  "A44",
    "Repairs":              "A45",
    "Education":            "A46",
    "Retraining":           "A48",
    "Business":             "A49",
    "Others":               "A410",
}
PERSONAL_STATUS = {
    "Male: divorced/separated":          "A91",
    "Female: divorced/separated/married":"A92",
    "Male: single":                      "A93",
    "Male: married/widowed":             "A94",
}
OTHER_DEBTORS = {
    "None":         "A101",
    "Co-applicant": "A102",
    "Guarantor":    "A103",
}
PROPERTY = {
    "Real estate":                      "A121",
    "Life insurance / building savings":"A122",
    "Car or other":                     "A123",
    "Unknown / no property":            "A124",
}
OTHER_INSTALMENT = {
    "Bank":     "A141",
    "Stores":   "A142",
    "None":     "A143",
}
HOUSING = {
    "Rent":     "A151",
    "Own":      "A152",
    "For free": "A153",
}
JOB = {
    "Unemployed / unskilled (non-resident)":    "A171",
    "Unskilled (resident)":                     "A172",
    "Skilled employee / official":              "A173",
    "Management / self-employed / highly qualified": "A174",
}
TELEPHONE = {
    "None":             "A191",
    "Yes, registered":  "A192",
}
FOREIGN_WORKER = {
    "Yes":  "A201",
    "No":   "A202",
}

# ---------------------------------------------------------------------------
# Sample applicants — loaded from examples/sample_applicants.csv
# ---------------------------------------------------------------------------
_SAMPLES_CSV = Path(__file__).parent.parent / "examples" / "sample_applicants.csv"


def _load_samples() -> dict:
    if not _SAMPLES_CSV.exists():
        return {}
    df = pd.read_csv(_SAMPLES_CSV)
    return {row["name"]: row.drop("name").to_dict() for _, row in df.iterrows()}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_fairness_audit():
    return run_fairness_audit()


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.title("🔍 CreditLens")
st.markdown(
    "**Explainable Credit Risk & Loan Approval Simulator** — "
    "powered by XGBoost + SHAP. Enter applicant details to get an "
    "instant decision with transparent reasoning and a fairness audit."
)

model = get_model()

# --- Sidebar: sample loader + form ---
with st.sidebar:
    st.header("Applicant Details")

    _samples = _load_samples()
    sample_choice = st.selectbox(
        "Load sample applicant",
        options=["(manual entry)"] + list(_samples.keys()),
    )
    defaults = _samples.get(sample_choice, {})

    def _sel(mapping, key, default_code=None):
        """Selectbox helper that honours a loaded sample default."""
        code = defaults.get(key, default_code or list(mapping.values())[0])
        idx = list(mapping.values()).index(code) if code in mapping.values() else 0
        label = st.selectbox(
            key.replace("_", " ").title(),
            options=list(mapping.keys()),
            index=idx,
            key=key,
        )
        return mapping[label]

    st.subheader("Financial Profile")
    checking     = _sel(CHECKING, "checking_account_status")
    savings      = _sel(SAVINGS,  "savings_account")
    credit_hist  = _sel(CREDIT_HISTORY, "credit_history")
    credit_amt   = st.number_input(
        "Credit Amount (DM)", min_value=1, max_value=200000,
        value=int(defaults.get("credit_amount", 5000)), step=100,
    )
    duration     = st.slider(
        "Duration (months)", 1, 72,
        value=int(defaults.get("duration_months", 24)),
    )
    install_rate = st.slider(
        "Installment Rate (% of income)", 1, 4,
        value=int(defaults.get("installment_rate_pct", 2)),
    )
    purpose      = _sel(PURPOSE, "purpose")

    st.subheader("Personal Details")
    age          = st.slider("Age", 18, 90, value=int(defaults.get("age", 35)))
    personal_st  = _sel(PERSONAL_STATUS, "personal_status_sex")
    employment   = _sel(EMPLOYMENT, "employment_since")
    housing      = _sel(HOUSING, "housing")
    job          = _sel(JOB, "job")
    residence    = st.slider(
        "Years at Residence", 1, 4,
        value=int(defaults.get("residence_since", 2)),
    )
    num_dep      = st.slider(
        "Number of Dependents", 1, 2,
        value=int(defaults.get("num_dependents", 1)),
    )
    telephone    = _sel(TELEPHONE, "telephone")
    foreign_wk   = _sel(FOREIGN_WORKER, "foreign_worker")

    st.subheader("Other Credit")
    property_t   = _sel(PROPERTY, "property")
    other_deb    = _sel(OTHER_DEBTORS, "other_debtors")
    other_inst   = _sel(OTHER_INSTALMENT, "other_installment_plans")
    existing_cr  = st.slider(
        "Existing Credits at This Bank", 1, 4,
        value=int(defaults.get("existing_credits_count", 1)),
    )

    submitted = st.button("Submit Application", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if not submitted:
    st.info("Fill in the applicant details in the sidebar and click **Submit Application**.")
else:
    # Validate
    if credit_amt <= 0:
        st.error("Credit amount must be greater than 0.")
        st.stop()

    raw_input = {
        "checking_account_status":  checking,
        "duration_months":          duration,
        "credit_amount":            credit_amt,
        "savings_account":          savings,
        "employment_since":         employment,
        "installment_rate_pct":     install_rate,
        "residence_since":          residence,
        "property":                 property_t,
        "age":                      age,
        "existing_credits_count":   existing_cr,
        "job":                      job,
        "num_dependents":           num_dep,
        "credit_history":           credit_hist,
        "purpose":                  purpose,
        "personal_status_sex":      personal_st,
        "other_debtors":            other_deb,
        "other_installment_plans":  other_inst,
        "housing":                  housing,
        "telephone":                telephone,
        "foreign_worker":           foreign_wk,
    }

    applicant_row = encode_single_applicant(raw_input)
    explanation   = explain_applicant(model, applicant_row)
    reason        = generate_reason_string(explanation)

    decision = explanation["decision"]
    prob     = explanation["probability_of_default"]

    # --- Decision banner ---
    col1, col2 = st.columns([1, 3])
    with col1:
        if decision == "Approve":
            st.success(f"## Approved")
        elif decision == "Refer":
            st.warning(f"## Refer for Review")
        else:
            st.error(f"## Rejected")
        st.metric("Probability of Default", f"{prob:.1%}")

    with col2:
        st.markdown(f"**Reason:** {reason}")

        risk_factors = explanation.get("top_risk_factors", [])
        protect_factors = explanation.get("top_protective_factors", [])

        rcol, pcol = st.columns(2)
        with rcol:
            st.markdown("**Top Risk Factors**")
            for f in risk_factors:
                st.markdown(f"- 🔴 **{f['feature']}** (impact: {f['impact']:.3f})")
        with pcol:
            st.markdown("**Top Protective Factors**")
            for f in protect_factors:
                st.markdown(f"- 🟢 **{f['feature']}** (impact: {f['impact']:.3f})")

    st.divider()

    # --- Fairness panel ---
    with st.expander("Fairness & Bias Audit", expanded=False):
        audit = get_fairness_audit()
        st.markdown("#### Group Metrics by Age Band")
        gm = audit["group_metrics"].copy()
        gm.index.name = "Age Band"
        gm.columns = ["Approval Rate", "False Rejection Rate", "Group Size"]
        gm["Approval Rate"] = gm["Approval Rate"].map("{:.1%}".format)
        gm["False Rejection Rate"] = gm["False Rejection Rate"].map(
            lambda x: f"{x:.1%}" if x == x else "N/A"
        )
        st.dataframe(gm, use_container_width=True)

        st.markdown("#### Disparate Impact (Four-Fifths Rule)")
        di = audit["disparate_impact_results"]
        for group, info in di.items():
            flag = "⚠️ FLAGGED" if info["disparate_impact_flag"] else "OK"
            st.markdown(
                f"- **{group}**: approval {info['approval_rate']:.1%}, "
                f"ratio to reference = {info['ratio_to_reference']:.2f} — {flag}"
            )

        st.markdown("#### Summary")
        st.info(audit["summary_text"])
