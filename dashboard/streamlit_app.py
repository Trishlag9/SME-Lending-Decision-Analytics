from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="SME Lending Workbench",
    page_icon="🏦",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "reports/sme_dashboard_v1/borrower_dashboard.csv"


@st.cache_data
def load_data(path, modified_time):
    return pd.read_csv(path, dtype={"applicant_id": "string"})


def percentage(value):
    return "Not available" if pd.isna(value) else f"{float(value):.2%}"


def rupees(value):
    return "Not available" if pd.isna(value) else f"₹{float(value):,.0f}"


def is_true(value):
    return pd.notna(value) and str(value).strip().lower() == "true"


st.title("SME Lending Workbench")
st.caption(
    "Synthetic portfolio demonstration • "
    "Application decisions and subsequent account management"
)

if not DATA_PATH.is_file():
    st.error("Dashboard dataset missing. Complete Notebook 06's export cell.")
    st.stop()

data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)

required_columns = {
    "applicant_id",
    "application_pd",
    "historical_booking_status",
    "application_policy_decision",
    "predicted_deterioration",
    "monitoring_status",
    "limit_increase_decision",
    "base_case_offer_increase"
}

missing_columns = required_columns - set(data.columns)
if missing_columns:
    st.error(f"Dataset is missing columns: {sorted(missing_columns)}")
    st.stop()

if data.empty or not data["applicant_id"].is_unique:
    st.error("The dataset must contain one row per applicant.")
    st.stop()

monitored = data["predicted_deterioration"].notna()
watchlisted = data["on_heldout_watchlist"].map(is_true)
has_offer = data["base_case_offer_increase"].notna()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Applicants", f"{len(data):,}")
c2.metric("Monitoring assessments", f"{monitored.sum():,}")
c3.metric("Prioritized for review", f"{watchlisted.sum():,}")
c4.metric("Base-case provisional offers", f"{has_offer.sum():,}")

st.sidebar.header("Borrower lookup")
population = st.sidebar.selectbox(
    "Choose a population",
    [
        "All applicants",
        "Monitored accounts",
        "Watchlist accounts",
        "Provisional offer candidates"
    ]
)

population_masks = {
    "All applicants": pd.Series(True, index=data.index),
    "Monitored accounts": monitored,
    "Watchlist accounts": watchlisted,
    "Provisional offer candidates": has_offer
}

visible = data.loc[population_masks[population]]
ids = sorted(visible["applicant_id"].tolist())

if not ids:
    st.info("No accounts in this population.")
    st.stop()

default_index = ids.index("SME008138") if "SME008138" in ids else 0
selected_id = st.sidebar.selectbox(
    "Applicant ID — type to search",
    ids,
    index=default_index
)

borrower = data.loc[data["applicant_id"].eq(selected_id)].iloc[0]

st.subheader(f"Borrower {selected_id}")
st.write(
    f"{borrower['industry']} · {borrower['region']} · "
    f"{borrower['entity_type']}"
)
st.write(f"Historical booking: **{borrower['historical_booking_status']}**")

application_tab, monitoring_tab, increase_tab = st.tabs([
    "Application decision & pricing",
    "Early warning",
    "Credit limit increase"
])

with application_tab:
    st.caption("Application-time assessment")

    a, b = st.columns(2)
    a.metric("Application default PD", percentage(borrower["application_pd"]))

    score = borrower.get("display_score")
    b.metric(
        "Application score",
        "Not exported" if pd.isna(score) else f"{float(score):.0f}"
    )

    decision = borrower["application_policy_decision"]
    if pd.isna(decision):
        st.info(
            "Application-policy decision not assessed in the saved "
            "validation demonstration."
        )
    else:
        st.write(f"Proposed application decision: **{decision}**")

    if pd.notna(borrower["application_target_rate"]):
        a, b = st.columns(2)
        a.metric(
            "Illustrative break-even rate",
            percentage(borrower["application_break_even_rate"])
        )
        b.metric(
            "Illustrative target rate",
            percentage(borrower["application_target_rate"])
        )
        st.caption(
            "Assumptions: LGD 45%, funding cost 8%, operating cost 2%, "
            "and target margin 2 percentage points. "
            "These are estimates, not historical quoted rates."
        )
    else:
        st.info(
            "No pricing shown: this applicant was not historically booked."
        )

    st.caption(
    f"Application scorecard sample: {borrower['application_sample']}. "
    "Independent application evaluation used validation accounts only. "
    "Scores and policy outputs are displayed for all applicants."
)

with monitoring_tab:
    st.caption("Review after 24 monthly observations — through month index 23")

    if pd.isna(borrower["predicted_deterioration"]):
        st.info("Not assessed in the held-out monitoring demonstration.")
    else:
        st.write(f"**{borrower['monitoring_status']}**")

        a, b = st.columns(2)
        a.metric(
            "Estimated deterioration probability",
            percentage(borrower["predicted_deterioration"])
        )
        b.metric("Review rank", f"{int(borrower['review_rank']):,}")

        st.write("**Behavioural warning signals**")
        st.write(borrower["warning_reasons"])

        a, b, c = st.columns(3)
        a.metric(
            "Recent average utilization",
            percentage(borrower["utilization_recent_3m"])
        )
        b.metric(
            "Deposit decline",
            percentage(borrower["deposit_decline_pct"])
        )
        c.metric(
            "Maximum recent days past due",
            f"{borrower['max_dpd_recent_3m']:.0f}"
        )

        st.caption(
            "A negative deposit decline means inflows increased. "
            "Warning signals are descriptive rules, not model attributions. "
            "Deterioration probability is not default PD."
        )

with increase_tab:
    st.caption("Account-management simulation at the month-23 review")

    st.write(f"**{borrower['limit_increase_decision']}**")

    if pd.notna(borrower["current_credit_limit"]):
        a, b = st.columns(2)
        a.metric(
            "Current credit limit",
            rupees(borrower["current_credit_limit"])
        )
        b.metric(
            "Latest outstanding balance",
            rupees(borrower["latest_outstanding_balance"])
        )

        if pd.notna(borrower["base_case_offer_increase"]):
            a, b = st.columns(2)
            a.metric(
                "Provisional increase",
                rupees(borrower["base_case_offer_increase"])
            )
            b.metric(
                "Provisional new limit",
                rupees(borrower["base_case_offer_new_limit"])
            )

            st.metric(
                "Estimated incremental pre-tax return on allocated equity",
                percentage(borrower["incremental_pretax_roe"])
            )

            if is_true(borrower["passes_all_tested_scenarios"]):
                st.success("Meets the return hurdle in all tested scenarios.")
            else:
                st.warning(
                    "Passes the base case but fails at least one "
                    "downside scenario."
                )

            st.caption(
                "Scenario assumptions: interest rate 16%, drawdown 50%, "
                "funding cost 8%, operating cost 2%, LGD 45%, "
                "equity allocation 10%, and pre-tax return hurdle 15%."
            )
        else:
            reasons = borrower.get("screen_failure_reasons")
            if pd.notna(reasons):
                st.write(f"Screening assessment: {reasons}")
            st.caption(
                "Passing initial screening does not guarantee "
                "positive headroom or an offer."
            )

st.divider()
st.caption(
    "Research demonstration, not a live lending system. "
    "Limit increases are rule-based provisional assessments requiring "
    "updated financial information and default-risk assessment. "
    "No customer offers have been issued."
)
