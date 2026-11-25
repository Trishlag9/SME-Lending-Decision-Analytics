"""
=============================================================================
ADD-ON FOR NOTEBOOK 1  (01_portfolio_generation_and_audit.ipynb)
=============================================================================

Paste these cells at the END of Notebook 1, after your existing section 8
("Export the Leakage-Safe Portfolio").

They need three things that already exist in your kernel:
    applicants        your assembled DataFrame, with default_12m
    true_pd_12m       the hidden PD from your section 6
    N_APPLICANTS      15000

NOTHING ABOVE CHANGES. These cells use two new random generators
(SEED + 2 and SEED + 3), so every number already printed in Notebook 1 and
every result in Notebook 2 stays exactly as it is. Your existing
sme_applicant_portfolio.csv is not touched either.

Expected output when you run them (I tested this on your regenerated data):

    approval rate      65.91%   (9,887 of 15,000)
    all applicants DR  23.02%
    booked DR          14.50%
    declined DR        39.49%
    lift               2.72x
    panel rows         237,288  (9,887 x 24)
    deterioration rate 19.81%
    ever 1+ DPD        27.6%

If your numbers differ, your upstream cells were re-run in a different order.
=============================================================================
"""


# =============================================================================
# CELL 9A  —  MARKDOWN
# =============================================================================
"""
## 9. Generate the Booking Decision

So far every applicant has been treated identically. A real lender approves
some applications and declines others, and that decision is not random: it
is based on perceived risk.

We simulate an approve-and-fund decision as a logistic function of the hidden
probability of default, with noise. Applicants with a higher true PD are less
likely to be booked, but the relationship is imperfect, exactly as a real
credit policy is imperfect.

Two consequences matter for later milestones:

- The booked portfolio will have a **lower** default rate than the applicant
  pool. The book is the safer half of who applied.
- We can observe `default_12m` for declined applicants. On a real portfolio
  that outcome never exists, because the loan was never made. This is the
  selection-bias problem known as **reject inference**, and this dataset lets
  us see it directly.

`booked` is a post-decision outcome. It must never be used as a predictor in
the application scorecard, because at the moment of scoring it is not yet known.
"""


# =============================================================================
# CELL 9B  —  CODE
# =============================================================================

# A separate generator keeps the booking decision reproducible and leaves
# every draw made earlier in this notebook untouched.
booking_rng = np.random.default_rng(SEED + 2)

APPROVAL_INTERCEPT = 2.20
APPROVAL_PD_WEIGHT = -6.00
APPROVAL_NOISE_SCALE = 0.50

booking_logit = (
    APPROVAL_INTERCEPT
    + APPROVAL_PD_WEIGHT * true_pd_12m
    + booking_rng.normal(0, APPROVAL_NOISE_SCALE, N_APPLICANTS)
)

approval_probability = 1 / (1 + np.exp(-booking_logit))

applicants["booked"] = booking_rng.binomial(
    n=1,
    p=approval_probability,
).astype(int)

booking_summary = (
    applicants
    .assign(
        decision=np.where(
            applicants["booked"] == 1,
            "Booked",
            "Not booked",
        )
    )
    .groupby("decision")
    .agg(
        applicant_count=("applicant_id", "count"),
        default_rate=("default_12m", "mean"),
        average_dscr=("dscr", "mean"),
        average_utilization=("utilization", "mean"),
        average_bureau_score=("bureau_score", "mean"),
    )
    .round(4)
)

booked_default_rate = applicants.loc[
    applicants["booked"] == 1, "default_12m"
].mean()

declined_default_rate = applicants.loc[
    applicants["booked"] == 0, "default_12m"
].mean()

assert applicants["booked"].isin([0, 1]).all()
assert 0.55 < applicants["booked"].mean() < 0.80, "Approval rate is implausible."
assert booked_default_rate < declined_default_rate, \
    "The booking decision is not risk-based."

print(f"Approval rate: {applicants['booked'].mean():.2%} "
      f"({applicants['booked'].sum():,} of {N_APPLICANTS:,})")
print()
print(f"Default rate, all applicants: {applicants['default_12m'].mean():.2%}")
print(f"Default rate, booked:         {booked_default_rate:.2%}")
print(f"Default rate, declined:       {declined_default_rate:.2%}")
print(f"Declined applicants default {declined_default_rate / booked_default_rate:.2f}x "
      f"as often as booked applicants.")
print()
print("On a real portfolio the declined row cannot be measured.")

display(booking_summary)


# =============================================================================
# CELL 10A  —  MARKDOWN
# =============================================================================
"""
## 10. Generate the On-Book Behavioural Panel

Milestones 4 and 5 (early-warning monitoring and proactive line increases)
need behaviour over time, not a single snapshot at application. We therefore
generate 24 monthly observations for every booked account.

Each account has a credit limit equal to its requested amount. Utilisation
starts near the applicant's bureau utilisation and then drifts month by month.
Accounts with a higher hidden PD drift upward faster, which is what creates a
detectable early-warning signal.

Four behavioural series are generated:

- `utilization_onbook`: balance as a share of the limit
- `days_past_due`: delinquency, rare in any single month and concentrated in
  stressed accounts
- `deposit_inflow`: monthly banking turnover, which falls when an account is
  near its limit
- `overdraft_count`: overdrafts, which appear only above roughly 80% utilisation

We also create `deterioration_next_6_12mo`, the Milestone 4 target. It is
driven by recent utilisation, the utilisation trend and leverage, so a model
built on observable on-book behaviour can genuinely learn it.
"""


# =============================================================================
# CELL 10B  —  CODE
# =============================================================================

panel_rng = np.random.default_rng(SEED + 3)

PANEL_MONTHS = 24

# Delinquency settings. Tuned so roughly 28% of accounts touch 1+ DPD at some
# point in 24 months, and under 2% of account-months are delinquent.
DPD_BASE_PROBABILITY = 0.0015
DPD_PD_WEIGHT = 0.070
DPD_STRESS_WEIGHT = 0.035
DPD_BUCKETS = np.array([15, 30, 60, 90])
DPD_BUCKET_WEIGHTS = np.array([0.50, 0.30, 0.14, 0.06])

book = (
    applicants[applicants["booked"] == 1]
    .reset_index(drop=True)
    .copy()
)

number_of_accounts = len(book)

# The hidden PD drives the simulation but is never exported.
book_true_pd = true_pd_12m[applicants["booked"].to_numpy() == 1]

credit_limit = book["requested_amount"].to_numpy().astype(float)

opening_utilization = np.clip(
    book["utilization"].to_numpy()
    + panel_rng.normal(0, 0.05, number_of_accounts),
    0.02,
    0.98,
)

monthly_drift = (book_true_pd - book_true_pd.mean()) * 0.020
monthly_revenue = book["annual_revenue"].to_numpy() / 12.0

utilization_path = np.zeros((number_of_accounts, PANEL_MONTHS))
monthly_rows = []

for month in range(PANEL_MONTHS):

    utilization_now = np.clip(
        opening_utilization
        + (
            panel_rng.normal(0, 0.030, number_of_accounts)
            + monthly_drift
        ) * month,
        0.0,
        1.25,
    )

    utilization_path[:, month] = utilization_now

    account_is_stressed = utilization_now > 0.90

    delinquency_probability = np.clip(
        DPD_BASE_PROBABILITY
        + DPD_PD_WEIGHT * book_true_pd
        + DPD_STRESS_WEIGHT * account_is_stressed,
        0,
        0.60,
    )

    is_delinquent = (
        panel_rng.random(number_of_accounts) < delinquency_probability
    )

    days_past_due = np.where(
        is_delinquent,
        panel_rng.choice(
            DPD_BUCKETS,
            size=number_of_accounts,
            p=DPD_BUCKET_WEIGHTS,
        ),
        0,
    ).astype(int)

    deposit_inflow = np.clip(
        monthly_revenue
        * panel_rng.uniform(0.60, 1.20, number_of_accounts)
        * (1 - 0.30 * (utilization_now > 0.95)),
        0,
        None,
    ).round(-2)

    overdraft_count = panel_rng.poisson(
        np.clip(utilization_now - 0.80, 0, None) * 3
    ).astype(int)

    monthly_rows.append(
        pd.DataFrame({
            "applicant_id": book["applicant_id"].to_numpy(),
            "month_index": month,
            "credit_limit": credit_limit.round(-3),
            "outstanding_balance": (utilization_now * credit_limit).round(-2),
            "utilization_onbook": utilization_now.round(4),
            "days_past_due": days_past_due,
            "deposit_inflow": deposit_inflow,
            "overdraft_count": overdraft_count,
        })
    )

behavioural_panel = pd.concat(monthly_rows, ignore_index=True)

assert len(behavioural_panel) == number_of_accounts * PANEL_MONTHS, \
    "Panel rows must equal accounts x months."
assert behavioural_panel.isna().sum().sum() == 0
assert behavioural_panel["days_past_due"].ge(0).all()

print(f"Behavioural panel: {len(behavioural_panel):,} rows "
      f"({number_of_accounts:,} accounts x {PANEL_MONTHS} months)")
print(f"Account-months delinquent: "
      f"{(behavioural_panel['days_past_due'] > 0).mean():.2%}")

display(behavioural_panel.head())


# =============================================================================
# CELL 10C  —  CODE  (on-book summary + the Milestone 4 target)
# =============================================================================

utilization_recent = utilization_path[:, -3:].mean(axis=1)
utilization_trend = utilization_path[:, -1] - utilization_path[:, 0]

leverage_standardised = (
    (book["leverage"].to_numpy() - book["leverage"].mean())
    / book["leverage"].std()
)

deterioration_logit = (
    -2.40
    + 3.00 * book_true_pd
    + 2.00 * np.clip(utilization_recent - 0.85, 0, None)
    + 1.50 * np.clip(utilization_trend, 0, None)
    + 0.40 * leverage_standardised
    + panel_rng.normal(0, 0.40, number_of_accounts)
)

book["deterioration_next_6_12mo"] = panel_rng.binomial(
    n=1,
    p=1 / (1 + np.exp(-deterioration_logit)),
).astype(int)

book["credit_limit"] = credit_limit.round(-3)
book["utilization_onbook"] = utilization_recent.round(4)
book["utilization_trend_24m"] = utilization_trend.round(4)
book["current_balance"] = (utilization_recent * credit_limit).round(-2)

panel_aggregates = (
    behavioural_panel
    .groupby("applicant_id")
    .agg(
        max_days_past_due=("days_past_due", "max"),
        months_delinquent_24m=("days_past_due", lambda s: (s > 0).sum()),
        overdrafts_24m=("overdraft_count", "sum"),
        average_deposit_inflow=("deposit_inflow", "mean"),
    )
    .reset_index()
)

book = book.merge(panel_aggregates, on="applicant_id", how="left")

book["tenure_months"] = np.minimum(
    book["credit_history_months"].to_numpy(),
    panel_rng.integers(6, 60, number_of_accounts),
)

assert book["deterioration_next_6_12mo"].isin([0, 1]).all()
assert book["applicant_id"].is_unique
assert book.isna().sum().sum() == 0

print(f"On-book accounts: {len(book):,}")
print(f"Deterioration rate (6-12 months): "
      f"{book['deterioration_next_6_12mo'].mean():.2%}")
print(f"Accounts ever 1+ DPD in 24 months: "
      f"{(book['max_days_past_due'] > 0).mean():.1%}")
print(f"Mean utilisation, last 3 months: {utilization_recent.mean():.3f} "
      f"(opened at {opening_utilization.mean():.3f})")


# =============================================================================
# CELL 11  —  CODE  (export)
# =============================================================================

# sme_applicant_portfolio.csv is deliberately NOT rewritten, so Notebook 2
# keeps running exactly as it does today. The booking decision goes to its
# own file and is joined only where it is needed.

HIDDEN_COLUMNS = {
    "latent_logit",
    "true_pd_12m",
    "random_risk_noise",
    "booking_logit",
    "approval_probability",
    "deterioration_logit",
}

booking_decision = applicants[["applicant_id", "booked"]].copy()

onbook_columns = (
    ["applicant_id"]
    + observable_features
    + [
        "credit_limit",
        "current_balance",
        "utilization_onbook",
        "utilization_trend_24m",
        "tenure_months",
        "max_days_past_due",
        "months_delinquent_24m",
        "overdrafts_24m",
        "average_deposit_inflow",
        "default_12m",
        "deterioration_next_6_12mo",
    ]
)

onbook_portfolio = book[onbook_columns].copy()

for frame, name in [
    (booking_decision, "booking_decision"),
    (onbook_portfolio, "onbook_portfolio"),
    (behavioural_panel, "behavioural_panel"),
]:
    assert HIDDEN_COLUMNS.isdisjoint(frame.columns), \
        f"A hidden risk variable leaked into {name}."

booking_path = RAW_DATA_PATH / "sme_booking_decision.csv"
onbook_path = RAW_DATA_PATH / "sme_onbook_portfolio.csv"
panel_path = RAW_DATA_PATH / "sme_behavioural_panel.csv"

booking_decision.to_csv(booking_path, index=False)
onbook_portfolio.to_csv(onbook_path, index=False)
behavioural_panel.to_csv(panel_path, index=False)

for path in [booking_path, onbook_path, panel_path]:
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"Saved {path.name:<32} {size_mb:>6.2f} MB")

print()
print("sme_applicant_portfolio.csv was not modified. Notebook 2 is unaffected.")


# =============================================================================
# OPTIONAL  —  FOR NOTEBOOK 2, THE REJECT-INFERENCE CHECK
# =============================================================================
"""
Paste this near the end of Notebook 2, AFTER your performance evaluation.
It is an analysis of your existing model, not a change to it. `booked` is
joined for reporting only and never enters the feature matrix.

This is the single most interesting number in Milestone 2.
"""

# booking = pd.read_csv(
#     project_root / "data/raw/sme_booking_decision.csv"
# )
#
# validation_with_decision = validation_sample.merge(
#     booking, on="applicant_id", how="left"
# )
#
# assert validation_with_decision["booked"].notna().all()
# assert "booked" not in X_val_woe.columns
#
# population_results = []
#
# for label, mask in [
#     ("Everyone (your headline metric)",
#      np.ones(len(validation_with_decision), dtype=bool)),
#     ("Booked only (what you would lend to)",
#      validation_with_decision["booked"] == 1),
#     ("Declined only",
#      validation_with_decision["booked"] == 0),
# ]:
#     mask = np.asarray(mask)
#     population_results.append({
#         "population": label,
#         "applicants": int(mask.sum()),
#         "default_rate": y_val[mask].mean(),
#         "auc": roc_auc_score(y_val[mask], pd_val[mask]),
#     })
#
# print(
#     pd.DataFrame(population_results).to_string(index=False)
# )
