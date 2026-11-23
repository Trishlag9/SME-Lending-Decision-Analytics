# SME Lending Decision Analytics

An educational demonstration of application credit scoring, lending policy,
illustrative pricing, early warning monitoring and proactive credit limit
increases using synthetic SME data.

Inspired by:
https://github.com/aayancheng/banking-analytics-workshop

This project adapts the workshop approach; it does not reproduce every
module or assumption.

## Project workflow

1. Generate and audit 15,000 synthetic SME applications.
2. Develop a logistic regression scorecard using nine WOE-transformed
   predictors.
3. Compare application policies and estimate illustrative loan pricing.
4. Train a LightGBM early warning model using borrower behaviour.
5. Simulate credit limit increases using eligibility, headroom and
   profitability rules.
6. Combine the saved results in a Streamlit borrower workbench.

## Data and evaluation populations

- Application development: 10,500 applicants.
- Application validation: 4,500 applicants.
- Historically booked accounts: 9,887.
- Behavioural panel: 237,288 monthly records, covering 24 observations
  per booked account.
- Early warning development: 6,920 accounts.
- Early warning test: 2,967 separate accounts.

Application and early warning splits are separate. The complete decision
pipeline has not been independently validated on a new population.

## Application policy

PD cutoffs were selected from development predictions:
- PD ≤ approximately 24.6795%: auto-approve.
- PD ≥ approximately 47.3969%: auto-decline.
- Otherwise: manual review.

The selected scenario targeted 65% auto-approval. It was a volume-based
illustrative choice, not a proven profit-maximizing policy.

On application validation:
- Historical booking: 66.09%, with 14.26% observed synthetic default rate.
- Proposed auto-approval: 65.69%, with 9.74% observed synthetic default rate.

This is a retrospective simulation, not evidence of a causal reduction
in defaults.

## Early warning results

Held-out test results:
- ROC-AUC: approximately 0.700.
- Average precision: approximately 0.379.
- Highest-risk 297 accounts captured 135 of 588 deterioration cases.
- Top-10% capture: 22.96%.
- Lift versus random selection: approximately 2.29×.

The deterioration target is synthetically generated using observed
behaviour and latent simulated risk. It is not an outcome observed in
additional future months.

Warning triggers describe behavioural signals; they are not model
attributions or causal explanations.

## Credit limit increase simulation

Among 2,967 assessed accounts:
- 22 passed initial screening.
- Two had positive headroom and passed base-case profitability.
- Total provisional additional limit: ₹450,000.
- One passed every tested downside scenario.

This module uses rules rather than a separate candidate prediction model.
No offers were issued and no post-increase outcomes were observed.

## Pricing assumptions

Application pricing:
- Expected loss rate = application PD × 45% LGD.
- Break-even rate = expected loss rate + 8% funding + 2% operating cost.
- Target rate = break-even rate + 2 percentage points.

Limit-increase economics:
- Interest rate: 16%.
- Additional drawdown: 50%.
- Funding cost: 8%.
- Operating cost: 2%.
- LGD: 45%.
- Allocated equity: 10% of additional drawn balance.
- Minimum pre-tax return on allocated equity: 15%.

These are simplified assumptions, not observed quotes or regulatory
capital calculations. Application PD and revenue are historical proxies
for the month-23 limit review. Deterioration probability is not default PD.

## Dashboard

From the repository root, with dependencies installed:

    .venv/bin/python -m streamlit run dashboard/streamlit_app.py

The dashboard displays application outputs for all 15,000 applicants.
Monitoring and limit-increase results cover the held-out monitoring
population only. Missing monitoring results mean “not assessed.”

Application pricing is displayed for historically booked accounts.
Displaying development-account results does not make them held-out
evaluation results.

## Verification completed

- Applicant ID uniqueness and join checks.
- Separate development and test accounts for early warning.
- Model save/reload prediction checks.
- Export/reload checks.
- Completed application outputs matched previously exported validation
  scores, decisions and pricing.
- Selected dashboard examples checked against notebook results.

## Differences from the workshop

- Different synthetic portfolio and sample sizes.
- Simplified application policy and pricing.
- Rule-based credit limit candidate selection.
- Different limit caps and profitability assumptions.
- Pre-tax incremental return rather than the workshop's fuller ROE framework.
- Additional downside sensitivity scenarios.

## Limitations

Synthetic results do not establish real-world predictive performance,
customer benefit or profitability. The project does not implement
production deployment, credit limit decreases, intervention-effect
measurement or a complete model-governance framework.