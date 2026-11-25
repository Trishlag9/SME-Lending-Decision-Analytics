# Synthetic data generation and provenance

## Sources and execution order

1. `notebooks/01_portfolio_generation_and_audit.ipynb`, Sections 1–8, creates
   the applicant population and its synthetic default labels. Seed: 2026.
2. Its final cell executes `src/milestone1_booking_and_panel.py`, the recovered
   original add-on, in the same namespace. The add-on's numerical logic is
   unchanged. It requires the latent `true_pd_12m`, which is intentionally not
   in the exported applicant CSV; the applicant CSV alone cannot reconstruct it.
3. The add-on creates booking decisions, the monthly panel and the on-book
   dataset including the synthetic deterioration label. It never rewrites the
   applicant CSV.

| Output | Rows | Columns | Producer |
| --- | ---: | ---: | --- |
| `sme_applicant_portfolio.csv` | 15,000 | 22 | Notebook 01 Sections 1–8 |
| `sme_booking_decision.csv` | 15,000 | 2 | Original add-on |
| `sme_onbook_portfolio.csv` | 9,887 | 32 | Original add-on |
| `sme_behavioural_panel.csv` | 237,288 | 8 | Original add-on |

The applicant file has 20 predictors, one ID and one default label.

## Safe command-line execution

Install `requirements.txt` in your Python environment. From the repository root:

```bash
python scripts/reproduce_data.py
```

The runner executes the committed notebook in a clean namespace and a temporary
working directory. It compares generated CSVs with the committed `data/raw`
files without overwriting them. It compares row/column order, dtypes and values,
with `rtol=1e-12`, `atol=1e-8` for floating-point serialization differences.
IDs, labels and integers must also match under those tolerances. Exit code zero
indicates success; a generation or comparison failure produces a nonzero exit.
It does not run Notebooks 02–06 or retrain any models.

To retain a separately generated copy:

```bash
python scripts/reproduce_data.py --output-dir /tmp/sme-generated-data
```

This mode generates files but does not compare them with reference CSVs. The
destination must be new or empty; existing files are never overwritten.

Alternatively, restart Notebook 01's kernel and run all its cells sequentially.
**The interactive notebook writes to `data/raw`**, as it originally did. Use the
runner's default verification mode when you want to protect existing outputs.

## Historical booking simulation

Use a separate NumPy generator seeded with 2028 (`SEED + 2`). For applicant i:

```text
booking_logit_i = 2.20 - 6.00 * true_pd_i + Normal(0, 0.50)
approval_probability_i = sigmoid(booking_logit_i)
booked_i ~ Bernoulli(approval_probability_i)
```

This is a noisy risk-based simulation, not a decision made by the fitted
scorecard. It produces 9,887 booked and 5,113 not-booked applicants. Synthetic
default outcomes exist for both groups by construction. They are not actual
loan outcomes observed for rejected applications, and their availability does
not solve real-world reject inference. `booked` is not an application predictor.

## Monthly behaviour simulation

A new NumPy generator seeded with 2029 (`SEED + 3`) generates 24 observations,
month indices 0–23, for each booked account. Requested amount sets the limit.
Opening utilization is application utilization plus Normal(0, 0.05), clipped
to [0.02, 0.98]. For month m:

```text
drift_i = (true_pd_i - mean_book_true_pd) * 0.020
utilization_i,m = clip(opening_utilization_i
                      + (Normal(0, 0.030) + drift_i) * m, 0, 1.25)
```

There is a new noise draw each month. This is NOT a cumulative random walk.
Utilization above 100% is permitted intentionally, representing over-limit use.
Monthly series are generated as follows:

- Delinquency probability: clip(0.0015 + 0.070 * latent PD +
  0.035 * indicator(utilization > 0.90), 0, 0.60).
- When delinquent, DPD is sampled from 15, 30, 60 or 90 with weights
  0.50, 0.30, 0.14 and 0.06. Months are not generated as a persistent DPD
  transition process, so these series should not be treated as calibrated
  roll-rate histories.
- Deposit inflow: annual revenue / 12 * Uniform(0.60, 1.20), with a 30%
  reduction when utilization exceeds 0.95; rounded to the nearest 100.
- Overdraft count: Poisson(max(utilization - 0.80, 0) * 3).
- Balance: utilization * limit, rounded to the nearest 100. Limit is rounded
  to the nearest 1,000; exported utilization is rounded to four decimals.

See the script for the exact draw order, clipping and rounding. Changing the
random draw order changes the dataset, even if the seeds remain the same.

## Synthetic deterioration target

`deterioration_next_6_12mo` is generated for booked accounts as follows:

```text
recent_util_i = mean(unrounded utilization in months 21–23)
trend_i = unrounded utilization in month 23 - month 0
z_leverage_i = (leverage_i - book mean leverage) / book sample std(leverage)

deterioration_logit_i = -2.40 + 3.00 * true_pd_i
    + 2.00 * max(recent_util_i - 0.85, 0)
    + 1.50 * max(trend_i, 0)
    + 0.40 * z_leverage_i + Normal(0, 0.40)

deterioration_next_6_12mo_i ~ Bernoulli(sigmoid(deterioration_logit_i))
```

The same panel generator continues into target generation. The target has
1,959 positives and 7,928 negatives (19.81% positive rate).

Despite the column name, **no additional future months 25–36 are observed**.
This is a simulated prospective label linked to past behaviour and latent risk.
Held-out model metrics measure performance against that synthetic construction,
not independently observed future deterioration or intervention effectiveness.
The label and latent risk variables must not enter the feature matrix.

## Summary-field cautions

The on-book `current_balance` is based on the last-three-month mean utilization,
not necessarily the final month's balance. Notebook 05 deliberately uses month
23 from the panel for its latest balance and limit.

The original `tenure_months` is sampled from 6–59 and capped by credit-history
months; it is not reconciled with the full 24-month panel. It should not be used
to assert actual facility age. This repair preserves that historical generator
and documents the limitation rather than silently changing the existing data.

## Verification of this repair

The restored source was run against the committed repository datasets. All
four datasets matched in full, including booked IDs, 24 monthly rows per booked
account, all panel values and all 1,959 target positives. Existing data and
reports were not modified. Dependency versions used for that check are recorded
in `docs/data_reproduction_check.json`; future dependency changes may require
rechecking numerical equivalence.
