See [Project summary, results and limitations](PROJECT_SUMMARY.md).

## Reproduce the synthetic data

Install the dependencies in `requirements.txt`, then run from the repository root:

```bash
python scripts/reproduce_data.py
```

This executes Notebook 01 and the original Milestone 1 add-on in a temporary
directory and checks all four generated CSVs against `data/raw/`. Existing data
and reports are not overwritten. To generate a separate copy instead:

```bash
python scripts/reproduce_data.py --output-dir /tmp/sme-generated-data
```

The destination must be new or empty. See [data-generation documentation](docs/data_generation.md)
for the booking, monthly behaviour and synthetic deterioration formulas.
