"""Reproduce Milestone 1 in isolation; default to checking committed CSVs.

Run from any directory: python /path/to/repository/scripts/reproduce_data.py
Use --output-dir /path/to/empty-folder to retain a separately generated dataset.
Neither mode modifies the repository's existing data or reports.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_portfolio_generation_and_audit.ipynb"
FILES = (
    "sme_applicant_portfolio.csv",
    "sme_booking_decision.csv",
    "sme_onbook_portfolio.csv",
    "sme_behavioural_panel.csv",
)


def generate(isolated_root: Path) -> Path:
    """Execute the committed notebook and original add-on in a clean namespace."""
    source = ROOT / "src" / "milestone1_booking_and_panel.py"
    (isolated_root / "src").mkdir()
    shutil.copy2(source, isolated_root / "src" / source.name)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "display": lambda *args, **kwargs: None}
    prior_cwd = Path.cwd()
    log = io.StringIO()
    try:
        os.chdir(isolated_root)
        with contextlib.redirect_stdout(log):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                code = "".join(cell["source"])
                if code.strip():
                    exec(compile(code, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
    except Exception:
        print(log.getvalue())
        raise
    finally:
        os.chdir(prior_cwd)
        plt.close("all")

    raw = isolated_root / "data" / "raw"
    for name in FILES:
        if not (raw / name).is_file():
            raise RuntimeError(f"Notebook did not generate {name}")
    return raw


def compare(generated: Path) -> None:
    """Compare all rows/columns; permit only floating-point serialization noise."""
    for name in FILES:
        committed = ROOT / "data" / "raw" / name
        if not committed.is_file():
            raise FileNotFoundError(f"Missing reference dataset: {committed}")
        expected = pd.read_csv(committed)
        actual = pd.read_csv(generated / name)
        pd.testing.assert_frame_equal(
            actual, expected, check_exact=False, rtol=1e-12, atol=1e-8,
            check_dtype=True, check_like=False,
        )
        print(f"MATCH {name}: {len(actual):,} rows, {len(actual.columns)} columns")

    book = pd.read_csv(generated / "sme_onbook_portfolio.csv")
    panel = pd.read_csv(generated / "sme_behavioural_panel.csv")
    booking = pd.read_csv(generated / "sme_booking_decision.csv")
    assert book["applicant_id"].is_unique
    assert not panel.duplicated(["applicant_id", "month_index"]).any()
    assert panel.groupby("applicant_id")["month_index"].nunique().eq(24).all()
    assert sorted(panel["month_index"].unique().tolist()) == list(range(24))
    assert set(book["applicant_id"]) == set(panel["applicant_id"])
    assert set(book["applicant_id"]) == set(
        booking.loc[booking["booked"].eq(1), "applicant_id"]
    )
    assert book["deterioration_next_6_12mo"].isin([0, 1]).all()
    print(f"Booked accounts: {len(book):,}")
    print(f"Simulated deterioration cases: {book['deterioration_next_6_12mo'].sum():,}")
    print("All four datasets match. Existing repository data was not modified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Copy all four generated CSVs to an empty directory.")
    args = parser.parse_args()
    destination = args.output_dir.resolve() if args.output_dir is not None else None
    if destination is not None and destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            parser.error("--output-dir must be a new or empty directory; existing files are never overwritten.")

    with tempfile.TemporaryDirectory(prefix="sme-data-reproduction-") as temporary:
        generated = generate(Path(temporary))
        if destination is None:
            compare(generated)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            for name in FILES:
                shutil.copy2(generated / name, destination / name)
            print(f"Generated four CSVs in {destination}")
            print("Existing repository data was not modified. No reference comparison was requested.")


if __name__ == "__main__":
    main()
