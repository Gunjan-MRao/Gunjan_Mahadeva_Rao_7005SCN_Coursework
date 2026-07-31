"""
Tests for the Phase 2 data-engineering pipeline. Uses small synthetic CSVs
(rather than the real 100MB+ raw files) so the suite runs in seconds and
does not depend on data being present, per FOI/Contracts Finder provenance
notes in the README.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_engineering import loaders


@pytest.fixture
def tmp_bradford_csv(tmp_path):
    path = tmp_path / "bradford_clean.csv"
    df = pd.DataFrame({
        "_dataset_source": ["bradford"] * 4,
        "department_family": ["Finance"] * 4,
        "entity": ["Bradford Teaching Hospitals NHS Foundation Trust"] * 4,
        "date": ["2019-03-01", "2019-03-02", "2020-04-01", "2021-05-01"],
        "expense_type": ["Supplies", "Supplies", "PPE", "Drugs"],
        "expense_area": ["Ward A", "Ward A", "Ward B", "Pharmacy"],
        "supplier": ["Acme Ltd", "Acme Ltd", "Beta Supplies", "Gamma Pharma"],
        "transaction_number": ["T1", "T2", "T3", "T4"],
        "amount": [1000.0, 2000.0, 5000.0, 3000.0],
        "vat_number": [np.nan] * 4,
        "invoice_number": [np.nan] * 4,
        "_source_file": ["jan.csv"] * 4,
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tmp_lincolnshire_csv(tmp_path):
    path = tmp_path / "lincolnshire_clean.csv"
    df = pd.DataFrame({
        "_dataset_source": ["lincolnshire"] * 3,
        "_source_file": ["jan.csv"] * 3,
        "entity": ["Department Of Health"] * 3,
        # deliberately mixed ISO and UK dd/mm/yyyy formats, matching the real
        # raw file's quirk that this loader must handle correctly
        "date": ["2019-01-10 00:00:00.0", "05/09/2019", "17/10/2019"],
        "expense_type": ["Supplies", "Supplies", "Drugs"],
        "expense_area": ["A", "B", "C"],
        "supplier": ["Delta Co", "Epsilon Ltd", "Zeta Health"],
        "amount": ["1,000.50", "2,500.00", "3,750.25"],
        "month": ["Jan-19", "Sep-19", "Oct-19"],
        "PUBLISHED AREA\t\t\t\t\t\t\t": [np.nan] * 3,
    })
    df.to_csv(path, index=False)
    return path


def test_bradford_loader_drops_null_rows_and_parses_amount(tmp_bradford_csv):
    out = loaders.load_bradford(tmp_bradford_csv)
    assert len(out) == 4
    assert set(out.columns) == set(loaders.COMMON_COLUMNS)
    assert out["amount"].dtype.kind == "f"
    assert out["source"].unique().tolist() == ["Bradford_Teaching_Hospitals"]
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_lincolnshire_loader_handles_mixed_date_formats_and_comma_amounts(tmp_lincolnshire_csv):
    out = loaders.load_lincolnshire(tmp_lincolnshire_csv)
    assert len(out) == 3
    # all three dates must parse successfully despite mixed ISO / UK formats
    assert out["date"].notna().all()
    assert sorted(out["date"].dt.year.unique().tolist()) == [2019]
    assert sorted(out["date"].dt.month.unique().tolist()) == [1, 9, 10]
    # comma-formatted string amounts converted to numeric
    assert np.isclose(out["amount"].sum(), 1000.50 + 2500.00 + 3750.25)


def test_clean_amount_handles_currency_symbols_and_commas():
    s = pd.Series(["£1,234.56", "48,523.68", "100", np.nan])
    out = loaders._clean_amount(s)
    assert np.isclose(out.iloc[0], 1234.56)
    assert np.isclose(out.iloc[1], 48523.68)
    assert np.isclose(out.iloc[2], 100.0)
    assert pd.isna(out.iloc[3])
