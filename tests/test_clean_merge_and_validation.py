"""
Tests for Phase 1 cleaning/feature logic and Phase 4 rule-based validation,
using small synthetic panels (not the real multi-hundred-MB dataset) so the
suite is fast and self-contained.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data_engineering import clean_merge
from src.validation import audit_validation


def _synthetic_standard_panel() -> pd.DataFrame:
    """A tiny panel already in the COMMON_COLUMNS schema, spanning pre-COVID,
    COVID and post-COVID dates with one supplier appearing repeatedly."""
    return pd.DataFrame({
        "source": ["NHS_England"] * 6,
        "entity": ["Test Trust"] * 6,
        "supplier": ["Acme Ltd", "Acme Ltd", "New Supplier Co", "Acme Ltd", "Acme Ltd", "Refund Corp"],
        "date": pd.to_datetime([
            "2019-06-01", "2019-09-01", "2020-06-01", "2020-07-01", "2022-06-01", "2019-06-15",
        ]),
        "amount": [1000.0, 1200.0, 500000.0, 1100.0, 1050.0, -50.0],
        "category": ["Supplies"] * 6,
        "sub_category": ["A"] * 6,
        "record_type": ["trust_spend"] * 6,
        "transaction_id": [f"T{i}" for i in range(6)],
    })


def test_clean_and_merge_drops_negative_and_out_of_range_amounts(monkeypatch, tmp_path):
    panel = _synthetic_standard_panel()
    monkeypatch.setattr(clean_merge, "load_all_sources", lambda: panel)
    out_path = tmp_path / "master.csv"
    monkeypatch.setattr(config, "MASTER_PANEL_PATH", out_path)

    result = clean_merge.clean_and_merge()

    # negative amount (refund) removed
    assert (result["amount"] > 0).all()
    assert "Refund Corp" not in result["supplier_norm"].values
    # no negative or zero amounts should survive
    assert not (result["amount"] <= 0).any()
    assert out_path.exists()


def test_new_supplier_flag_excludes_burn_in_window(monkeypatch, tmp_path):
    panel = _synthetic_standard_panel()
    monkeypatch.setattr(clean_merge, "load_all_sources", lambda: panel)
    monkeypatch.setattr(config, "MASTER_PANEL_PATH", tmp_path / "master.csv")

    result = clean_merge.clean_and_merge()

    # "Acme Ltd"'s very first transaction (2019-06-01) falls inside the
    # 6-month burn-in window -> must NOT be flagged as new (left-censoring).
    # note: supplier normalisation strips the "LTD" company-type suffix
    first_acme = result[result["supplier_norm"] == "ACME"].sort_values("date").iloc[0]
    assert first_acme["is_new_supplier"] == False  # noqa: E712

    # "New Supplier Co" first appears in 2020-06-01, well after burn-in,
    # and has no prior history -> SHOULD be flagged as new.
    new_supplier_row = result[result["supplier_norm"] == "NEW SUPPLIER CO"].iloc[0]
    assert new_supplier_row["is_new_supplier"] == True  # noqa: E712


def test_period_assignment_matches_config_boundaries(monkeypatch, tmp_path):
    panel = _synthetic_standard_panel()
    monkeypatch.setattr(clean_merge, "load_all_sources", lambda: panel)
    monkeypatch.setattr(config, "MASTER_PANEL_PATH", tmp_path / "master.csv")

    result = clean_merge.clean_and_merge()
    row_2019 = result[result["date"] == "2019-06-01"].iloc[0]
    row_2020 = result[result["date"] == "2020-06-01"].iloc[0]
    row_2022 = result[result["date"] == "2022-06-01"].iloc[0]

    assert row_2019["period"] == "pre_covid"
    assert row_2020["period"] == "covid"
    assert row_2022["period"] == "post_covid"


def test_flag_price_spike_detects_outlier_above_3x_median():
    panel = pd.DataFrame({
        "record_type": ["trust_spend"] * 5,
        "entity_norm": ["TRUST A"] * 5,
        "supplier_norm": ["SUPPLIER X"] * 5,
        "amount": [100.0, 110.0, 90.0, 105.0, 500.0],  # last one is a 3x+ spike
        "period": ["pre_covid"] * 4 + ["covid"],
        "category": ["Supplies"] * 5,
    })
    flags = audit_validation.flag_price_spike(panel)
    assert flags.iloc[4] == True  # noqa: E712
    assert not flags.iloc[:4].any()


def test_flag_round_amount_only_flags_exact_multiples_of_10k():
    panel = pd.DataFrame({
        "record_type": ["trust_spend"] * 4,
        "amount": [10000.0, 20000.0, 12345.67, 9999.0],
    })
    flags = audit_validation.flag_round_amount(panel)
    assert flags.tolist() == [True, True, False, False]
