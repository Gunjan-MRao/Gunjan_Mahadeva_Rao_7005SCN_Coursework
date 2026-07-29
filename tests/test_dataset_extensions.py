"""
Tests for the Phase 6 dataset-extension modules: composite supplier risk
score, category-level deep dive, and robustness checks. Mirrors the style of
tests/test_advanced_modules.py -- small synthetic inputs, config paths
monkeypatched to tmp_path CSVs, no dependency on the full dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.analysis import supplier_risk_score
from src.analysis import category_deep_dive
from src.analysis import robustness_checks


# ---------------------------------------------------------------------------
# supplier_risk_score.py
# ---------------------------------------------------------------------------

def test_attach_hub_status_produces_clean_boolean_column(tmp_path, monkeypatch):
    # Regression test for the fillna(False)-without-astype(bool) bug: the
    # merged column must be a genuine bool dtype so that `~column` performs
    # logical negation, not bitwise-NOT on Python objects.
    network = pd.DataFrame({
        "node": ["SUP_A", "SUP_B", "TRUST_X"],
        "node_type": ["supplier", "supplier", "buyer"],
        "is_hub_supplier": [True, False, False],
    })
    path = tmp_path / "network_node_metrics.csv"
    network.to_csv(path, index=False)
    monkeypatch.setattr(config, "NETWORK_NODE_METRICS_PATH", path)

    suppliers = pd.DataFrame({"supplier": ["SUP_A", "SUP_B", "SUP_UNSEEN"]})
    result = supplier_risk_score.attach_hub_status(suppliers)

    assert result["is_hub_supplier"].dtype == bool
    # SUP_UNSEEN has no network row -> should fill to False, not NaN/object
    assert result.loc[result["supplier"] == "SUP_UNSEEN", "is_hub_supplier"].iloc[0] == False  # noqa: E712
    # Negation must behave logically (0/1), not bitwise (-1/-2)
    negated = ~result["is_hub_supplier"]
    assert set(negated.unique()).issubset({True, False})


def test_attach_hub_status_handles_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NETWORK_NODE_METRICS_PATH", tmp_path / "does_not_exist.csv")
    suppliers = pd.DataFrame({"supplier": ["SUP_A"]})
    result = supplier_risk_score.attach_hub_status(suppliers)
    assert result["is_hub_supplier"].isna().all()


def test_compute_composite_score_ranks_and_tiers_correctly(monkeypatch):
    monkeypatch.setattr(config, "SUPPLIER_RISK_MIN_TRANSACTIONS", 1)
    monkeypatch.setattr(config, "SUPPLIER_RISK_WEIGHTS", {
        "anomaly_rate": 1 / 3, "mean_anomaly_score": 1 / 3, "rule_flag_rate": 1 / 3,
    })
    monkeypatch.setattr(config, "SUPPLIER_RISK_TIER_THRESHOLDS", {"critical": 90, "high": 70, "medium": 40})

    # 10 suppliers with monotonically increasing risk signals -> the top
    # supplier should always land in "Critical" and the bottom in "Low".
    n = 10
    df = pd.DataFrame({
        "supplier": [f"SUP_{i}" for i in range(n)],
        "n_transactions": [5] * n,
        "anomaly_rate": np.linspace(0.0, 1.0, n),
        "mean_anomaly_score": np.linspace(0.0, 1.0, n),
        "rule_flag_rate": np.linspace(0.0, 1.0, n),
    })
    scored = supplier_risk_score.compute_composite_score(df)

    assert len(scored) == n
    top = scored.sort_values("composite_risk_score", ascending=False).iloc[0]
    bottom = scored.sort_values("composite_risk_score", ascending=False).iloc[-1]
    assert top["supplier"] == "SUP_9"
    assert top["risk_tier"] == "Critical"
    assert bottom["supplier"] == "SUP_0"
    assert bottom["risk_tier"] == "Low"
    assert top["composite_risk_score"] > bottom["composite_risk_score"]


def test_compute_composite_score_excludes_low_transaction_suppliers(monkeypatch):
    monkeypatch.setattr(config, "SUPPLIER_RISK_MIN_TRANSACTIONS", 5)
    monkeypatch.setattr(config, "SUPPLIER_RISK_WEIGHTS", {
        "anomaly_rate": 1 / 3, "mean_anomaly_score": 1 / 3, "rule_flag_rate": 1 / 3,
    })
    monkeypatch.setattr(config, "SUPPLIER_RISK_TIER_THRESHOLDS", {"critical": 95, "high": 85, "medium": 60})
    df = pd.DataFrame({
        "supplier": ["THIN", "THICK"],
        "n_transactions": [1, 10],
        "anomaly_rate": [0.9, 0.1],
        "mean_anomaly_score": [0.9, 0.1],
        "rule_flag_rate": [0.9, 0.1],
    })
    scored = supplier_risk_score.compute_composite_score(df)
    assert "THIN" not in scored["supplier"].values
    assert "THICK" in scored["supplier"].values


def test_compare_hub_vs_nonhub_risk_returns_empty_dict_with_one_group():
    df = pd.DataFrame({
        "composite_risk_score": [10.0, 20.0, 30.0],
        "is_hub_supplier": [False, False, False],
    })
    result = supplier_risk_score.compare_hub_vs_nonhub_risk(df)
    assert result == {}


def test_compare_hub_vs_nonhub_risk_computes_group_means():
    df = pd.DataFrame({
        "composite_risk_score": [90.0, 80.0, 10.0, 20.0, 15.0],
        "is_hub_supplier": [True, True, False, False, False],
    })
    result = supplier_risk_score.compare_hub_vs_nonhub_risk(df)
    assert result["n_hub"] == 2
    assert result["n_nonhub"] == 3
    assert result["mean_risk_score_hub"] == pytest.approx(85.0)
    assert result["mean_risk_score_nonhub"] == pytest.approx(15.0)
    assert result["mean_risk_score_hub"] > result["mean_risk_score_nonhub"]


# ---------------------------------------------------------------------------
# category_deep_dive.py
# ---------------------------------------------------------------------------

def test_hhi_perfect_monopoly_and_even_split():
    monopoly = pd.Series({"SUP_A": 1000.0})
    assert category_deep_dive._hhi(monopoly) == pytest.approx(10_000.0)

    even_split = pd.Series({"SUP_A": 250.0, "SUP_B": 250.0, "SUP_C": 250.0, "SUP_D": 250.0})
    # 4 equal shares of 25% each -> HHI = 4 * (25^2) = 2500
    assert category_deep_dive._hhi(even_split) == pytest.approx(2500.0)


def test_hhi_handles_zero_total():
    empty = pd.Series({"SUP_A": 0.0})
    assert np.isnan(category_deep_dive._hhi(empty))


def _tiny_category_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "category": ["PPE", "PPE", "PPE", "IT", "IT"],
        "period": ["pre_covid", "covid", "covid", "pre_covid", "covid"],
        "supplier_norm": ["SUP_1", "SUP_1", "SUP_2", "SUP_3", "SUP_3"],
        "amount": [100.0, 500.0, 500.0, 200.0, 220.0],
        "is_new_supplier": [False, False, True, False, False],
        "is_anomaly": [False, True, False, False, False],
    })


def test_build_category_period_table_computes_shares_and_hhi():
    trust = _tiny_category_frame()
    table = category_deep_dive.build_category_period_table(trust)

    # PPE/covid cell: SUP_1=500, SUP_2=500 -> even split -> HHI = 5000
    ppe_covid = table[(table["category"] == "PPE") & (table["period"] == "covid")].iloc[0]
    assert ppe_covid["n_transactions"] == 2
    assert ppe_covid["n_suppliers"] == 2
    assert ppe_covid["hhi"] == pytest.approx(5000.0)
    assert ppe_covid["new_supplier_rate_pct"] == pytest.approx(50.0)
    assert ppe_covid["anomaly_rate_pct"] == pytest.approx(50.0)

    # spend_share_within_period_pct sums to (approximately) 100 within each period
    for period, grp in table.groupby("period"):
        assert grp["spend_share_within_period_pct"].sum() == pytest.approx(100.0, abs=0.01)


def test_build_covid_shock_ranking_identifies_correct_growth_direction():
    trust = _tiny_category_frame()
    table = category_deep_dive.build_category_period_table(trust)
    ranking = category_deep_dive.build_covid_shock_ranking(table)

    ppe_row = ranking[ranking["category"] == "PPE"].iloc[0]
    it_row = ranking[ranking["category"] == "IT"].iloc[0]
    # PPE grew from 100 -> 1000 (900%); IT grew from 200 -> 220 (10%)
    assert ppe_row["pct_change_pre_to_covid"] == pytest.approx(900.0)
    assert it_row["pct_change_pre_to_covid"] == pytest.approx(10.0)
    assert ppe_row["pct_change_pre_to_covid"] > it_row["pct_change_pre_to_covid"]
    # ranking should be sorted descending by growth
    assert ranking.iloc[0]["category"] == "PPE"


# ---------------------------------------------------------------------------
# robustness_checks.py
# ---------------------------------------------------------------------------

def test_jaccard_identical_and_disjoint_sets():
    a = pd.Series([True, True, False, False])
    b = pd.Series([True, True, False, False])
    assert robustness_checks._jaccard(a, b) == pytest.approx(1.0)

    c = pd.Series([True, True, False, False])
    d = pd.Series([False, False, True, True])
    assert robustness_checks._jaccard(c, d) == pytest.approx(0.0)


def test_jaccard_empty_union_is_nan():
    a = pd.Series([False, False, False])
    b = pd.Series([False, False, False])
    assert np.isnan(robustness_checks._jaccard(a, b))


def test_assign_period_boundaries_are_inclusive_and_ordered():
    dates = pd.Series(pd.to_datetime(["2019-01-01", "2020-06-01", "2021-01-01", "2022-06-01"]))
    covid_start = pd.Timestamp("2020-03-23")
    covid_end = pd.Timestamp("2022-02-23")
    result = robustness_checks._assign_period(dates, covid_start, covid_end)
    assert list(result) == ["pre_covid", "covid", "covid", "post_covid"]


def test_threshold_sensitivity_produces_stable_period_ordering(tmp_path, monkeypatch):
    # Construct scores where covid period is clearly the most anomalous and
    # pre_covid the least, at every reasonable threshold -> ordering should
    # be reported as STABLE across all three percentiles.
    rng = np.random.RandomState(0)
    n_per_period = 300
    periods = (["pre_covid"] * n_per_period) + (["covid"] * n_per_period) + (["post_covid"] * n_per_period)
    scores = np.concatenate([
        rng.normal(0.1, 0.05, n_per_period),
        rng.normal(0.5, 0.05, n_per_period),
        rng.normal(0.3, 0.05, n_per_period),
    ])
    scores_df = pd.DataFrame({
        "record_id": range(len(periods)),
        "period": periods,
        "anomaly_score": scores,
    })
    scores_path = tmp_path / "anomaly_scores.csv"
    scores_df.to_csv(scores_path, index=False)

    validation_df = pd.DataFrame({
        "record_id": range(len(periods)),
        "rule_flagged": [False] * len(periods),
    })
    validation_path = tmp_path / "validation_redflags.csv"
    validation_df.to_csv(validation_path, index=False)

    monkeypatch.setattr(config, "ANOMALY_SCORES_PATH", scores_path)
    monkeypatch.setattr(config, "VALIDATION_PATH", validation_path)
    monkeypatch.setattr(config, "ROBUSTNESS_THRESHOLD_PERCENTILES", [90, 95, 99])

    result = robustness_checks.threshold_sensitivity()
    assert len(result) == 3
    # covid should have the highest anomaly rate at every threshold tested
    assert (result["rate_covid_pct"] >= result["rate_pre_covid_pct"]).all()
    assert (result["rate_covid_pct"] >= result["rate_post_covid_pct"]).all()
