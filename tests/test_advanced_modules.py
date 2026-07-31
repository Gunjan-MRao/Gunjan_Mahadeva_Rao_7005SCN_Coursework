"""
Tests for the Phase 6 dissertation-advancement modules: multi-method
comparison, synthetic anomaly injection, statistical significance testing,
and supplier-buyer network analysis. Uses small synthetic inputs so the
suite is fast and self-contained (no dependency on the full multi-hundred-MB
dataset).
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling import method_comparison
from src.modeling import synthetic_evaluation
from src.analysis import statistical_tests
from src.network import supplier_network


# ---------------------------------------------------------------------------
# method_comparison.py
# ---------------------------------------------------------------------------

def test_flag_top_percentile_flags_correct_fraction():
    scores = np.arange(100)  # 0..99
    flags = method_comparison._flag_top_percentile(scores, 90)
    # top 10% of 100 values -> the highest 10 flagged (approximately, percentile-based)
    assert flags.sum() == 10
    assert flags[-1] == True  # noqa: E712 (highest score always flagged)
    assert flags[0] == False  # noqa: E712 (lowest score never flagged)


def test_fit_isolation_forest_returns_higher_score_for_outlier():
    rng = np.random.RandomState(0)
    X_train = pd.DataFrame(rng.normal(0, 1, size=(200, 3)), columns=["a", "b", "c"])
    X_eval = pd.concat([X_train.iloc[:5], pd.DataFrame([[50, 50, 50]], columns=["a", "b", "c"])], ignore_index=True)
    scores = method_comparison.fit_isolation_forest(X_train, X_eval)
    # the injected extreme outlier (last row) should have the highest anomaly score
    assert scores[-1] == scores.max()


# ---------------------------------------------------------------------------
# synthetic_evaluation.py
# ---------------------------------------------------------------------------

def test_inject_invoice_inflation_increases_amount():
    row = pd.Series({"amount": 100.0, "supplier": "X"})
    rng = np.random.RandomState(42)
    mutated = synthetic_evaluation._inject_invoice_inflation(row, rng)
    assert mutated["amount"] > row["amount"] * 5 - 1e-6  # at least the lower bound of the inflation range
    assert mutated["injection_type"] == "invoice_inflation"


def test_inject_ghost_vendor_burst_sets_new_supplier_flags():
    row = pd.Series({"amount": 100.0, "is_new_supplier": 0, "supplier_txn_seq": 12, "days_since_last_txn": 45.0})
    rng = np.random.RandomState(42)
    mutated = synthetic_evaluation._inject_ghost_vendor_burst(row, rng, category_p90=1000.0)
    assert mutated["is_new_supplier"] == 1
    assert mutated["supplier_txn_seq"] == 1
    assert mutated["days_since_last_txn"] == 0.0
    assert mutated["amount"] >= 1000.0  # at least the category p90 baseline before the multiplier
    assert mutated["injection_type"] == "ghost_vendor_burst"


def test_inject_round_number_kickback_produces_round_amount():
    row = pd.Series({"amount": 12345.67})
    rng = np.random.RandomState(42)
    mutated = synthetic_evaluation._inject_round_number_kickback(row, rng)
    magnitude_options = [10_000, 25_000, 50_000, 100_000, 250_000]
    assert any(mutated["amount"] % m == 0 for m in magnitude_options)
    assert mutated["injection_type"] == "round_number_kickback"


# ---------------------------------------------------------------------------
# statistical_tests.py
# ---------------------------------------------------------------------------

def test_bootstrap_ci_on_constant_array_has_zero_width_interval():
    values = np.full(500, 7.0)
    point, lo, hi = statistical_tests._bootstrap_ci(values, np.mean, n_bootstrap=200, random_state=42)
    assert point == 7.0
    assert lo == pytest.approx(7.0, abs=1e-9)
    assert hi == pytest.approx(7.0, abs=1e-9)


def test_bootstrap_ci_interval_widens_with_more_variance():
    rng = np.random.RandomState(0)
    low_var = rng.normal(0, 1, size=1000)
    high_var = rng.normal(0, 10, size=1000)
    _, lo1, hi1 = statistical_tests._bootstrap_ci(low_var, np.mean, n_bootstrap=500, random_state=1)
    _, lo2, hi2 = statistical_tests._bootstrap_ci(high_var, np.mean, n_bootstrap=500, random_state=1)
    assert (hi2 - lo2) > (hi1 - lo1)


def test_hypergeometric_extreme_overlap_gives_tiny_p_value(tmp_path, monkeypatch):
    from src import config
    # Construct a validation table where ML flags are a perfect subset of rule flags
    # (maximal possible overlap given the margins) -> should be highly significant.
    n = 1000
    val = pd.DataFrame({
        "is_anomaly": [True] * 50 + [False] * 950,
        "rule_flagged": [True] * 200 + [False] * 800,
    })
    path = tmp_path / "validation_redflags.csv"
    val.to_csv(path, index=False)
    monkeypatch.setattr(config, "VALIDATION_PATH", path)

    result = statistical_tests.hypergeometric_triangulation_test()
    assert result["observed_overlap"] == 50
    assert result["p_value"] < 0.01 or result["log_p_value"] < np.log(0.01)


def test_hypergeometric_circularity_check_drops_r3_only(tmp_path, monkeypatch):
    from src import config
    # R3 (flag_new_supplier_large_covid) flags an extra record that no other
    # rule flags and that IS ML-flagged; the other three rules flag a
    # disjoint, larger set that overlaps ML more. Removing R3 should drop
    # the overlap by exactly 1 and leave K (rule-flagged count) unaffected
    # for the remaining rules.
    val = pd.DataFrame({
        "is_anomaly":                     [True, True, True, False, False, False, False, False, False, False],
        "flag_direct_award_covid":        [True, False, False, True, False, False, False, False, False, False],
        "flag_price_spike":               [False, True, False, False, True, False, False, False, False, False],
        "flag_new_supplier_large_covid":  [False, False, True, False, False, False, False, False, False, False],
        "flag_round_amount":              [False, False, False, False, False, True, False, False, False, False],
    })
    path = tmp_path / "validation_redflags.csv"
    val.to_csv(path, index=False)
    monkeypatch.setattr(config, "VALIDATION_PATH", path)

    df = statistical_tests.hypergeometric_triangulation_circularity_check()
    all_rules = df[df["scenario"] == "all_4_rules"].iloc[0]
    excl_r3 = df[df["scenario"] == "excl_r3_new_supplier_rule"].iloc[0]

    assert all_rules["observed_overlap"] == 3  # ML rows 0,1,2 all rule-flagged (by R1, R2, R3 respectively)
    assert excl_r3["observed_overlap"] == 2    # only ML rows 0,1 still corroborated once R3 is dropped
    assert excl_r3["n_ml_flagged"] == all_rules["n_ml_flagged"]  # ML flags themselves are untouched by which rules we compare against
    assert "new_supplier_large_covid" not in excl_r3["rule_cols_used"]


# ---------------------------------------------------------------------------
# synthetic_evaluation.py -- new_supplier_amount_baseline
# ---------------------------------------------------------------------------

def test_new_supplier_amount_baseline_only_scores_new_suppliers():
    X = pd.DataFrame({
        "is_new_supplier": [1, 0, 1, 0],
        "amount_zscore_category": [5.0, 5.0, -1.0, 99.0],
    })
    scores = synthetic_evaluation.fit_new_supplier_amount_baseline(X)
    assert scores[0] == 5.0
    assert scores[2] == -1.0
    # Established suppliers (is_new_supplier == 0) are floored to the sentinel
    # regardless of how extreme their amount is, so they can never be flagged.
    assert scores[1] == synthetic_evaluation.BASELINE_SENTINEL
    assert scores[3] == synthetic_evaluation.BASELINE_SENTINEL


def test_new_supplier_amount_baseline_ranks_new_suppliers_above_established():
    X = pd.DataFrame({
        "is_new_supplier": [0, 1],
        "amount_zscore_category": [1000.0, 0.1],  # established supplier has the bigger raw amount z-score
    })
    scores = synthetic_evaluation.fit_new_supplier_amount_baseline(X)
    # Despite the established supplier's far larger amount, only the new
    # supplier is even eligible to be flagged under this rule.
    assert scores[1] > scores[0]
    assert np.isfinite(scores).all()  # must stay finite for average_precision_score


# ---------------------------------------------------------------------------
# supplier_network.py
# ---------------------------------------------------------------------------

def _tiny_trust_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "entity_norm": ["TRUST_A", "TRUST_A", "TRUST_B", "TRUST_B", "TRUST_A", "TRUST_B"],
        "supplier_norm": ["SUP_1", "SUP_1", "SUP_1", "SUP_2", "SUP_2", "SUP_3"],
        "amount": [100.0, 200.0, 150.0, 50.0, 60.0, 10.0],
        "category": ["Cat_X"] * 6,
        "year_month": ["2020-01"] * 6,
    })


def test_build_bipartite_graph_edge_filtering_and_weights():
    trust = _tiny_trust_frame()
    import src.config as config_module
    orig = config_module.NETWORK_MIN_TRANSACTIONS
    config_module.NETWORK_MIN_TRANSACTIONS = 2
    try:
        G = supplier_network.build_bipartite_graph(trust)
    finally:
        config_module.NETWORK_MIN_TRANSACTIONS = orig

    # SUP_1-TRUST_A has 2 transactions (>=2, kept); SUP_3-TRUST_B has 1 (<2, dropped)
    assert G.has_edge("TRUST_A", "SUP_1")
    assert G["TRUST_A"]["SUP_1"]["weight"] == 2
    assert not G.has_node("SUP_3") or not G.has_edge("TRUST_B", "SUP_3")


def test_multi_trust_supplier_correctly_flagged_as_hub():
    G = nx.Graph()
    G.add_node("TRUST_A", node_type="buyer")
    G.add_node("TRUST_B", node_type="buyer")
    G.add_node("SUP_MULTI", node_type="supplier")
    G.add_node("SUP_SINGLE", node_type="supplier")
    G.add_edge("TRUST_A", "SUP_MULTI", weight=5, total_amount=500.0)
    G.add_edge("TRUST_B", "SUP_MULTI", weight=3, total_amount=300.0)
    G.add_edge("TRUST_A", "SUP_SINGLE", weight=2, total_amount=100.0)

    node_df = supplier_network.compute_bipartite_centrality(G)
    multi = node_df[node_df["node"] == "SUP_MULTI"].iloc[0]
    single = node_df[node_df["node"] == "SUP_SINGLE"].iloc[0]

    assert multi["is_multi_trust_supplier"] == True  # noqa: E712
    assert single["is_multi_trust_supplier"] == False  # noqa: E712
    assert multi["is_hub_supplier"] == True  # noqa: E712 (multi-trust suppliers are always hubs)


def test_cooccurrence_projection_links_suppliers_sharing_a_pool():
    trust = _tiny_trust_frame()
    G = supplier_network.build_cooccurrence_projection(trust)
    # Group (TRUST_A, Cat_X, 2020-01) = {SUP_1, SUP_2} -> linked
    assert G.has_edge("SUP_1", "SUP_2")
    # Group (TRUST_B, Cat_X, 2020-01) = {SUP_1, SUP_2, SUP_3} -> SUP_3 linked to both
    assert G.has_edge("SUP_1", "SUP_3")
    assert G.has_edge("SUP_2", "SUP_3")
    # SUP_1-SUP_2 co-occur in BOTH groups -> edge weight should be 2
    assert G["SUP_1"]["SUP_2"]["weight"] == 2
    # SUP_1-SUP_3 and SUP_2-SUP_3 co-occur in only 1 group -> weight 1
    assert G["SUP_1"]["SUP_3"]["weight"] == 1
