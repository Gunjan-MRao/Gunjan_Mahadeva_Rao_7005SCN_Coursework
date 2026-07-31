"""
Phase 3/4/5 extension — Statistical significance testing.

The original pipeline reported headline percentages (STL shock %, new-supplier
rate, anomaly rate by period, ML/rule triangulation overlap) as point
estimates only. This module adds the significance/uncertainty layer expected
of a Masters-level quantitative dissertation:

  1. Exact hypergeometric test on the ML/rule-based triangulation overlap
     (equivalent to a one-sided Fisher's exact test / permutation test on
     fixed-margin 2x2 counts) — is 62.7% overlap significantly higher than
     chance, given the rule-flagged base rate?
  2. Percentile bootstrap confidence intervals on the three key period-level
     descriptive statistics: Isolation Forest anomaly rate, new-supplier
     rate, and STL % deviation from baseline, by period.
  3. Mann-Whitney U tests comparing the full Isolation Forest anomaly-score
     distribution between period pairs (pre-COVID vs COVID, pre-COVID vs
     post-COVID, COVID vs post-COVID) — a non-parametric test appropriate
     for the heavily right-skewed anomaly-score distribution.
  4. Triangulation circularity check: rule R3 (`flag_new_supplier_large_covid`)
     and the Isolation Forest feature set share an input (`is_new_supplier`,
     plus the closely related `supplier_txn_seq`), so part of the headline
     59.6% ML/rule overlap could in principle be a mechanical consequence of
     that shared feature rather than independent corroboration. The
     hypergeometric triangulation test is therefore re-run on rule flags
     R1+R2+R4 only (R3 excluded) to check whether the overlap survives
     without the one rule that is not feature-independent of the ML model.

All tests read from the CSV artefacts already produced by earlier pipeline
stages, so this module must run AFTER data_engineering, eda, stl_shock, the
Isolation Forest model, and audit_validation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def hypergeometric_triangulation_test() -> dict:
    """Exact test: given N total records, K of which are rule-flagged, and n
    ML-flagged, what is the probability of observing >= the actual overlap
    purely by chance if ML flags were assigned independently of rule flags?
    This is mathematically equivalent to a one-sided permutation test that
    randomly redraws which n records are "ML-flagged" without replacement,
    but computed exactly via the hypergeometric distribution rather than by
    Monte Carlo simulation."""
    val = pd.read_csv(config.VALIDATION_PATH, usecols=["is_anomaly", "rule_flagged"], low_memory=False)
    N = len(val)
    K = int(val["rule_flagged"].sum())
    n = int(val["is_anomaly"].sum())
    observed_overlap = int((val["is_anomaly"] & val["rule_flagged"]).sum())

    expected_overlap_by_chance = n * K / N
    # P(X >= observed) under Hypergeometric(N, K, n)
    p_value = stats.hypergeom.sf(observed_overlap - 1, N, K, n)
    log_p_value = stats.hypergeom.logsf(observed_overlap - 1, N, K, n)  # avoids float underflow to 0.0

    result = {
        "test": "hypergeometric_triangulation",
        "N_total": N, "K_rule_flagged": K, "n_ml_flagged": n,
        "observed_overlap": observed_overlap,
        "expected_overlap_under_chance": round(expected_overlap_by_chance, 1),
        "overlap_rate_pct": round(observed_overlap / n * 100, 2) if n else np.nan,
        "chance_overlap_rate_pct": round(expected_overlap_by_chance / n * 100, 2) if n else np.nan,
        "p_value": p_value,
        "log_p_value": log_p_value,
        "p_value_lt": "1e-300" if p_value == 0.0 else str(p_value),
    }
    logger.info(
        "Hypergeometric test: observed overlap=%d vs expected-by-chance=%.1f (N=%d, K=%d, n=%d) -> log(p)=%.1f (p is astronomically small; underflows float64 to 0.0)",
        observed_overlap, expected_overlap_by_chance, N, K, n, log_p_value,
    )
    return result


RULE_COLS_ALL = ["flag_direct_award_covid", "flag_price_spike", "flag_new_supplier_large_covid", "flag_round_amount"]
# R3 (flag_new_supplier_large_covid) is excluded here because it and the
# Isolation Forest feature set both key on the supplier's new/incumbent
# status (is_new_supplier directly, and supplier_txn_seq — the SHAP-dominant
# feature — indirectly, since a low sequence number is definitionally what
# "new" means). R1 (direct-award/contract), R2 (price spike vs. historical
# median) and R4 (round amount) do not use supplier tenure at all.
RULE_COLS_EXCL_R3 = ["flag_direct_award_covid", "flag_price_spike", "flag_round_amount"]


def _hypergeometric_test_for_rule_set(val: pd.DataFrame, rule_cols: list[str], scenario: str) -> dict:
    """Exact hypergeometric overlap test between `is_anomaly` (ML) and the
    union of the given rule-flag columns. Shared logic behind both the
    headline (all-rules) test and the R3-excluded circularity-check variant."""
    rule_flagged = val[rule_cols].any(axis=1)
    N = len(val)
    K = int(rule_flagged.sum())
    n = int(val["is_anomaly"].sum())
    observed_overlap = int((val["is_anomaly"] & rule_flagged).sum())

    expected_overlap_by_chance = n * K / N
    p_value = stats.hypergeom.sf(observed_overlap - 1, N, K, n)
    log_p_value = stats.hypergeom.logsf(observed_overlap - 1, N, K, n)

    result = {
        "scenario": scenario, "rule_cols_used": "+".join(c.replace("flag_", "") for c in rule_cols),
        "N_total": N, "K_rule_flagged": K, "n_ml_flagged": n,
        "observed_overlap": observed_overlap,
        "expected_overlap_under_chance": round(expected_overlap_by_chance, 1),
        "overlap_rate_pct": round(observed_overlap / n * 100, 2) if n else np.nan,
        "chance_overlap_rate_pct": round(expected_overlap_by_chance / n * 100, 2) if n else np.nan,
        "p_value": p_value,
        "log_p_value": log_p_value,
        "p_value_lt": "1e-300" if p_value == 0.0 else str(p_value),
    }
    logger.info(
        "[%s] rules=%s: observed overlap=%d (%.2f%% of ML flags) vs expected-by-chance=%.1f (%.2f%%) "
        "(N=%d, K=%d, n=%d) -> log(p)=%.1f",
        scenario, result["rule_cols_used"], observed_overlap, result["overlap_rate_pct"],
        expected_overlap_by_chance, result["chance_overlap_rate_pct"], N, K, n, log_p_value,
    )
    return result


def hypergeometric_triangulation_circularity_check() -> pd.DataFrame:
    """Re-runs the ML/rule triangulation hypergeometric test twice: once with
    all four audit red-flag rules (reproduces the headline 59.6% figure as a
    sanity check) and once with R3 (`flag_new_supplier_large_covid`) removed,
    since R3 shares the `is_new_supplier` input with the Isolation Forest
    feature set and is therefore not a feature-independent corroboration
    source. If the overlap rate survives (remains far above the chance rate
    and remains highly significant) with R3 excluded, the triangulation
    result is not simply an artefact of the shared feature."""
    val = pd.read_csv(config.VALIDATION_PATH, usecols=["is_anomaly"] + RULE_COLS_ALL, low_memory=False)
    for col in RULE_COLS_ALL:
        val[col] = val[col].astype(bool)
    val["is_anomaly"] = val["is_anomaly"].astype(bool)

    rows = [
        _hypergeometric_test_for_rule_set(val, RULE_COLS_ALL, "all_4_rules"),
        _hypergeometric_test_for_rule_set(val, RULE_COLS_EXCL_R3, "excl_r3_new_supplier_rule"),
    ]
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, statistic_fn, n_bootstrap: int, random_state: int) -> tuple[float, float, float]:
    rng = np.random.RandomState(random_state)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = values[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic_fn(sample)
    point = statistic_fn(values)
    lo, hi = np.percentile(boot_stats, [2.5, 97.5])
    return point, lo, hi


def bootstrap_anomaly_rate_by_period() -> pd.DataFrame:
    scores = pd.read_csv(config.ANOMALY_SCORES_PATH, usecols=["period", "is_anomaly"], low_memory=False)
    rows = []
    for period, grp in scores.groupby("period"):
        vals = grp["is_anomaly"].astype(float).values
        point, lo, hi = _bootstrap_ci(vals, np.mean, config.N_BOOTSTRAP, config.RANDOM_STATE)
        rows.append({"metric": "isolation_forest_anomaly_rate_pct", "period": period,
                     "point_estimate_pct": round(point * 100, 3), "ci_lower_pct": round(lo * 100, 3), "ci_upper_pct": round(hi * 100, 3)})
    return pd.DataFrame(rows)


def bootstrap_new_supplier_rate_by_period() -> pd.DataFrame:
    panel = pd.read_csv(config.MASTER_PANEL_PATH, usecols=["record_type", "period", "is_new_supplier"], low_memory=False)
    trust = panel[panel["record_type"] == "trust_spend"]
    rows = []
    for period, grp in trust.groupby("period"):
        vals = grp["is_new_supplier"].astype(float).values
        point, lo, hi = _bootstrap_ci(vals, np.mean, config.N_BOOTSTRAP, config.RANDOM_STATE)
        rows.append({"metric": "new_supplier_rate_pct", "period": period,
                     "point_estimate_pct": round(point * 100, 3), "ci_lower_pct": round(lo * 100, 3), "ci_upper_pct": round(hi * 100, 3)})
    return pd.DataFrame(rows)


def bootstrap_stl_deviation_by_period() -> pd.DataFrame:
    """Block-limited bootstrap: resamples the (small number of) monthly
    observations within each period. NOTE: monthly spend is autocorrelated,
    so this simple percentile bootstrap likely understates true uncertainty
    versus a full block-bootstrap or ARIMA-based interval; reported as an
    indicative interval, with this limitation documented in the write-up."""
    decomp = pd.read_csv(config.STL_PATH.parent / "stl_decomposition.csv", parse_dates=["date"])
    summary = pd.read_csv(config.STL_PATH)
    baseline = float(summary["baseline_avg_monthly_spend"].iloc[0])

    period_bounds = {
        "pre_covid": (config.PRE_COVID_START, config.PRE_COVID_END),
        "covid": (config.COVID_START, config.COVID_END),
        "post_covid": (config.POST_COVID_START, config.POST_COVID_END),
    }
    rows = []
    for period, (start, end) in period_bounds.items():
        mask = (decomp["date"] >= start) & (decomp["date"] <= end)
        vals = decomp.loc[mask, "observed"].values
        stat_fn = lambda arr: (arr.mean() - baseline) / baseline * 100
        point, lo, hi = _bootstrap_ci(vals, stat_fn, config.N_BOOTSTRAP, config.RANDOM_STATE)
        rows.append({"metric": "stl_pct_deviation_from_baseline", "period": period,
                     "point_estimate_pct": round(point, 2), "ci_lower_pct": round(lo, 2), "ci_upper_pct": round(hi, 2),
                     "n_months": int(mask.sum())})
    return pd.DataFrame(rows)


def mann_whitney_anomaly_score_by_period() -> pd.DataFrame:
    scores = pd.read_csv(config.ANOMALY_SCORES_PATH, usecols=["period", "anomaly_score"], low_memory=False)
    pairs = [("pre_covid", "covid"), ("pre_covid", "post_covid"), ("covid", "post_covid")]
    rows = []
    for a, b in pairs:
        x = scores.loc[scores["period"] == a, "anomaly_score"].values
        y = scores.loc[scores["period"] == b, "anomaly_score"].values
        u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
        # rank-biserial effect size
        n1, n2 = len(x), len(y)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        rows.append({
            "test": "mann_whitney_u", "group_a": a, "group_b": b,
            "median_a": float(np.median(x)), "median_b": float(np.median(y)),
            "u_statistic": float(u_stat), "p_value": p_value,
            "rank_biserial_effect_size": round(effect_size, 4),
            "n_a": n1, "n_b": n2,
        })
    return pd.DataFrame(rows)


def run_statistical_tests() -> pd.DataFrame:
    logger.info("Running hypergeometric triangulation significance test...")
    hyper_result = hypergeometric_triangulation_test()

    logger.info("Running triangulation circularity check (all rules vs. R3-excluded)...")
    circularity_df = hypergeometric_triangulation_circularity_check()

    logger.info("Running bootstrap confidence intervals (n=%d resamples each)...", config.N_BOOTSTRAP)
    boot_anomaly = bootstrap_anomaly_rate_by_period()
    boot_new_supplier = bootstrap_new_supplier_rate_by_period()
    boot_stl = bootstrap_stl_deviation_by_period()

    logger.info("Running Mann-Whitney U tests on anomaly-score distributions across periods...")
    mwu = mann_whitney_anomaly_score_by_period()

    logger.info("Bootstrap CI — Isolation Forest anomaly rate by period:\n%s", boot_anomaly.to_string(index=False))
    logger.info("Bootstrap CI — new-supplier rate by period:\n%s", boot_new_supplier.to_string(index=False))
    logger.info("Bootstrap CI — STL %% deviation from baseline by period:\n%s", boot_stl.to_string(index=False))
    logger.info("Mann-Whitney U tests (anomaly score, period pairs):\n%s", mwu.to_string(index=False))

    bootstrap_all = pd.concat([boot_anomaly, boot_new_supplier, boot_stl], ignore_index=True)
    bootstrap_all.to_csv(config.STATS_RESULTS_PATH, index=False)
    mwu.to_csv(str(config.STATS_RESULTS_PATH).replace(".csv", "_mannwhitney.csv"), index=False)
    pd.DataFrame([hyper_result]).to_csv(str(config.STATS_RESULTS_PATH).replace(".csv", "_hypergeometric.csv"), index=False)
    circularity_df.to_csv(str(config.STATS_RESULTS_PATH).replace(".csv", "_hypergeometric_circularity_check.csv"), index=False)
    logger.info(
        "Saved statistical test results -> %s (+ _mannwhitney / _hypergeometric / _hypergeometric_circularity_check variants)",
        config.STATS_RESULTS_PATH,
    )

    return bootstrap_all, mwu, hyper_result, circularity_df


if __name__ == "__main__":
    run_statistical_tests()
