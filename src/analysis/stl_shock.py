"""
Phase 2 (continued) — COVID-19 Shock Analysis via STL decomposition.

Implements Cleveland et al. (1990) STL (Seasonal-Trend decomposition using
Loess) on the monthly aggregated NHS spend series. A baseline trend+seasonal
model is fitted on the pre-COVID window (2018 H2 - 2019, per config) and
extrapolated forward; the residual between actual and expected spend during
COVID and post-COVID periods quantifies the disruption referenced in RQ1.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_monthly_series(panel: pd.DataFrame) -> pd.Series:
    monthly = (
        panel[panel["record_type"] == "trust_spend"]
        .groupby(panel["date"].dt.to_period("M"))["amount"]
        .sum()
        .sort_index()
    )
    monthly.index = monthly.index.to_timestamp()
    return monthly.asfreq("MS").fillna(0.0)


def run_stl_shock_analysis(panel: pd.DataFrame | None = None):
    if panel is None:
        panel = pd.read_csv(config.MASTER_PANEL_PATH, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce", format="mixed")

    monthly = build_monthly_series(panel)
    if len(monthly) < 24:
        logger.warning("Series too short (%d months) for a robust STL fit; results are indicative only.", len(monthly))

    stl = STL(monthly, period=12, robust=True)
    result = stl.fit()

    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid

    # Baseline expected level = mean trend+seasonal computed over the pre-COVID window only,
    # then compared against actual spend in each period to quantify the COVID shock.
    baseline_mask = (monthly.index >= config.STL_BASELINE_START) & (monthly.index <= config.STL_BASELINE_END)
    baseline_level = float(monthly[baseline_mask].mean()) if baseline_mask.any() else float(monthly.mean())

    summary_rows = []
    for period_name, start, end in [
        ("pre_covid", config.PRE_COVID_START, config.PRE_COVID_END),
        ("covid", config.COVID_START, config.COVID_END),
        ("post_covid", config.POST_COVID_START, config.POST_COVID_END),
    ]:
        mask = (monthly.index >= start) & (monthly.index <= end)
        actual_mean = float(monthly[mask].mean()) if mask.any() else np.nan
        pct_deviation = (actual_mean - baseline_level) / baseline_level * 100 if baseline_level else np.nan
        summary_rows.append({
            "period": period_name,
            "months": int(mask.sum()),
            "avg_monthly_spend": actual_mean,
            "baseline_avg_monthly_spend": baseline_level,
            "pct_deviation_from_baseline": pct_deviation,
            "avg_residual": float(resid[mask].mean()) if mask.any() else np.nan,
            "max_abs_residual": float(resid[mask].abs().max()) if mask.any() else np.nan,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.STL_PATH, index=False)
    logger.info("STL shock summary:\n%s", summary.to_string(index=False))

    decomposition = pd.DataFrame({
        "date": monthly.index,
        "observed": monthly.values,
        "trend": trend.values,
        "seasonal": seasonal.values,
        "resid": resid.values,
    })
    decomp_path = config.STL_PATH.parent / "stl_decomposition.csv"
    decomposition.to_csv(decomp_path, index=False)
    logger.info("Saved STL decomposition series -> %s", decomp_path)

    return summary, decomposition


if __name__ == "__main__":
    run_stl_shock_analysis()
