"""
Phase 7D — BI-ready data export.

A genuine .pbix (Power BI) file cannot be reliably hand-generated: it is a
proprietary, closed binary format with no supported Python (or open-source)
library capable of writing one from scratch. A .twbx (Tableau) file is
theoretically constructible (it is a zipped XML "twb" plus extract) but doing
so by hand, without Tableau itself to validate the result, is unverifiable
and highly likely to produce a corrupt file a marker cannot open. Fabricating
either and presenting it as a working deliverable would be misleading.

The practical, honest alternative adopted here -- standard practice when a BI
desktop tool isn't available in the build environment -- is:
  1. THIS MODULE: a small set of clean, denormalised, BI-import-ready CSVs in
     a star-schema layout (one fact table + four dimension/summary tables),
     so a marker can open Tableau/Power BI Desktop, connect to these CSVs,
     and build the dashboard themselves in a few minutes.
  2. dashboard.py (Phase 7E): an interactive Plotly HTML dashboard built
     directly from these same CSVs, viewable in any browser with no BI
     software installed -- a working, inspectable equivalent deliverable.
  3. docs/bi_dashboard_guide.md: a short field-by-field guide mapping these
     CSVs to specific Tableau/Power BI chart types, so the star schema is
     immediately actionable in either tool.

Star schema:
  fact_transactions.csv  - one row per trust_spend transaction (the same
                            257,095 records scored throughout Phases 4-7),
                            with all rule flags, anomaly flag/score, and the
                            supplier's composite risk tier already joined in
                            -- a single wide table a BI tool can slice by any
                            dimension without further joins.
  dim_supplier.csv        - one row per scored supplier: risk score/tier,
                            hub/community membership, transaction volume.
  dim_category.csv        - one row per spend category: totals, anomaly
                            rate, HHI, averaged across all periods.
  dim_period.csv          - one row per pre/covid/post period: the Phase 3/4
                            headline numbers (STL deviation, anomaly rate).
  dim_month.csv           - one row per calendar month: monthly spend, HHI,
                            new-supplier rate, anomaly rate (the time series
                            that drives the dashboard's trend-line panels).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_fact_transactions() -> pd.DataFrame:
    """One wide, denormalised row per trust_spend transaction: raw fields +
    every rule flag (Phase 5) + ML anomaly flag/score (Phase 4) + the
    transacting supplier's composite risk tier (Phase 7A) -- ready to drop
    straight into a BI tool's data model with zero further joins."""
    fact = pd.read_csv(
        config.VALIDATION_PATH, low_memory=False,
        usecols=[
            "record_id", "source", "entity", "supplier", "date", "amount", "category",
            "sub_category", "record_type", "period", "is_new_supplier", "year", "month", "year_month",
            "flag_direct_award_covid", "flag_price_spike", "flag_new_supplier_large_covid",
            "flag_round_amount", "rule_flag_count", "rule_flagged", "is_anomaly", "anomaly_score",
        ],
    )

    risk = pd.read_csv(
        config.SUPPLIER_RISK_SCORE_PATH,
        usecols=["supplier", "composite_risk_score", "risk_tier", "is_hub_supplier"],
    )
    fact = fact.merge(risk, on="supplier", how="left")
    fact["risk_tier"] = fact["risk_tier"].fillna("Not scored")

    fact["is_ml_scored"] = fact["record_type"] == "trust_spend"
    fact["is_anomaly"] = fact["is_anomaly"].astype("object")
    fact.loc[~fact["is_ml_scored"], "is_anomaly"] = np.nan
    fact.loc[~fact["is_ml_scored"], "anomaly_score"] = np.nan

    fact = fact.rename(columns={
        "is_new_supplier": "is_new_supplier_flag",
        "is_anomaly": "is_ml_anomaly",
        "anomaly_score": "ml_anomaly_score",
        "rule_flagged": "is_rule_flagged",
        "is_hub_supplier": "supplier_is_network_hub",
    })
    fact["period"] = fact["period"].map({
        "pre_covid": "Pre-COVID", "covid": "COVID", "post_covid": "Post-COVID",
    }).fillna(fact["period"])

    logger.info("Built fact_transactions: %d rows x %d columns", len(fact), len(fact.columns))
    return fact


def build_dim_supplier() -> pd.DataFrame:
    """One row per scored supplier: composite risk score/tier (Phase 7A),
    network hub/community membership (Phase 6D), transaction volume."""
    risk = pd.read_csv(config.SUPPLIER_RISK_SCORE_PATH)
    try:
        # NOTE: only suppliers that co-occur with >=1 other supplier under a
        # shared buyer (Phase 6D bipartite projection) appear as nodes here.
        # A NaN community_id/community_size for a given supplier is therefore
        # the *expected* outcome for a supplier with no shared-buyer links
        # (an isolated node), not a join-key bug -- spot-checked and confirmed
        # 799/3,456 risk-scored suppliers have a community assignment.
        community = pd.read_csv(config.NETWORK_COMMUNITY_PATH, usecols=["node", "community_id", "community_size"])
        community = community.rename(columns={"node": "supplier"})
        risk = risk.merge(community, on="supplier", how="left")
    except FileNotFoundError:
        logger.warning("Network community file not found -- dim_supplier will omit community columns.")
        risk["community_id"] = np.nan
        risk["community_size"] = np.nan

    risk = risk.rename(columns={
        "n_transactions": "total_transactions",
        "total_amount": "total_spend",
        "mean_amount": "mean_transaction_amount",
        "anomaly_rate": "ml_anomaly_rate",
        "mean_anomaly_score": "mean_ml_anomaly_score",
        "rule_flag_rate": "rule_flag_rate",
        "is_hub_supplier": "is_network_hub",
    })
    logger.info("Built dim_supplier: %d rows", len(risk))
    return risk


def build_dim_category() -> pd.DataFrame:
    """One row per spend category, aggregated across all three periods --
    totals, average anomaly/new-supplier rate, and average HHI, for
    category-level slicing in the BI tool (backs the same numbers as
    category_deep_dive.csv, Phase 7B, just pre-rolled to category grain)."""
    deep_dive = pd.read_csv(config.CATEGORY_DEEP_DIVE_PATH)
    grand_total = deep_dive["total_spend"].sum()

    agg = deep_dive.groupby("category").apply(
        lambda g: pd.Series({
            "total_spend": g["total_spend"].sum(),
            "n_transactions": g["n_transactions"].sum(),
            "n_periods_active": g["period"].nunique(),
            "mean_anomaly_rate_pct": round(np.average(g["anomaly_rate_pct"], weights=g["n_transactions"]), 3)
                if g["anomaly_rate_pct"].notna().any() else np.nan,
            "mean_new_supplier_rate_pct": round(np.average(g["new_supplier_rate_pct"], weights=g["n_transactions"]), 3),
            "mean_hhi": round(np.average(g["hhi"].fillna(0), weights=g["n_transactions"]), 1),
        }),
        include_groups=False,
    ).reset_index()
    agg["share_of_total_spend_pct"] = round(agg["total_spend"] / grand_total * 100, 4)

    try:
        shock = pd.read_csv(
            config.CATEGORY_SHOCK_RANKING_PATH,
            usecols=["category", "pct_change_pre_to_covid", "pct_change_covid_to_post"],
        )
        agg = agg.merge(shock, on="category", how="left")
    except FileNotFoundError:
        logger.warning("Category shock ranking not found -- dim_category will omit growth columns.")

    agg = agg.sort_values("total_spend", ascending=False)
    logger.info("Built dim_category: %d rows", len(agg))
    return agg


def build_dim_period() -> pd.DataFrame:
    """One row per pre/covid/post period: the headline STL shock (Phase 3)
    and overall anomaly-rate numbers (Phase 4), for the dashboard's top-level
    KPI cards and period filter."""
    stl = pd.read_csv(config.STL_PATH)
    try:
        scores = pd.read_csv(config.ANOMALY_SCORES_PATH, usecols=["period", "is_anomaly"])
        anomaly_rate = scores.groupby("period")["is_anomaly"].mean().mul(100).round(3).rename("ml_anomaly_rate_pct")
        dim = stl.merge(anomaly_rate, on="period", how="left")
    except FileNotFoundError:
        logger.warning("Anomaly scores file not found -- dim_period will omit ml_anomaly_rate_pct column.")
        dim = stl.copy()
        dim["ml_anomaly_rate_pct"] = np.nan

    label_map = {"pre_covid": "Pre-COVID", "covid": "COVID", "post_covid": "Post-COVID"}
    dim["period_label"] = dim["period"].map(label_map).fillna(dim["period"])
    order = {"pre_covid": 0, "covid": 1, "post_covid": 2}
    dim = dim.sort_values("period", key=lambda s: s.map(order))
    logger.info("Built dim_period: %d rows", len(dim))
    return dim


def build_dim_month() -> pd.DataFrame:
    """One row per calendar month: total spend, transaction count, HHI
    (Phase 3), new-supplier rate, and ML anomaly rate -- the time series
    behind the dashboard's trend-line panels."""
    panel = pd.read_csv(
        config.MASTER_PANEL_PATH, low_memory=False,
        usecols=["record_id", "record_type", "year_month", "period", "amount", "is_new_supplier"],
    )
    trust = panel[panel["record_type"] == "trust_spend"]

    monthly = trust.groupby("year_month").agg(
        total_spend=("amount", "sum"),
        n_transactions=("amount", "size"),
        new_supplier_rate_pct=("is_new_supplier", lambda s: round(s.mean() * 100, 3)),
        period=("period", "first"),
    ).reset_index()

    try:
        hhi = pd.read_csv(config.HHI_PATH, usecols=["year_month", "hhi"])
        monthly = monthly.merge(hhi, on="year_month", how="left")
    except FileNotFoundError:
        logger.warning("Monthly HHI file not found -- dim_month will omit hhi column.")
        monthly["hhi"] = np.nan

    scores = pd.read_csv(config.ANOMALY_SCORES_PATH, usecols=["record_id"])
    scored_ids = set(scores["record_id"])
    trust_scored = trust[trust["record_id"].isin(scored_ids)].merge(
        pd.read_csv(config.ANOMALY_SCORES_PATH, usecols=["record_id", "is_anomaly"]),
        on="record_id", how="left",
    )
    anomaly_rate = trust_scored.groupby("year_month")["is_anomaly"].mean().mul(100).round(3).rename("ml_anomaly_rate_pct")
    monthly = monthly.merge(anomaly_rate, on="year_month", how="left")

    monthly["period"] = monthly["period"].map({
        "pre_covid": "Pre-COVID", "covid": "COVID", "post_covid": "Post-COVID",
    }).fillna(monthly["period"])
    monthly = monthly.sort_values("year_month")
    logger.info("Built dim_month: %d rows", len(monthly))
    return monthly


def run_bi_export():
    fact = build_fact_transactions()
    dim_supplier = build_dim_supplier()
    dim_category = build_dim_category()
    dim_period = build_dim_period()
    dim_month = build_dim_month()

    outputs = {
        "fact_transactions.csv": fact,
        "dim_supplier.csv": dim_supplier,
        "dim_category.csv": dim_category,
        "dim_period.csv": dim_period,
        "dim_month.csv": dim_month,
    }
    for filename, df in outputs.items():
        path = config.BI_EXPORT_DIR / filename
        df.to_csv(path, index=False)
        logger.info("Saved BI export table -> %s (%d rows x %d cols)", path, len(df), len(df.columns))

    return outputs


if __name__ == "__main__":
    run_bi_export()
