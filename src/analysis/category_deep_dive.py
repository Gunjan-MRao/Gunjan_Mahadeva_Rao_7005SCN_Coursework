"""
Phase 6B — Category-level deep dive.

The headline STL/HHI/anomaly results (Phases 2-4) are reported at the
whole-panel level. This module disaggregates by spend `category` x `period`
to answer: which categories actually drove the aggregate COVID-19 spend
shock, and do anomaly/new-supplier/concentration patterns differ materially
across categories (e.g. PPE and medical-equipment categories are expected a
priori to show the largest COVID-period spend growth and supplier churn, per
the National Audit Office's 2020 review of the UK Government's PPE
procurement, HC 959).

For each (category, period) cell over trust_spend records, this computes:
  - total_spend, n_transactions, n_suppliers
  - spend_share_within_period_pct : the category's share of that period's
    total trust spend (lets us see e.g. "PPE went from 3% of spend pre-COVID
    to 22% of spend during COVID")
  - anomaly_rate_pct   : Isolation Forest flag rate within the cell
  - new_supplier_rate_pct : is_new_supplier flag rate within the cell
  - hhi                : Herfindahl-Hirschman Index of supplier concentration
                          within the cell (same HHI definition used
                          project-wide in Phase 2, i.e. sum of squared supplier
                          spend shares x 10,000; see Rhoades, 1993, "The
                          Herfindahl-Hirschman Index", Federal Reserve
                          Bulletin, 79, 188-189)

A companion ranking table (category_covid_shock_ranking.csv) sorts categories
by pre-COVID -> COVID spend growth, to directly identify the categories that
drove the aggregate STL shock (Phase 2) most.

Reference:
  National Audit Office (2020). "Investigation into government procurement
  during the COVID-19 pandemic." HC 959, Session 2019-2021.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MIN_CATEGORY_TRANSACTIONS = 20  # exclude near-empty categories from headline ranking (still in full CSV)


def load_trust_spend_with_flags() -> pd.DataFrame:
    panel = pd.read_csv(
        config.MASTER_PANEL_PATH, low_memory=False,
        usecols=["record_id", "record_type", "category", "supplier_norm", "amount", "period", "is_new_supplier"],
    )
    trust = panel[panel["record_type"] == "trust_spend"].copy()
    trust["category"] = trust["category"].fillna("Unspecified")
    trust["is_new_supplier"] = trust["is_new_supplier"].astype(bool)

    try:
        anomalies = pd.read_csv(config.ANOMALY_SCORES_PATH, low_memory=False, usecols=["record_id", "is_anomaly"])
        trust = trust.merge(anomalies, on="record_id", how="left")
        trust["is_anomaly"] = trust["is_anomaly"].fillna(False)
    except FileNotFoundError:
        logger.warning("Anomaly scores not found -- category deep dive will omit anomaly_rate_pct.")
        trust["is_anomaly"] = np.nan

    return trust


def _hhi(spend_by_supplier: pd.Series) -> float:
    total = spend_by_supplier.sum()
    if total <= 0:
        return np.nan
    shares = spend_by_supplier / total
    return float((shares ** 2).sum() * 10_000)


def build_category_period_table(trust: pd.DataFrame) -> pd.DataFrame:
    period_totals = trust.groupby("period")["amount"].sum()

    rows = []
    for (category, period), grp in trust.groupby(["category", "period"]):
        supplier_spend = grp.groupby("supplier_norm")["amount"].sum()
        rows.append({
            "category": category,
            "period": period,
            "n_transactions": len(grp),
            "n_suppliers": grp["supplier_norm"].nunique(),
            "total_spend": grp["amount"].sum(),
            "spend_share_within_period_pct": round(grp["amount"].sum() / period_totals[period] * 100, 3),
            "anomaly_rate_pct": round(grp["is_anomaly"].mean() * 100, 3) if grp["is_anomaly"].notna().any() else np.nan,
            "new_supplier_rate_pct": round(grp["is_new_supplier"].mean() * 100, 3),
            "hhi": round(_hhi(supplier_spend), 1),
        })
    table = pd.DataFrame(rows).sort_values(["category", "period"])
    logger.info(
        "Category x period deep dive built: %d categories x up to 3 periods = %d cells",
        table["category"].nunique(), len(table),
    )
    return table


def build_covid_shock_ranking(table: pd.DataFrame) -> pd.DataFrame:
    """Rank categories by pre-COVID -> COVID spend growth to identify which
    categories drove the aggregate STL shock (Phase 2) most."""
    pivot = table.pivot(index="category", columns="period", values="total_spend").fillna(0.0)
    n_txn = table.groupby("category")["n_transactions"].sum()

    for col in ("pre_covid", "covid", "post_covid"):
        if col not in pivot.columns:
            pivot[col] = 0.0

    pivot["pct_change_pre_to_covid"] = np.where(
        pivot["pre_covid"] > 0, (pivot["covid"] - pivot["pre_covid"]) / pivot["pre_covid"] * 100, np.nan,
    )
    pivot["pct_change_covid_to_post"] = np.where(
        pivot["covid"] > 0, (pivot["post_covid"] - pivot["covid"]) / pivot["covid"] * 100, np.nan,
    )
    pivot["n_transactions_total"] = n_txn
    pivot = pivot.reset_index().rename(columns={
        "pre_covid": "spend_pre_covid", "covid": "spend_covid", "post_covid": "spend_post_covid",
    })

    ranked = pivot[pivot["n_transactions_total"] >= MIN_CATEGORY_TRANSACTIONS].sort_values(
        "pct_change_pre_to_covid", ascending=False,
    )
    logger.info(
        "Top 5 categories by pre-COVID -> COVID spend growth (min %d transactions):\n%s",
        MIN_CATEGORY_TRANSACTIONS,
        ranked[["category", "spend_pre_covid", "spend_covid", "pct_change_pre_to_covid"]].head(5).to_string(index=False),
    )
    return pivot.sort_values("pct_change_pre_to_covid", ascending=False)


def run_category_deep_dive():
    trust = load_trust_spend_with_flags()
    table = build_category_period_table(trust)
    ranking = build_covid_shock_ranking(table)

    table.to_csv(config.CATEGORY_DEEP_DIVE_PATH, index=False)
    logger.info("Saved category x period deep dive -> %s", config.CATEGORY_DEEP_DIVE_PATH)
    ranking.to_csv(config.CATEGORY_SHOCK_RANKING_PATH, index=False)
    logger.info("Saved category COVID-shock ranking -> %s", config.CATEGORY_SHOCK_RANKING_PATH)

    return table, ranking


if __name__ == "__main__":
    run_category_deep_dive()
