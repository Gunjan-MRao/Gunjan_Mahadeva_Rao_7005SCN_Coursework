"""Phase 3: exploratory data analysis.

Constructs source-level data-quality diagnostics, period-specific expenditure
distributions, and monthly new-supplier entry rates. These descriptive outputs
characterise coverage, missingness, distributional heterogeneity, and changes
in procurement participation before formal modelling.
"""
from __future__ import annotations

import logging

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_panel() -> pd.DataFrame:
    """Load the master panel and standardise analytical data types."""
    panel = pd.read_csv(config.MASTER_PANEL_PATH, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce", format="mixed")
    panel["is_new_supplier"] = panel["is_new_supplier"].astype(bool)
    return panel


def data_quality_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarise source-specific coverage, missingness, and expenditure scale."""
    rows = []
    for source, grp in panel.groupby("source"):
        rows.append({
            "source": source,
            "n_rows": len(grp),
            "date_min": grp["date"].min(),
            "date_max": grp["date"].max(),
            "pct_null_supplier": grp["supplier"].isna().mean() * 100,
            "pct_null_category": grp["category"].isna().mean() * 100,
            "median_amount": grp["amount"].median(),
            "total_spend": grp["amount"].sum(),
        })
    report = pd.DataFrame(rows)
    logger.info("Data quality report:\n%s", report.to_string(index=False))
    return report


def spend_distribution_by_period(panel: pd.DataFrame) -> pd.DataFrame:
    """Describe trust-spend distributions across the study-period regimes."""
    trust = panel[panel["record_type"] == "trust_spend"]
    dist = trust.groupby("period")["amount"].describe(percentiles=[0.5, 0.9, 0.99])
    logger.info("Spend distribution (trust_spend) by period:\n%s", dist.to_string())
    return dist


def new_supplier_rate_by_month(panel: pd.DataFrame) -> pd.DataFrame:
    """Estimate the monthly share of trust-spend transactions from new suppliers."""
    trust = panel[panel["record_type"] == "trust_spend"].copy()
    monthly = trust.groupby("year_month").agg(
        n_transactions=("is_new_supplier", "size"),
        n_new_supplier_txns=("is_new_supplier", "sum"),
    )
    monthly["new_supplier_rate_pct"] = monthly["n_new_supplier_txns"] / monthly["n_transactions"] * 100
    monthly = monthly.reset_index().sort_values("year_month")

    # Map each monthly estimate to its predefined analytical period.
    period_lookup = trust.drop_duplicates("year_month").set_index("year_month")["period"]
    monthly["period"] = monthly["year_month"].map(period_lookup)

    period_avg = monthly.groupby("period")["new_supplier_rate_pct"].mean()
    logger.info("Average new-supplier rate (%%) by period:\n%s", period_avg.to_string())

    out_path = config.DATA_PROCESSED_DIR / "new_supplier_rate_by_month.csv"
    monthly.to_csv(out_path, index=False)
    logger.info("Saved new-supplier rate series -> %s", out_path)
    return monthly


def run_eda():
    panel = load_panel()
    dq = data_quality_report(panel)
    dq.to_csv(config.DATA_PROCESSED_DIR / "data_quality_report.csv", index=False)
    dist = spend_distribution_by_period(panel)
    dist.to_csv(config.DATA_PROCESSED_DIR / "spend_distribution_by_period.csv")
    new_supplier = new_supplier_rate_by_month(panel)
    return dq, dist, new_supplier


if __name__ == "__main__":
    run_eda()
