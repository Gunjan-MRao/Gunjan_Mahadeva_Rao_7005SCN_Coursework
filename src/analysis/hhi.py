"""Phase 3: supplier-concentration analysis using the HHI.

Computes the Herfindahl-Hirschman Index monthly, by period, and by source for
trust-spend records with identified suppliers. Within each analytical group,
HHI = 10,000 × Σᵢsᵢ², where sᵢ is supplier i's expenditure share; source-level
estimates avoid conflating heterogeneous trust populations.
"""
from __future__ import annotations

import logging

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _hhi(group: pd.DataFrame) -> float:
    """Calculate supplier expenditure concentration for a single analytical group."""
    total = group["amount"].sum()
    if total <= 0:
        return float("nan")
    shares = group.groupby("supplier_norm")["amount"].sum() / total
    return float((shares ** 2).sum() * 10_000)


def compute_hhi(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Estimate monthly and period-source HHI series from trust-spend records."""
    if panel is None:
        panel = pd.read_csv(config.MASTER_PANEL_PATH, low_memory=False)
    trust = panel[panel["record_type"] == "trust_spend"].dropna(subset=["supplier_norm"])

    monthly = (
        trust.groupby(["year_month", "period"])
        .apply(_hhi, include_groups=False)
        .reset_index(name="hhi")
    )
    monthly_by_source = (
        trust.groupby(["year_month", "period", "source"])
        .apply(_hhi, include_groups=False)
        .reset_index(name="hhi")
    )
    monthly_by_source.to_csv(config.DATA_PROCESSED_DIR / "hhi_monthly_by_source.csv", index=False)
    by_period_source = (
        trust.groupby(["period", "source"])
        .apply(_hhi, include_groups=False)
        .reset_index(name="hhi")
    )

    monthly.to_csv(config.HHI_PATH, index=False)
    logger.info("Saved monthly HHI series -> %s", config.HHI_PATH)
    logger.info("HHI by period/source:\n%s", by_period_source.to_string(index=False))

    # Report a descriptive supplier-spend concentration snapshot.
    top_suppliers = (
        trust.groupby("supplier_norm")["amount"].sum().sort_values(ascending=False).head(10)
    )
    logger.info("Top 10 suppliers by total spend:\n%s", top_suppliers.to_string())

    return monthly, by_period_source, top_suppliers


if __name__ == "__main__":
    compute_hhi()
