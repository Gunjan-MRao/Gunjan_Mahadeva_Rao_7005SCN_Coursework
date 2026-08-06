"""Phase 2b: analytical cleaning and procurement-panel construction.

Transforms the source-standardised records into the analysis panel by applying
the pre-specified observation window, monetary quality filters, supplier
normalisation, econometric period labels, and detection covariates. The
resulting panel is written to ``master_procurement_panel.csv``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.data_engineering.loaders import load_all_sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _assign_period(dates: pd.Series) -> pd.Series:
    """Assign mutually exclusive study periods using inclusive calendar boundaries."""
    conditions = [
        dates.between(config.PRE_COVID_START, config.PRE_COVID_END),
        dates.between(config.COVID_START, config.COVID_END),
        dates.between(config.POST_COVID_START, config.POST_COVID_END),
    ]
    choices = ["pre_covid", "covid", "post_covid"]
    return np.select(conditions, choices, default="out_of_scope")


def _normalise_supplier(series: pd.Series) -> pd.Series:
    """Canonicalise supplier names for entity-level longitudinal matching."""
    s = series.astype(str).str.upper().str.strip()
    s = s.str.replace(r"[^A-Z0-9& ]", "", regex=True)
    s = s.str.replace(r"\s+(LTD|LIMITED|PLC|LLP|INC)\.?$", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def clean_and_merge() -> pd.DataFrame:
    """Construct the cleaned, feature-ready procurement panel.

    The procedure applies documented inclusion criteria before generating
    longitudinal supplier and amount-distribution covariates for anomaly
    detection and period-based inference.
    """
    panel = load_all_sources()
    n_raw = len(panel)

    # 1. Restrict records to the pre-specified econometric observation window.
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel[panel["date"].between(config.PRE_COVID_START, config.POST_COVID_END)]

    # 2. Exclude null, non-positive, and implausibly large monetary values.
    panel = panel[panel["amount"].notna()]
    panel = panel[panel["amount"] > 0]
    # Values above £50m for trust invoices or £2bn for notices are treated as
    # data-entry errors or framework ceilings rather than single transactions.
    panel = panel[
        ((panel["record_type"] == "trust_spend") & (panel["amount"] <= 5e7))
        | ((panel["record_type"] == "contract_notice") & (panel["amount"] <= 2e9))
    ]

    # 3. Harmonise observed supplier strings; contract notices retain null suppliers.
    panel["supplier_norm"] = np.where(
        panel["supplier"].notna(), _normalise_supplier(panel["supplier"]), np.nan
    )
    panel["entity_norm"] = panel["entity"].astype(str).str.upper().str.strip()

    # 4. Assign the period indicator used in comparative analyses.
    panel["period"] = _assign_period(panel["date"])
    panel = panel[panel["period"] != "out_of_scope"]

    # 5. Flag the first observed supplier-entity occurrence in trust expenditure.
    # The six-month burn-in window (1 January–30 June 2019) avoids classifying
    # incumbents as new solely because pre-panel supplier history is unobserved.
    panel = panel.sort_values("date")
    trust_mask = panel["record_type"] == "trust_spend"
    panel["is_new_supplier"] = False
    trust_rows = panel[trust_mask].copy()
    first_seen = trust_rows.duplicated(subset=["entity_norm", "supplier_norm"], keep="first")
    burn_in_end = pd.Timestamp(config.PRE_COVID_START) + pd.DateOffset(months=6)
    trust_rows["is_new_supplier"] = (~first_seen) & (trust_rows["date"] > burn_in_end)
    panel.loc[trust_mask, "is_new_supplier"] = trust_rows["is_new_supplier"].values

    # 6. Generate covariates for anomaly-detection models.
    panel["log_amount"] = np.log1p(panel["amount"])
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel["year_month"] = panel["date"].dt.to_period("M").astype(str)
    panel["day_of_week"] = panel["date"].dt.dayofweek

    # Entity-supplier frequency and recency covariates, retaining null-supplier groups.
    panel = panel.sort_values(["entity_norm", "supplier_norm", "date"])
    grp = panel.groupby(["entity_norm", "supplier_norm"], dropna=False)
    panel["days_since_last_txn"] = grp["date"].diff().dt.days
    panel["supplier_txn_seq"] = grp.cumcount() + 1

    # Within-source/category standardisation controls for heterogeneous expenditure scales.
    cat_stats = panel.groupby(["source", "category"], dropna=False)["log_amount"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    )
    panel["amount_zscore_category"] = cat_stats

    panel = panel.reset_index(drop=True)
    panel.insert(0, "record_id", panel.index)

    logger.info(
        "Merge complete: %d raw rows -> %d rows after cleaning/filtering "
        "(%.1f%% retained)",
        n_raw, len(panel), 100 * len(panel) / n_raw,
    )
    logger.info("Rows by source:\n%s", panel["source"].value_counts().to_string())
    logger.info("Rows by period:\n%s", panel["period"].value_counts().to_string())

    panel.to_csv(config.MASTER_PANEL_PATH, index=False)
    logger.info("Saved master panel -> %s", config.MASTER_PANEL_PATH)
    return panel


if __name__ == "__main__":
    clean_and_merge()
