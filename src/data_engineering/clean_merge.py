"""
Phase 2 (continued) — cleaning, quality filtering, feature-ready merge.

Takes the standardised panel from `loaders.load_all_sources()` and:
  1. restricts to the 2019-01-01 -> 2024-12-31 study window
  2. removes non-monetary / clearly erroneous rows (zero, negative, and
     extreme-outlier placeholder amounts)
  3. normalises supplier names for matching across sources
  4. assigns the pre-COVID / COVID / post-COVID period label
  5. flags first-time suppliers ("is_new_supplier") per entity
  6. computes a small set of anomaly-detection-ready numeric features

Running this module directly regenerates `data/processed/master_procurement_panel.csv`.
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
    conditions = [
        dates.between(config.PRE_COVID_START, config.PRE_COVID_END),
        dates.between(config.COVID_START, config.COVID_END),
        dates.between(config.POST_COVID_START, config.POST_COVID_END),
    ]
    choices = ["pre_covid", "covid", "post_covid"]
    return np.select(conditions, choices, default="out_of_scope")


def _normalise_supplier(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    s = s.str.replace(r"[^A-Z0-9& ]", "", regex=True)
    s = s.str.replace(r"\s+(LTD|LIMITED|PLC|LLP|INC)\.?$", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def clean_and_merge() -> pd.DataFrame:
    panel = load_all_sources()
    n_raw = len(panel)

    # 1. study window
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel[panel["date"].between(config.PRE_COVID_START, config.POST_COVID_END)]

    # 2. remove non-monetary / erroneous amounts
    panel = panel[panel["amount"].notna()]
    panel = panel[panel["amount"] > 0]
    # NHS trust invoices above £50m and contract notices above £2bn are treated as
    # data-entry / framework-ceiling artefacts rather than genuine single transactions
    panel = panel[
        ((panel["record_type"] == "trust_spend") & (panel["amount"] <= 5e7))
        | ((panel["record_type"] == "contract_notice") & (panel["amount"] <= 2e9))
    ]

    # 3. normalise supplier names (trust_spend rows only; contract notices have no supplier)
    panel["supplier_norm"] = np.where(
        panel["supplier"].notna(), _normalise_supplier(panel["supplier"]), np.nan
    )
    panel["entity_norm"] = panel["entity"].astype(str).str.upper().str.strip()

    # 4. period label
    panel["period"] = _assign_period(panel["date"])
    panel = panel[panel["period"] != "out_of_scope"]

    # 5. new-supplier flag: first time this supplier is seen for this entity.
    # A 6-month burn-in window (start of panel -> 2019-06-30) is excluded from the
    # flag because every supplier active in that window would otherwise look
    # "new" purely due to left-censoring (no earlier history exists in the data
    # to compare against), which would artificially inflate the pre-COVID rate.
    panel = panel.sort_values("date")
    trust_mask = panel["record_type"] == "trust_spend"
    panel["is_new_supplier"] = False
    trust_rows = panel[trust_mask].copy()
    first_seen = trust_rows.duplicated(subset=["entity_norm", "supplier_norm"], keep="first")
    burn_in_end = pd.Timestamp(config.PRE_COVID_START) + pd.DateOffset(months=6)
    trust_rows["is_new_supplier"] = (~first_seen) & (trust_rows["date"] > burn_in_end)
    panel.loc[trust_mask, "is_new_supplier"] = trust_rows["is_new_supplier"].values

    # 6. modelling features
    panel["log_amount"] = np.log1p(panel["amount"])
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel["year_month"] = panel["date"].dt.to_period("M").astype(str)
    panel["day_of_week"] = panel["date"].dt.dayofweek

    # supplier-level frequency & recency features (trust_spend only)
    panel = panel.sort_values(["entity_norm", "supplier_norm", "date"])
    grp = panel.groupby(["entity_norm", "supplier_norm"], dropna=False)
    panel["days_since_last_txn"] = grp["date"].diff().dt.days
    panel["supplier_txn_seq"] = grp.cumcount() + 1

    # category-level z-score of amount (within source + category), robust to scale differences
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
