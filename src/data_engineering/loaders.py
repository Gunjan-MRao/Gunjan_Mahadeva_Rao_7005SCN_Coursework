"""
Phase 1 — Data Engineering.

Loads the four raw source files and standardises each into a common schema:

    source, entity, supplier, date, amount, category, sub_category,
    record_type, transaction_id

Each `load_*` function is source-specific because the four public datasets
were published with different column names, date formats and (in the case of
United Lincolnshire Hospitals) badly-parsed multi-row headers baked into the
CSV during the original FOI export. Centralising the quirks here means every
downstream stage (EDA, STL, Isolation Forest, SHAP) can work off one clean
schema instead of re-discovering these issues.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COMMON_COLUMNS = [
    "source", "entity", "supplier", "date", "amount",
    "category", "sub_category", "record_type", "transaction_id",
]


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    """United Lincolnshire Hospitals' raw export mixes ISO timestamps
    ('2019-01-10 00:00:00.0') with UK-format dates ('05/09/2019') across
    different monthly source files — a concrete data-quality issue this
    project's RQ1 investigates. Parse both formats explicitly rather than
    letting a single `pd.to_datetime` call silently coerce the minority
    format to NaT."""
    s = series.astype(str).str.strip()
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    parsed.loc[iso_mask] = pd.to_datetime(s.loc[iso_mask], errors="coerce")
    parsed.loc[~iso_mask] = pd.to_datetime(s.loc[~iso_mask], errors="coerce", dayfirst=True)
    return parsed


def _clean_amount(series: pd.Series) -> pd.Series:
    """Convert '48,523.68'-style strings (and floats) into numeric GBP."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_bradford(path=None) -> pd.DataFrame:
    path = path or config.RAW_FILES["bradford"]
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["date", "amount", "supplier"]).copy()

    out = pd.DataFrame({
        "source": "Bradford_Teaching_Hospitals",
        "entity": df["entity"].fillna("Bradford Teaching Hospitals NHS Foundation Trust"),
        "supplier": df["supplier"].str.strip().str.upper(),
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "amount": _clean_amount(df["amount"]),
        "category": df["expense_type"],
        "sub_category": df["expense_area"],
        "record_type": "trust_spend",
        "transaction_id": df["transaction_number"].astype(str),
    })
    logger.info("Bradford: loaded %d clean rows (raw had %d)", len(out), len(pd.read_csv(path)))
    return out


def load_lincolnshire(path=None) -> pd.DataFrame:
    """United Lincolnshire Hospitals — FOI export contains repeated blank/header
    rows between monthly blocks, so rows must be filtered on real content
    rather than trusted at face value."""
    path = path or config.RAW_FILES["lincolnshire"]
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["date", "amount", "supplier", "entity"]).copy()
    # drop stray repeated header rows that survived cleaning (e.g. "SUPPLIER" as a value)
    df = df[~df["supplier"].astype(str).str.fullmatch(r"(?i)supplier|expense.*", na=False)]

    out = pd.DataFrame({
        "source": "United_Lincolnshire_Hospitals",
        "entity": "United Lincolnshire Hospitals NHS Trust",
        "supplier": df["supplier"].str.strip().str.upper(),
        "date": _parse_mixed_dates(df["date"]),
        "amount": _clean_amount(df["amount"]),
        "category": df["expense_type"],
        "sub_category": df["expense_area"],
        "record_type": "trust_spend",
        "transaction_id": np.nan,
    })
    logger.info("Lincolnshire: loaded %d clean rows (raw had %d)", len(out), len(pd.read_csv(path)))
    return out


def load_nhs_england(path=None) -> pd.DataFrame:
    path = path or config.RAW_FILES["nhs_england"]
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["date", "amount", "supplier"]).copy()

    out = pd.DataFrame({
        "source": "NHS_England",
        "entity": df["entity"].fillna("NHS England"),
        "supplier": df["supplier"].str.strip().str.upper(),
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "amount": _clean_amount(df["amount"]),
        "category": df["expense_type"],
        "sub_category": df["expense_area"],
        "record_type": "trust_spend",
        "transaction_id": df["transaction_number"].astype(str),
    })
    logger.info("NHS England: loaded %d clean rows (raw had %d)", len(out), len(pd.read_csv(path)))
    return out


def load_contracts_finder(path=None) -> pd.DataFrame:
    """UK Contracts Finder OCDS export. Only the `main.csv` block carries a
    usable buyer + tender value at the contract-notice level; `awards.csv`
    and `awards_suppliers.csv` blocks are sparse joins kept out of the
    anomaly-detection panel but retained in raw form for supplier-network
    follow-up analysis."""
    path = path or config.RAW_FILES["contracts_finder"]
    df = pd.read_csv(path, low_memory=False)
    main = df[df["_source_file"] == "main.csv"].copy()
    main = main.dropna(subset=["date", "tender_value_amount", "buyer_name"])
    main = main[main["tender_value_amount"] > 0]

    out = pd.DataFrame({
        "source": "UK_Contracts_Finder",
        "entity": main["buyer_name"].str.strip(),
        "supplier": np.nan,  # buyer-side notice; awarded supplier lives in awards_suppliers.csv
        "date": pd.to_datetime(main["date"], errors="coerce", utc=True).dt.tz_localize(None),
        "amount": pd.to_numeric(main["tender_value_amount"], errors="coerce"),
        "category": main["tender_procurementMethod"],
        "sub_category": main["tender_mainProcurementCategory"],
        "record_type": "contract_notice",
        "transaction_id": main["tender_id"].astype(str),
    })
    logger.info("Contracts Finder: loaded %d clean rows (raw main.csv had %d)", len(out), len(main))
    return out


def load_all_sources() -> pd.DataFrame:
    frames = [
        load_bradford(),
        load_lincolnshire(),
        load_nhs_england(),
        load_contracts_finder(),
    ]
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[COMMON_COLUMNS]
    return panel
