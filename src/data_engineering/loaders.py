"""Phase 2: source-specific loading and schema harmonisation.

The four consolidated public sources are mapped to a common procurement schema:
``source, entity, supplier, date, amount, category, sub_category, record_type,
transaction_id``. Source-specific handling centralises variation in field
names, date representations, and embedded header artefacts before analysis.
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
    """Parse Lincolnshire's ISO timestamps and UK-format dates separately.

    Explicit format partitioning prevents minority representations from being
    coerced to missing values; ``dayfirst=True`` reflects UK reporting practice.
    """
    s = series.astype(str).str.strip()
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    parsed.loc[iso_mask] = pd.to_datetime(s.loc[iso_mask], errors="coerce")
    parsed.loc[~iso_mask] = pd.to_datetime(s.loc[~iso_mask], errors="coerce", dayfirst=True)
    return parsed


def _clean_amount(series: pd.Series) -> pd.Series:
    """Coerce monetary strings and numerics to GBP after removing display symbols."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_bradford(path=None) -> pd.DataFrame:
    """Load Bradford expenditure records into the common procurement schema.

    Records lacking date, amount, or supplier fields are excluded because they
    cannot support transaction-level temporal or supplier analyses.
    """
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
    """Load United Lincolnshire Hospitals expenditure into the common schema.

    The consolidated FOI export retains repeated blank/header rows between
    monthly blocks, requiring content-based exclusion before date parsing.
    """
    path = path or config.RAW_FILES["lincolnshire"]
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["date", "amount", "supplier", "entity"]).copy()
    # Exclude embedded header labels retained as apparent supplier observations.
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
    """Load NHS England expenditure records into the common procurement schema.

    Records missing date, amount, or supplier fields are excluded before
    standardising identifiers and monetary values.
    """
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
    """Load health-sector Contracts Finder notices into the common schema.

    Only ``main.csv`` provides buyer and tender-value fields at notice level.
    Award and supplier blocks remain in the raw archive for subsequent
    provenance and network-oriented analyses.
    """
    path = path or config.RAW_FILES["contracts_finder"]
    df = pd.read_csv(path, low_memory=False)
    main = df[df["_source_file"] == "main.csv"].copy()
    main = main.dropna(subset=["date", "tender_value_amount", "buyer_name"])
    main = main[main["tender_value_amount"] > 0]

    out = pd.DataFrame({
        "source": "UK_Contracts_Finder",
        "entity": main["buyer_name"].str.strip(),
        "supplier": np.nan,  # Awarded suppliers are recorded separately in awards_suppliers.csv.
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
    """Load and concatenate all sources under the prescribed common schema."""
    frames = [
        load_bradford(),
        load_lincolnshire(),
        load_nhs_england(),
        load_contracts_finder(),
    ]
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[COMMON_COLUMNS]
    return panel
