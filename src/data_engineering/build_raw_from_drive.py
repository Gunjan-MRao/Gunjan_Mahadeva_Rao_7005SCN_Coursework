"""Phase 1: raw-data acquisition and source consolidation.

Acquires the public FOI and Contracts Finder archive and consolidates it into
the four source-specific inputs consumed by Phase 2 (`loaders.py`):

    data/raw/bradford_clean.csv
    data/raw/lincolnshire_clean.csv
    data/raw/nhs_england_clean.csv
    data/raw/contracts_clean.csv

The historical `_clean` suffix denotes consolidated raw data, not analytical
cleaning. This module preserves source lineage while harmonising headers;
missing-value treatment, mixed-format date parsing, monetary coercion, and
supplier normalisation are intentionally deferred to `loaders.py`.

Run standalone with:
    python -m src.data_engineering.build_raw_from_drive [--force]
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source-archive structure
# ---------------------------------------------------------------------------
# Maps each Drive top-level directory to its internal source identifier.
SOURCE_FOLDERS = {
    "NHS Bradford teaching hospitals": "bradford",
    "NHS England": "nhs_england",
    "United Lincolnshire Hospitals": "lincolnshire",
    "NHS UK Contracts": "contracts",
}

# Only these Contracts Finder annual-export files contain fields required by
# downstream analyses; excluding ancillary files reduces acquisition volume.
CONTRACTS_WANTED_FILES = {"main.csv", "awards.csv", "awards_suppliers.csv"}

SPREADSHEET_SUFFIXES = {".xls", ".xlsx", ".xlsm"}

# Public Drive access is rate-limited; pacing and back-off mitigate incomplete
# acquisition of the approximately 500 MB archival corpus.
DOWNLOAD_MAX_ATTEMPTS = 4
DOWNLOAD_RETRY_BACKOFF_S = 60
DOWNLOAD_PACING_S = 0.5

# ---------------------------------------------------------------------------
# Health-sector relevance filter for the national Contracts Finder export
# ---------------------------------------------------------------------------
# The archive contains the full UK Contracts Finder export, whereas the study
# population is health-sector procurement. Keyword matching reconstructs an
# approximate health-sector subset and is not equivalent to the original query.
NHS_KEYWORDS = (
    r"nhs|health|hospital|clinical|ambulance|blood|commissioning support|"
    r"clinical commissioning|integrated care board|hospice|primary care|care trust"
)
_NHS_RE = re.compile(NHS_KEYWORDS, re.IGNORECASE)

# Searchable fields differ by Contracts Finder file: buyer/tender text in
# main.csv, descriptions in awards.csv, and supplier names in awards_suppliers.csv.
CONTRACTS_TEXT_COLUMNS = {
    "main.csv": ["buyer_name", "tender_title", "tender_description"],
    "awards.csv": ["description"],
    "awards_suppliers.csv": ["name"],
}

# ---------------------------------------------------------------------------
# Raw-to-intermediate schema mapping
# ---------------------------------------------------------------------------
# Header spellings vary by year and publisher; token normalisation supports
# deterministic mapping to the intermediate schema.
SPEND_COLUMN_ALIASES = {
    "department_family": ["department family", "department of health", "department"],
    "entity": ["entity"],
    "date": ["date", "date of payment", "payment date", "date paid", "transaction date"],
    # Lincolnshire changes the expenditure-category header across years; NHS
    # England adds the suffix " for Report" from November 2024.
    "expense_type": [
        "expense type", "expense type for report", "expenditure type",
        "nature of spend", "9an level 9 account name",
    ],
    "expense_area": ["expense area", "expense area for report"],
    "supplier": ["supplier", "supplier name"],
    "transaction_number": ["transaction number", "transaction no"],
    "amount": ["ap amount", "amount", "ap amount gbp", "analysed gross"],
    "vat_number": ["vat registration number", "vat number", "vat registration no"],
    "invoice_number": ["purchase invoice number", "invoice number", "purchase invoice no"],
    "month": ["month"],
}

# Minimum alias matches required to identify a header row. A threshold avoids
# classifying Lincolnshire data rows containing "Department Of Health" as headers.
HEADER_ALIAS_MIN_HITS = 3

BRADFORD_COLUMNS = [
    "_dataset_source", "department_family", "entity", "date", "expense_type",
    "expense_area", "supplier", "transaction_number", "amount", "vat_number",
    "invoice_number", "_source_file",
]
NHS_ENGLAND_COLUMNS = BRADFORD_COLUMNS

# Lincolnshire publishes the same eight fields with and without a header row.
LINCOLNSHIRE_POSITIONAL_COLUMNS = [
    "department_family", "entity", "date", "expense_type",
    "expense_area", "supplier", "amount", "month",
]
LINCOLNSHIRE_COLUMNS = [
    "_dataset_source", "_source_file", "entity", "date", "expense_type",
    "expense_area", "supplier", "amount", "month",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_header(name: object) -> str:
    """Canonicalise a raw header for case-insensitive alias matching."""
    text = re.sub(r"[^0-9a-z]+", " ", str(name).lower())
    return re.sub(r"\s+", " ", text).strip()


_ALIAS_LOOKUP = {
    alias: canon for canon, aliases in SPEND_COLUMN_ALIASES.items() for alias in aliases
}


def _build_rename_map(columns) -> dict:
    """Map recognised raw headers to canonical intermediate-schema fields."""
    rename = {}
    for col in columns:
        canon = _ALIAS_LOOKUP.get(_normalise_header(col))
        if canon and canon not in rename.values():
            rename[col] = canon
    return rename


def _header_alias_hits(cells) -> int:
    """Count cells matching recognised spend-file header aliases."""
    return sum(1 for c in cells if _normalise_header(c) in _ALIAS_LOOKUP)


def _looks_like_header(cells) -> bool:
    return _header_alias_hits(cells) >= HEADER_ALIAS_MIN_HITS


def _sniff_delimiter(path: Path) -> str:
    """Infer the delimiter because the May 2019 Lincolnshire CSV is tab-delimited."""
    with open(path, encoding="latin1") as fh:
        head = "".join(next(fh, "") for _ in range(5))
    return "\t" if head.count("\t") > head.count(",") else ","


def _read_csv_any_encoding(path: Path, **kwargs) -> pd.DataFrame:
    """Read FOI exports under their observed, heterogeneous character encodings.

    Sequential decoding preserves monthly coverage when UTF-8 and legacy
    Windows/Latin-1 encodings coexist within a source archive.
    """
    kwargs.setdefault("sep", _sniff_delimiter(path))
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", encoding_errors="replace", low_memory=False, **kwargs)


def _find_header_row(probe: pd.DataFrame) -> int:
    """Locate the schema header within an initial file probe.

    Some 2024 Bradford exports prepend a report-title row, so row zero cannot
    be assumed to contain field names.
    """
    for idx in range(len(probe)):
        if _looks_like_header(probe.iloc[idx].tolist()):
            return idx
    return 0


def _read_spend_file(path: Path) -> pd.DataFrame | None:
    """Read a monthly CSV or workbook after identifying its schema header."""
    try:
        if path.suffix.lower() in SPREADSHEET_SUFFIXES:
            # Bradford January 2019 is available only as .xlsm; legacy .xls files
            # require pandas to select the compatible reader engine.
            engine = "openpyxl" if path.suffix.lower() in {".xlsx", ".xlsm"} else None
            probe = pd.read_excel(path, header=None, nrows=6, dtype=str, engine=engine)
            return pd.read_excel(path, header=_find_header_row(probe), dtype=str, engine=engine)
        probe = _read_csv_any_encoding(path, header=None, nrows=6, dtype=str)
        return _read_csv_any_encoding(path, header=_find_header_row(probe), dtype=str)
    except Exception as exc:  # noqa: BLE001 - one unreadable month must not abort the build
        logger.warning("  ! could not read %s: %s", path.name, exc)
        return None


def _iter_source_files(staging_dir: Path, folder_name: str):
    """Yield non-hidden source files in deterministic path order."""
    root = Path(staging_dir) / folder_name
    if not root.is_dir():
        logger.warning("Staging folder missing: %s", root)
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            yield path


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _select_drive_files(entries) -> list:
    """Select non-duplicated source files required for consolidation.

    For spend data, CSV is preferred to a co-located workbook, except where
    only a workbook exists. For Contracts Finder, the retained files provide
    notice, award, and supplier fields used in subsequent analyses.
    """
    by_month: dict[tuple, list] = {}
    selected = []
    for entry in entries:
        parts = Path(entry.path).parts
        if len(parts) < 2 or SOURCE_FOLDERS.get(parts[0]) is None:
            continue
        name = parts[-1]
        suffix = Path(name).suffix.lower()

        if SOURCE_FOLDERS[parts[0]] == "contracts":
            if name in CONTRACTS_WANTED_FILES:
                selected.append(entry)
            continue

        if suffix != ".csv" and suffix not in SPREADSHEET_SUFFIXES:
            continue
        # Group co-located CSV/workbook representations of one reporting month.
        by_month.setdefault((parts[:-1], Path(name).stem.lower()), []).append(entry)

    for candidates in by_month.values():
        csvs = [e for e in candidates if Path(e.path).suffix.lower() == ".csv"]
        selected.append(csvs[0] if csvs else candidates[0])

    return selected


def download_drive_folder(dest_dir=None) -> Path:
    """Acquire the selected public archive subset into ``dest_dir``.

    Existing non-empty files are retained, enabling resumable acquisition.
    Individual retrieval failures are logged rather than terminating the
    complete data-ingestion stage.
    """
    import gdown  # Deferred dependency: required only by the acquisition stage.

    dest_dir = Path(dest_dir or config.RAW_STAGING_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Listing Google Drive archive: %s", config.GOOGLE_DRIVE_RAW_FOLDER_URL)
    entries = gdown.download_folder(
        config.GOOGLE_DRIVE_RAW_FOLDER_URL, output=str(dest_dir), skip_download=True, quiet=True
    )
    if not entries:
        raise RuntimeError(
            "Could not list the Google Drive folder. Check the link is still shared publicly, "
            "or place the four data/raw/*_clean.csv files manually (see README)."
        )

    wanted = _select_drive_files(entries)
    logger.info("Drive archive holds %d files; %d needed for this pipeline", len(entries), len(wanted))

    n_downloaded = n_cached = 0
    failures = []
    for entry in wanted:
        target = dest_dir / Path(entry.path)
        if target.exists() and target.stat().st_size > 0:
            n_cached += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if _download_one(gdown, entry, target):
            n_downloaded += 1
        else:
            failures.append(entry.path)
        time.sleep(DOWNLOAD_PACING_S)

    logger.info(
        "Download complete: %d fetched, %d already present, %d failed",
        n_downloaded, n_cached, len(failures),
    )
    if failures:
        logger.warning(
            "%d file(s) could not be downloaded and are excluded from the consolidated "
            "output: %s. Re-run this module to retry them: already-downloaded files are "
            "skipped, so a second pass only fetches the gaps.",
            len(failures), ", ".join(failures[:10]) + (" ..." if len(failures) > 10 else ""),
        )
    return dest_dir


def _download_via_gdown(gdown, entry, target: Path) -> None:
    gdown.download(id=entry.id, output=str(target), quiet=True)


def _download_direct(entry, target: Path) -> None:
    """Retrieve a Drive file through its direct download endpoint.

    This route avoids the metadata request that is commonly rate-limited under
    anonymous access and complements the ``gdown`` fallback.
    """
    import requests  # Installed transitively with the gdown acquisition dependency.

    url = f"https://drive.usercontent.google.com/download?id={entry.id}&export=download&confirm=t"
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
    with requests.get(url, headers=headers, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        # HTML responses indicate an interstitial or quota notice, not file content.
        if "text/html" in resp.headers.get("Content-Type", ""):
            raise RuntimeError("Drive returned an HTML interstitial rather than the file")
        with open(target, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def _download_one(gdown, entry, target: Path) -> bool:
    """Retrieve one file through complementary transports with bounded retries.

    Back-off preserves acquisition completeness under transient public-access
    rate limiting; an omitted file can remove an entire reporting month.
    """
    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        errors = []
        # Direct retrieval avoids a rate-limited metadata pre-flight; gdown remains
        # necessary for confirmation-token and large-file interstitial handling.
        for label, fetch in (
            ("direct", lambda: _download_direct(entry, target)),
            ("gdown", lambda: _download_via_gdown(gdown, entry, target)),
        ):
            try:
                fetch()
                if not target.exists() or target.stat().st_size == 0:
                    raise RuntimeError("empty download")
                return True
            except Exception as exc:  # noqa: BLE001 - try the other transport, then back off
                if target.exists():
                    os.remove(target)
                message = str(exc).strip().splitlines()[0] or type(exc).__name__
                errors.append(f"{label}: {message}")

        if attempt == DOWNLOAD_MAX_ATTEMPTS:
            logger.warning("  ! download failed for %s (%s)", entry.path, "; ".join(errors))
            return False
        logger.info(
            "  . retry %d/%d for %s in %ds (%s)",
            attempt, DOWNLOAD_MAX_ATTEMPTS - 1, entry.path,
            DOWNLOAD_RETRY_BACKOFF_S * attempt, "; ".join(errors),
        )
        time.sleep(DOWNLOAD_RETRY_BACKOFF_S * attempt)
    return False


# ---------------------------------------------------------------------------
# Consolidation: spend sources
# ---------------------------------------------------------------------------
_ISO_MIDNIGHT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]00:00:00(?:\.0+)?$")


def _drop_zero_time_component(series: pd.Series) -> pd.Series:
    """Remove only a zero-time suffix from otherwise identical ISO dates.

    This harmonises Bradford's date-only and midnight-timestamp publications
    without interpreting ambiguous day/month formats, which remain the
    responsibility of Lincolnshire's ``_parse_mixed_dates`` routine.
    """
    return series.astype("string").str.replace(_ISO_MIDNIGHT_RE, r"\1", regex=True)


def _consolidate_spend(staging_dir, folder_name: str, dataset_source: str, columns: list) -> pd.DataFrame:
    frames = []
    for path in _iter_source_files(staging_dir, folder_name):
        df = _read_spend_file(path)
        if df is None or df.empty:
            continue
        df = df.rename(columns=_build_rename_map(df.columns))
        keep = [c for c in columns if c in df.columns]
        if "supplier" not in keep or "amount" not in keep:
            logger.warning("  ! %s has no recognisable supplier/amount column, skipped", path.name)
            continue
        block = df[keep].copy()
        block["_dataset_source"] = dataset_source
        block["_source_file"] = path.name
        frames.append(block)

    if not frames:
        raise RuntimeError(f"No readable source files found for {folder_name} under {staging_dir}")

    out = pd.concat(frames, ignore_index=True)
    out = out.reindex(columns=columns)
    out["date"] = _drop_zero_time_component(out["date"])
    logger.info("%s: consolidated %d rows from %d source files", dataset_source, len(out), len(frames))
    return out


def consolidate_bradford(staging_dir=None) -> pd.DataFrame:
    return _consolidate_spend(
        staging_dir or config.RAW_STAGING_DIR,
        "NHS Bradford teaching hospitals",
        "Bradford_Teaching_Hospitals",
        BRADFORD_COLUMNS,
    )


def consolidate_nhs_england(staging_dir=None) -> pd.DataFrame:
    out = _consolidate_spend(
        staging_dir or config.RAW_STAGING_DIR,
        "NHS England",
        "NHS_England",
        NHS_ENGLAND_COLUMNS,
    )
    # The publisher is implicit in some monthly exports, leaving entity blank.
    out["entity"] = out["entity"].fillna("NHS England")
    return out


def consolidate_lincolnshire(staging_dir=None) -> pd.DataFrame:
    """Consolidate heterogeneous United Lincolnshire Hospitals monthly exports.

    Headered and headerless layouts coexist. This stage retains original dates
    and amounts; Phase 2 applies ``_parse_mixed_dates`` and monetary cleaning.
    """
    staging_dir = staging_dir or config.RAW_STAGING_DIR
    frames = []
    n_headerless = 0

    for path in _iter_source_files(staging_dir, "United Lincolnshire Hospitals"):
        try:
            probe = _read_csv_any_encoding(path, header=None, nrows=6, dtype=str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  ! could not read %s: %s", path.name, exc)
            continue
        if probe.empty:
            continue

        # May 2019 includes a title row before its header; the header is not
        # necessarily the first row.
        header_row = next(
            (i for i in range(len(probe)) if _looks_like_header(probe.iloc[i].tolist())), None
        )

        try:
            if header_row is not None:
                df = _read_csv_any_encoding(path, header=header_row, dtype=str)
                df = df.rename(columns=_build_rename_map(df.columns))
            else:
                df = _read_csv_any_encoding(path, header=None, dtype=str)
                n_headerless += 1
                # Map the eight documented positional fields and discard trailing fields.
                width = min(len(LINCOLNSHIRE_POSITIONAL_COLUMNS), df.shape[1])
                df = df.iloc[:, :width]
                df.columns = LINCOLNSHIRE_POSITIONAL_COLUMNS[:width]
        except Exception as exc:  # noqa: BLE001
            logger.warning("  ! could not read %s: %s", path.name, exc)
            continue

        keep = [c for c in LINCOLNSHIRE_COLUMNS if c in df.columns]
        block = df[keep].copy()
        block["_dataset_source"] = "United_Lincolnshire_Hospitals"
        block["_source_file"] = path.name
        frames.append(block)

    if not frames:
        raise RuntimeError(f"No readable United Lincolnshire files found under {staging_dir}")

    out = pd.concat(frames, ignore_index=True).reindex(columns=LINCOLNSHIRE_COLUMNS)
    out["date"] = _drop_zero_time_component(out["date"])
    logger.info(
        "United_Lincolnshire_Hospitals: consolidated %d rows from %d source files (%d headerless)",
        len(out), len(frames), n_headerless,
    )
    return out


# ---------------------------------------------------------------------------
# Consolidation: Contracts Finder
# ---------------------------------------------------------------------------
def _is_nhs_related(df: pd.DataFrame, text_columns: list) -> pd.Series:
    """Return a mask for records whose available text matches health-sector terms."""
    present = [c for c in text_columns if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    text = df[present[0]].astype(str)
    for col in present[1:]:
        text = text + " " + df[col].astype(str)
    return text.str.contains(_NHS_RE, na=False)


def consolidate_contracts(staging_dir=None) -> pd.DataFrame:
    """Consolidate the health-sector subset of the Contracts Finder OCDS export.

    An outer schema union accommodates annual field drift, while ``_source_file``
    preserves provenance and permits Phase 2 to select the ``main.csv`` notices.
    """
    staging_dir = staging_dir or config.RAW_STAGING_DIR
    frames = []
    stats = {}

    for path in _iter_source_files(staging_dir, "NHS UK Contracts"):
        if path.name not in CONTRACTS_WANTED_FILES:
            continue
        try:
            df = _read_csv_any_encoding(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  ! could not read %s/%s: %s", path.parent.name, path.name, exc)
            continue

        mask = _is_nhs_related(df, CONTRACTS_TEXT_COLUMNS.get(path.name, []))
        kept = df[mask].copy()
        kept["_source_file"] = path.name
        frames.append(kept)

        raw_n, kept_n = stats.get(path.name, (0, 0))
        stats[path.name] = (raw_n + len(df), kept_n + len(kept))
        logger.info(
            "  %s/%s: %d of %d rows matched the health-sector filter",
            path.parent.name, path.name, len(kept), len(df),
        )

    if not frames:
        raise RuntimeError(f"No readable Contracts Finder files found under {staging_dir}")

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "_dataset_source", "NHS_UK_Contracts")
    for name, (raw_n, kept_n) in sorted(stats.items()):
        logger.info("NHS_UK_Contracts %s: kept %d of %d national rows", name, kept_n, raw_n)
    logger.info("NHS_UK_Contracts: consolidated %d rows, %d columns", len(out), out.shape[1])
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
BUILDERS = {
    "bradford": consolidate_bradford,
    "lincolnshire": consolidate_lincolnshire,
    "nhs_england": consolidate_nhs_england,
    "contracts_finder": consolidate_contracts,
}


def raw_files_present() -> bool:
    """Return whether every required consolidated raw-data file is present."""
    return all(Path(p).exists() for p in config.RAW_FILES.values())


def build_all(force: bool = False, staging_dir=None, out_dir=None) -> dict:
    """Acquire and consolidate the source archive into ``data/raw``.

    Returns source-level row counts. Existing complete outputs are retained
    unless ``force`` is set, preserving a manually provisioned raw-data state.
    """
    out_dir = Path(out_dir or config.DATA_RAW_DIR)

    if not force and raw_files_present():
        logger.info("All four consolidated raw files already present in %s, skipping Phase 1", out_dir)
        return {}

    logger.info("=" * 70)
    logger.info("PHASE 1: Data Acquisition & Consolidation (Google Drive raw archive)")
    logger.info("=" * 70)

    staging_dir = download_drive_folder(staging_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for key, builder in BUILDERS.items():
        df = builder(staging_dir)
        target = out_dir / Path(config.RAW_FILES[key]).name
        df.to_csv(target, index=False)
        summary[key] = len(df)
        logger.info("Wrote %s (%d rows, %d cols)", target, len(df), df.shape[1])

    logger.info("Phase 1 complete: %s", ", ".join(f"{k}={v:,}" for k, v in summary.items()))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and consolidate the raw NHS procurement archive from Google Drive."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="rebuild data/raw/*_clean.csv even if they already exist",
    )
    args = parser.parse_args()
    build_all(force=args.force)


if __name__ == "__main__":
    main()
