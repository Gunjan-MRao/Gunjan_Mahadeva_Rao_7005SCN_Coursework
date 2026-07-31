"""
Phase 1 — Data Acquisition & Consolidation.

Fetches the original raw FOI/Contracts Finder archive from the project's public
Google Drive folder and consolidates it into the four files that Phase 2
(`loaders.py`) consumes:

    data/raw/bradford_clean.csv
    data/raw/lincolnshire_clean.csv
    data/raw/nhs_england_clean.csv
    data/raw/contracts_clean.csv

Despite the historical `_clean` suffix these outputs are *consolidated raw*, not
analytically clean: this module only concatenates the per-month source files and
standardises column names. Every substantive cleaning decision (dropna, mixed
date parsing, amount coercion, supplier normalisation) deliberately stays in
`loaders.py` so the separation of concerns documented for Phases 2-7 is
preserved — see docs/dissertation_sections.md "Phase 1".

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
# Drive layout
# ---------------------------------------------------------------------------
# Top-level folder name in the Drive archive -> logical source key.
SOURCE_FOLDERS = {
    "NHS Bradford teaching hospitals": "bradford",
    "NHS England": "nhs_england",
    "United Lincolnshire Hospitals": "lincolnshire",
    "NHS UK Contracts": "contracts",
}

# The Contracts Finder bulk export ships 13 CSVs per year; only these three
# carry fields used downstream. The rest (parties, *_documents, planning_*,
# tender_*, relatedProcesses) are 10-30MB each and unused, so they are never
# downloaded.
CONTRACTS_WANTED_FILES = {"main.csv", "awards.csv", "awards_suppliers.csv"}

SPREADSHEET_SUFFIXES = {".xls", ".xlsx", ".xlsm"}

# Anonymous access to a public Drive folder is rate-limited. Downloading the
# ~500MB archive flat-out reliably trips Google's abuse threshold partway
# through, after which every subsequent request fails for several minutes, so
# requests are paced and retried with a long backoff rather than hammered.
DOWNLOAD_MAX_ATTEMPTS = 4
DOWNLOAD_RETRY_BACKOFF_S = 60
DOWNLOAD_PACING_S = 0.5

# ---------------------------------------------------------------------------
# Health/NHS relevance filter for the Contracts Finder national bulk export
# ---------------------------------------------------------------------------
# The Drive archive holds the *full UK national* Contracts Finder export
# (~50,000 notices per year across every public-sector buyer), whereas this
# study covers the health sector only. This keyword filter reconstructs the
# health subset. It is a best-effort reconstruction, not a byte-exact replay of
# whatever bespoke search produced the original extract — see the limitation
# recorded in docs/dissertation_sections.md "Phase 1".
NHS_KEYWORDS = (
    r"nhs|health|hospital|clinical|ambulance|blood|commissioning support|"
    r"clinical commissioning|integrated care board|hospice|primary care|care trust"
)
_NHS_RE = re.compile(NHS_KEYWORDS, re.IGNORECASE)

# Text columns searched per Contracts Finder file. main.csv exposes buyer and
# tender text; awards.csv only a free-text description; awards_suppliers.csv
# only the awarded supplier name.
CONTRACTS_TEXT_COLUMNS = {
    "main.csv": ["buyer_name", "tender_title", "tender_description"],
    "awards.csv": ["description"],
    "awards_suppliers.csv": ["name"],
}

# ---------------------------------------------------------------------------
# Raw -> intermediate column mapping
# ---------------------------------------------------------------------------
# Header spellings drift across years and sources ("AP Amount" vs
# "AP Amount (£)", "Transaction Number" vs "Transaction number", a trailing
# space in Lincolnshire's "Supplier "), so headers are normalised to lowercase
# alphanumeric-and-space tokens before lookup.
SPEND_COLUMN_ALIASES = {
    "department_family": ["department family", "department of health", "department"],
    "entity": ["entity"],
    "date": ["date", "date of payment", "payment date", "date paid", "transaction date"],
    # Lincolnshire renames its spend-category column almost every year
    # ("Expenditure type" -> "Nature of spend" -> "9AN - Level 9 Account Name");
    # NHS England appends " for Report" from Nov-2024 onwards.
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

# Minimum number of cells in a row that must look like known column headers for
# that row to be treated as the header. A threshold (rather than looking for one
# specific token) is required because United Lincolnshire's headerless files
# begin with the literal value "Department Of Health" in column 1 — identical to
# the header token — so single-token tests cannot tell header from data there.
# A real header row matches 7-10 aliases; a data row matches at most 1.
HEADER_ALIAS_MIN_HITS = 3

BRADFORD_COLUMNS = [
    "_dataset_source", "department_family", "entity", "date", "expense_type",
    "expense_area", "supplier", "transaction_number", "amount", "vat_number",
    "invoice_number", "_source_file",
]
NHS_ENGLAND_COLUMNS = BRADFORD_COLUMNS

# United Lincolnshire publishes two shapes of the same 8 positional fields:
# some months carry a header row, others are headerless (first row is data).
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
    """'AP Amount (£)' -> 'ap amount';  'Supplier ' -> 'supplier'."""
    text = re.sub(r"[^0-9a-z]+", " ", str(name).lower())
    return re.sub(r"\s+", " ", text).strip()


_ALIAS_LOOKUP = {
    alias: canon for canon, aliases in SPEND_COLUMN_ALIASES.items() for alias in aliases
}


def _build_rename_map(columns) -> dict:
    """Map a raw file's actual columns onto canonical intermediate names."""
    rename = {}
    for col in columns:
        canon = _ALIAS_LOOKUP.get(_normalise_header(col))
        if canon and canon not in rename.values():
            rename[col] = canon
    return rename


def _header_alias_hits(cells) -> int:
    """How many cells in this row look like known spend-file column headers?"""
    return sum(1 for c in cells if _normalise_header(c) in _ALIAS_LOOKUP)


def _looks_like_header(cells) -> bool:
    return _header_alias_hits(cells) >= HEADER_ALIAS_MIN_HITS


def _sniff_delimiter(path: Path) -> str:
    """Lincolnshire's May-19 month is published as tab-separated data under a
    `.csv` extension, so the extension cannot be trusted to imply commas."""
    with open(path, encoding="latin1") as fh:
        head = "".join(next(fh, "") for _ in range(5))
    return "\t" if head.count("\t") > head.count(",") else ","


def _read_csv_any_encoding(path: Path, **kwargs) -> pd.DataFrame:
    """FOI exports are published with inconsistent encodings (UTF-8 in most
    years, Windows/Latin-1 in others). Try each in turn rather than losing a
    whole month to a UnicodeDecodeError."""
    kwargs.setdefault("sep", _sniff_delimiter(path))
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", encoding_errors="replace", low_memory=False, **kwargs)


def _find_header_row(probe: pd.DataFrame) -> int:
    """Locate the real header row within the first few rows of a spend file.

    2024 Bradford exports prepend a report-title row
    ("A3131. Expenditure Over Threshold Report (AP),Unnamed: 1,...") above the
    genuine header, so the header cannot be assumed to be row 0.
    """
    for idx in range(len(probe)):
        if _looks_like_header(probe.iloc[idx].tolist()):
            return idx
    return 0


def _read_spend_file(path: Path) -> pd.DataFrame | None:
    """Read one monthly spend export (CSV or Excel), skipping any title row."""
    try:
        if path.suffix.lower() in SPREADSHEET_SUFFIXES:
            # openpyxl handles .xlsx/.xlsm (Bradford's Jan-19 is published only as
            # .xlsm); legacy .xls needs pandas' own engine selection (xlrd).
            engine = "openpyxl" if path.suffix.lower() in {".xlsx", ".xlsm"} else None
            probe = pd.read_excel(path, header=None, nrows=6, dtype=str, engine=engine)
            return pd.read_excel(path, header=_find_header_row(probe), dtype=str, engine=engine)
        probe = _read_csv_any_encoding(path, header=None, nrows=6, dtype=str)
        return _read_csv_any_encoding(path, header=_find_header_row(probe), dtype=str)
    except Exception as exc:  # noqa: BLE001 - one unreadable month must not abort the build
        logger.warning("  ! could not read %s: %s", path.name, exc)
        return None


def _iter_source_files(staging_dir: Path, folder_name: str):
    """Yield readable source files for one Drive top-level folder, sorted."""
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
    """Pick the subset of the Drive archive actually needed.

    * spend sources — prefer the `.csv` publication of each month and skip its
      co-located Excel duplicate; fall back to the Excel file only where no CSV
      exists (Bradford's `Jan-19.xlsm` is published without a CSV counterpart).
    * Contracts Finder — keep only main/awards/awards_suppliers.
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
        # group the .csv and its .xls/.xlsx twin under one key (dir + stem)
        by_month.setdefault((parts[:-1], Path(name).stem.lower()), []).append(entry)

    for candidates in by_month.values():
        csvs = [e for e in candidates if Path(e.path).suffix.lower() == ".csv"]
        selected.append(csvs[0] if csvs else candidates[0])

    return selected


def download_drive_folder(dest_dir=None) -> Path:
    """Download the needed subset of the Drive raw archive into `dest_dir`.

    The folder is a public "anyone with the link" share, so no credentials or
    API key are required. Individual failures are logged and skipped so a
    transient error on one month does not abort the whole acquisition step.
    Already-downloaded files are left alone, making the step resumable.
    """
    import gdown  # imported lazily so the rest of the pipeline never needs it

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
            "output: %s. Re-run this module to retry them — already-downloaded files are "
            "skipped, so a second pass only fetches the gaps.",
            len(failures), ", ".join(failures[:10]) + (" ..." if len(failures) > 10 else ""),
        )
    return dest_dir


def _download_via_gdown(gdown, entry, target: Path) -> None:
    gdown.download(id=entry.id, output=str(target), quiet=True)


def _download_direct(entry, target: Path) -> None:
    """Fetch straight from Drive's file-download endpoint.

    gdown's pre-flight metadata lookup is the first thing Google rate-limits: it
    starts raising "Cannot retrieve the public link ... have had many accesses"
    while this endpoint keeps serving files normally. Trying it as a fallback
    recovers most of an otherwise-stalled run.
    """
    import requests  # a gdown dependency, so always present alongside it

    url = f"https://drive.usercontent.google.com/download?id={entry.id}&export=download&confirm=t"
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
    with requests.get(url, headers=headers, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        # Drive serves an HTML interstitial (virus-scan warning / quota notice)
        # instead of the file when it does not want to hand it over anonymously.
        if "text/html" in resp.headers.get("Content-Type", ""):
            raise RuntimeError("Drive returned an HTML interstitial rather than the file")
        with open(target, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def _download_one(gdown, entry, target: Path) -> bool:
    """Fetch one Drive file, trying both transports and retrying with backoff.

    Google rate-limits anonymous access to public folders, which shows up as
    sporadic then sustained per-file failures partway through a ~230-file run.
    These are transient, so each file gets a few spaced retries across both
    transports before being given up on; a lost file otherwise silently drops a
    whole month of spend from the consolidated output.
    """
    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        errors = []
        # `direct` first: it is a single HTTP GET, whereas gdown does a metadata
        # pre-flight that is both slower and the first thing Google rate-limits.
        # gdown remains the fallback because it handles the confirm-token /
        # virus-scan interstitial flow that large files need.
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
# Consolidation — spend sources
# ---------------------------------------------------------------------------
_ISO_MIDNIGHT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]00:00:00(?:\.0+)?$")


def _drop_zero_time_component(series: pd.Series) -> pd.Series:
    """Render ISO dates one way so a concatenated `date` column has one format.

    Bradford publishes 2019-2021 as `2019-04-30` but 2022-2024 as
    `2023-06-30 00:00:00`. Both are unambiguous ISO-8601, but `pd.to_datetime`
    infers a single format from the first value and coerces the rest to NaT, so
    the mixed column silently loses whichever half it did not infer. Only an
    exact zero time component is stripped; anything else (including ambiguous
    dd/mm vs mm/dd forms) is passed through untouched for `loaders.py` to
    interpret, so Lincolnshire's `_parse_mixed_dates` behaviour is unaffected.
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
            logger.warning("  ! %s has no recognisable supplier/amount column — skipped", path.name)
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
    # Mirrors load_nhs_england()'s .fillna("NHS England"): a handful of monthly
    # exports leave the entity column blank because the publisher is implicit.
    out["entity"] = out["entity"].fillna("NHS England")
    return out


def consolidate_lincolnshire(staging_dir=None) -> pd.DataFrame:
    """United Lincolnshire Hospitals — the messiest source.

    Two published shapes coexist: months with a header row, and headerless
    months whose first row is already data. Filenames also contain non-ASCII
    characters (en-dashes) and doubled dots. Date/amount messiness is left
    untouched here; `loaders.py::_parse_mixed_dates` handles it in Phase 2.
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

        # May-19 carries a "PUBLISHED AREA" title row above the real header, so
        # the header (when present at all) is not necessarily row 0.
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
                # assign the 8 known fields positionally; ignore extra trailing columns
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
# Consolidation — Contracts Finder
# ---------------------------------------------------------------------------
def _is_nhs_related(df: pd.DataFrame, text_columns: list) -> pd.Series:
    """Row mask: does the buyer/supplier/title/description text look health-sector?"""
    present = [c for c in text_columns if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    text = df[present[0]].astype(str)
    for col in present[1:]:
        text = text + " " + df[col].astype(str)
    return text.str.contains(_NHS_RE, na=False)


def consolidate_contracts(staging_dir=None) -> pd.DataFrame:
    """UK Contracts Finder OCDS bulk export, filtered to the health sector.

    Concatenates main.csv / awards.csv / awards_suppliers.csv across 2019-2024
    with an outer union of columns (the OCDS schema drifts slightly between
    annual exports), tagging each row with its originating `_source_file` so
    `load_contracts_finder()` can keep using only the `main.csv` block.
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
    """True when all four consolidated raw files already exist on disk."""
    return all(Path(p).exists() for p in config.RAW_FILES.values())


def build_all(force: bool = False, staging_dir=None, out_dir=None) -> dict:
    """Download + consolidate the Drive raw archive into `data/raw/`.

    Returns a {key: row_count} summary. No-ops when all four outputs already
    exist and `force` is False, so a working manual setup is never disturbed.
    """
    out_dir = Path(out_dir or config.DATA_RAW_DIR)

    if not force and raw_files_present():
        logger.info("All four consolidated raw files already present in %s — skipping Phase 1", out_dir)
        return {}

    logger.info("=" * 70)
    logger.info("PHASE 1 — Data Acquisition & Consolidation (Google Drive raw archive)")
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
