# Data directory guide

This folder is intentionally mostly empty in git. It documents where the
data lives, how it is structured, and how to regenerate it — see
`.gitignore` at the repo root for the exact list of excluded paths and the
reasoning behind each exclusion.

## Why the raw files aren't committed

The underlying sources are large, third-party FOI (Freedom of Information)
and Contracts Finder exports (~106MB combined for `data/raw/`, and up to
~500MB of original per-month files when staged from Google Drive) — not
code, and fully regenerable. Keeping them out of git keeps the repository
light and avoids re-distributing third-party data dumps. A handful of large
*derived* outputs (the full analytical panel, per-record anomaly scores,
per-record validation flags, the per-record method-comparison score dump,
and the BI fact table) are excluded for the same reason — they sit close to
GitHub's 100MB per-file limit and provide no review value as raw dumps. All
small summary/aggregate CSVs that directly support the report's figures and
tables **are** committed.

## Expected folder structure

```
data/
├── raw/                        # gitignored — 4 consolidated per-source CSVs
│   ├── bradford_clean.csv
│   ├── lincolnshire_clean.csv
│   ├── nhs_england_clean.csv
│   └── contracts_clean.csv
├── _raw_staging/                # gitignored — ~500MB of original per-month
│                                 # FOI/Contracts Finder files from Google Drive
├── raw_ground_truth_backup/     # gitignored — local safety copy of data/raw/
└── processed/                   # small summary CSVs committed; large
    │                             # per-record dumps gitignored (see above)
    ├── master_procurement_panel.csv        # gitignored (~79MB)
    ├── anomaly_scores.csv                  # gitignored (~38MB)
    ├── validation_redflags.csv             # gitignored (~94MB)
    ├── method_comparison_scores.csv        # gitignored (~59MB)
    ├── data_quality_report.csv             # committed
    ├── hhi_monthly.csv                     # committed
    ├── hhi_monthly_by_source.csv           # committed
    ├── new_supplier_rate_by_month.csv      # committed
    ├── shap_top_anomalies.csv              # committed
    ├── spend_distribution_by_period.csv    # committed
    ├── stl_decomposition.csv               # committed
    ├── stl_shock_summary.csv               # committed
    ├── method_comparison_summary.csv       # committed
    ├── method_comparison_summary_agreement.csv  # committed
    ├── synthetic_injection_evaluation.csv        # committed
    ├── synthetic_injection_evaluation_by_type.csv # committed
    ├── network_communities.csv             # committed
    ├── network_supplier_metrics.csv        # committed
    ├── robustness_threshold_sensitivity.csv # committed
    ├── robustness_period_shift.csv         # committed
    ├── robustness_feature_ablation.csv     # committed
    ├── supplier_risk_score.csv             # committed
    ├── category_deep_dive.csv              # committed
    ├── category_covid_shock_ranking.csv    # committed
    ├── statistical_tests_summary*.csv      # committed
    └── bi_export/                          # star-schema export
        ├── fact_transactions.csv           # gitignored (~74MB)
        ├── dim_supplier.csv                # committed
        ├── dim_category.csv                # committed
        ├── dim_period.csv                  # committed
        └── dim_month.csv                   # committed
```

## Regenerating the raw data

The four `data/raw/*_clean.csv` files are built from a public Google Drive
archive of the original FOI and Contracts Finder exports. To rebuild them
from scratch:

```bash
python -m src.data_engineering.build_raw_from_drive
```

Add `--force` to re-download and re-consolidate even if the raw files
already exist locally. This step downloads and consolidates:

- **Bradford Teaching Hospitals** — 13,925 rows from 72 source files
- **United Lincolnshire Hospitals** — 8,280 rows from 72 source files
- **NHS England** — 271,598 rows from 72 source files
- **Contracts Finder** — 81,482 rows, 45 columns (health-sector filtered
  from the national `awards.csv` / `awards_suppliers.csv` / `main.csv`
  bulk exports)

If you'd rather not use Google Drive, you can place the four
`data/raw/*_clean.csv` files manually in `data/raw/` (matching the column
structure the loaders in `src/data_engineering/` expect) and skip this step
— `gdown` is only required for the automated download path.

## Regenerating the processed data

Once `data/raw/` is populated (either via the step above or manually), run
the full pipeline:

```bash
python -m src.run_pipeline
```

This merges and cleans the four raw sources into the analytical panel
(**326,991 rows** after cleaning, from 349,584 raw merged rows — 93.5%
retained), then runs the EDA/shock analysis, anomaly detection (Isolation
Forest + SHAP), rule-based validation, multi-method comparison, statistical
significance testing, network analysis, composite risk scoring, category
deep-dive, robustness checks, BI export, and interactive dashboard stages in
sequence, writing all outputs into `data/processed/`.

To regenerate the report figures afterwards:

```bash
python -m src.analysis.make_figures
```

## Key processed output files

| File | Description |
|---|---|
| `master_procurement_panel.csv` (gitignored) | The full analytical panel: 326,991 cleaned transaction rows across all four sources, 2019–2024 |
| `anomaly_scores.csv` (gitignored) | Per-record Isolation Forest anomaly scores and flags |
| `validation_redflags.csv` (gitignored) | Per-record rule-based audit red-flag indicators (direct-award-during-COVID, price spike, new-supplier-large-COVID, round amount) |
| `shap_top_anomalies.csv` | SHAP feature-attribution summary for the top-200 highest-confidence anomalies |
| `supplier_risk_score.csv` | Composite 0–100 risk score and tier (Low/Medium/High/Critical) per supplier |
| `network_communities.csv` | Supplier co-occurrence community assignments from the collusion-indicator network analysis |
| `network_supplier_metrics.csv` | Bipartite buyer–supplier graph node metrics, including hub-supplier classification |
| `statistical_tests_summary*.csv` | Hypergeometric triangulation significance test, circularity check, and Mann-Whitney U test results |
| `bi_export/dim_supplier.csv`, `dim_category.csv`, `dim_period.csv`, `dim_month.csv` | Star-schema dimension tables for BI tools (Power BI / Tableau) |
| `bi_export/fact_transactions.csv` (gitignored) | Star-schema fact table (326,991 rows × 26 columns), regenerable via `python -m src.analysis.bi_export` |

See the root [README.md](../README.md) and [docs/dissertation_sections.md](../docs/dissertation_sections.md) for the full methodology and findings that these files support.
