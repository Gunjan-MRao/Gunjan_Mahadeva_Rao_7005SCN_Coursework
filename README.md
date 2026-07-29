# From Manual to Machine: An Intelligent Anomaly Detection System for UK Public Sector Procurement Spend (2019–2024)

MSc Data Science coursework (module 7005SCN, Coventry University). Supervisor: Dr. Nahid Salimi.

## Research question

> To what extent can unsupervised anomaly detection models trained on pre-COVID NHS procurement data identify anomalous spending patterns during and after COVID-19, and how can explainable AI improve interpretation of the detected anomalies?

This repository implements the full analytical pipeline behind that question: data engineering, exploratory/shock analysis, unsupervised anomaly detection, explainability, and a defensible proxy validation against literature-derived audit red-flag criteria.

## Repository structure

```
data/
  raw/            4 pre-cleaned source extracts (gitignored — see "Reproducing the data" below)
  processed/      pipeline outputs (small summary CSVs committed; large ones gitignored, see .gitignore)
docs/
  figures/        report figures (PNG, committed)
src/
  config.py               central configuration (paths, COVID period boundaries, model hyperparameters)
  data_engineering/
    loaders.py            per-source loading + schema standardisation
    clean_merge.py         cleaning, filtering, feature engineering, merge into master panel
  analysis/
    eda.py                 data quality report, spend distribution, new-supplier rate
    hhi.py                  Herfindahl-Hirschman Index supplier concentration
    stl_shock.py            STL decomposition / COVID-19 shock quantification
    make_figures.py         generates all docs/figures/*.png
  modeling/
    isolation_forest_shap.py   Isolation Forest training/scoring + SHAP explainability
  validation/
    audit_validation.py    rule-based red-flag validation + ML/rule triangulation
  run_pipeline.py           orchestrates all phases end-to-end
tests/                      pytest unit tests (schema, cleaning logic, feature/rule correctness)
requirements.txt
```

## Methodology (mapped to the original proposal's 5-stage design)

**Stage 1 — Data engineering.** Four public procurement datasets are ingested and standardised into one schema (`source, entity, supplier, date, amount, category, sub_category, record_type, transaction_id`):

| Source | Rows (post-clean) | Coverage | Notes |
|---|---|---|---|
| NHS England (national trust spend, FOI ≥£25k+ disclosures) | 249,613 | 2019-01 – 2024-12 | Primary source; near-complete |
| Bradford Teaching Hospitals NHS FT | 6,476 | 2019-02 – 2022-02 | Trust-level FOI transparency data |
| United Lincolnshire Hospitals NHS Trust | 1,006 | 2019-01 – 2021-07 | Raw file mixes ISO and UK (`dd/mm/yyyy`) date formats across monthly sub-files — handled explicitly in `loaders._parse_mixed_dates` |
| UK Contracts Finder (OCDS contract notices) | 30,976 | 2019-01 – 2024-12 | `tender_procurementMethod` used as the single-source/direct-award signal for R1 in Stage 5 |

Cleaning steps: drop null separator rows, parse mixed date formats, convert comma/currency-formatted amount strings to numeric, remove non-positive and extreme-outlier amounts (>£50m for trust spend, >£2bn for contract notices — treated as data-entry/framework-ceiling artefacts), normalise supplier names (case, punctuation, company-type suffixes) for cross-record matching.

**COVID-19 period boundaries** (UK government lockdown/reopening dates):
- Pre-COVID: 2019-01-01 → 2020-03-22
- COVID: 2020-03-23 → 2022-02-23
- Post-COVID: 2022-02-24 → 2024-12-31

**Stage 2 — Exploratory & shock analysis.**
- *Supplier concentration (HHI)*: computed monthly per source. National NHS England aggregate HHI is naturally low (hundreds of distinct trusts/suppliers); the individually-tracked small trusts (Bradford, Lincolnshire) show genuine concentration swings, including sustained periods above the HHI=2500 "highly concentrated" threshold (Hirschman, 1945).
- *STL decomposition* (Cleveland et al., 1990) on monthly aggregate trust spend, baseline window 2018-07 – 2019-12 (extended pre-period for a more stable seasonal/trend fit), quantifies the COVID shock: **+33.0% average deviation from the pre-COVID baseline during COVID, +38.6% post-COVID** — spend did not revert after the acute pandemic period.
- *New-supplier rate*: the proportion of transactions going to a supplier never seen before for that entity. A 6-month burn-in window is excluded from this flag to avoid left-censoring bias (every supplier looks "new" in the first observed months purely because there's no earlier history to compare against — this was found and corrected during development, see `clean_merge.py` comments). Result: new-supplier rate rises from **1.32% (pre-COVID) to 3.21% (COVID)**, dropping back to 1.71% post-COVID — consistent with the emergency/direct-award procurement risk documented in Transparency International UK's [Track and Trace](https://www.transparency.org.uk/sites/default/files/2024-11/Track%20and%20Trace%20-%20Transparency%20International%20UK.pdf) (2021) and its 2024 follow-up investigation, ["Landmark investigation finds corruption red flags in £15.3 billion of UK COVID contracts"](https://www.transparency.org.uk/news/report-landmark-investigation-finds-corruption-red-flags-ps153-billion-uk-covid-contracts), both of which flag contracts awarded to companies with no prior track record as a key corruption-risk indicator.

**Stage 3 — Unsupervised anomaly detection.** Isolation Forest (Liu, Ting & Zhou, 2008; `n_estimators=300, contamination=0.02, random_state=42`) trained **only on pre-COVID trust-spend records** so it learns "normal" pre-pandemic behaviour, then scores every record. Features: log-amount, within-source-category amount z-score, days since the supplier's last transaction, transaction sequence number for that supplier, new-supplier flag, month, day-of-week.

Note: an earlier iteration also included ordinal-encoded `source`/`category` as raw features. SHAP analysis showed ~60% of top anomalies were then driven almost entirely by which (small) dataset a record came from, not genuine spend anomalies — an artefact of NHS England dominating the training sample. These identity features were removed; `amount_zscore_category` already captures the useful within-group signal without that bias.

Records are flagged as anomalous at the 98th percentile of anomaly score. Result: **2.00% of records flagged overall, but the rate roughly triples between periods: 0.37% pre-COVID (training baseline) → 1.82% COVID → 2.64% post-COVID**, indicating growing divergence from pre-pandemic "normal" procurement behaviour that persists well after the acute crisis.

**Stage 4 — Explainability.** SHAP `TreeExplainer` (Lundberg & Lee, 2017) computed on all flagged anomalies plus a contrast sample of normal records. Across the top 200 anomalies, the dominant driver is `supplier_txn_seq` (169/200) — i.e. transactions very early in a new supplier relationship, consistent with the new-supplier-rate finding — followed by transaction recency, category-relative amount, and new-supplier status.

**Stage 5 — Validation.** Confidential NAO / NHS Counter Fraud Authority case-level audit data is not publicly accessible, so this project uses a defensible literature-derived proxy: four explicit rule-based red flags (direct-award COVID contracts, >3× historical-median price spikes, new-supplier + large COVID contract, suspiciously round invoice amounts). These map onto the corruption-risk indicators published in Transparency International UK's [Track and Trace](https://www.transparency.org.uk/sites/default/files/2024-11/Track%20and%20Trace%20-%20Transparency%20International%20UK.pdf) (2021) — which found over 20% of £18bn UK pandemic procurement contracts raised at least one red flag, predominantly single-source/direct-award and no-track-record-supplier signals — and the NHS Counter Fraud Authority's published fraud-risk quick guides, including [Preventing procurement fraud in the NHS](https://cfa.nhs.uk/resources/downloads/documents/fraud-reports/Preventing_procurement.pdf) (2022) and [Buying goods and services](https://cfa.nhs.uk/resources/downloads/guidance/fraud-awareness/quick-reference-guides/Buying_goods-and-services.pdf) (2026), which list invoice manipulation, high-risk/no-track-record suppliers, and non-PO spend as key vulnerability areas. **62.7% of ML-flagged anomalies are corroborated by at least one independent rule** (vs. an 18.9% base rate if rule-flags were randomly distributed), providing triangulated construct validity for the unsupervised model without needing access to confidential audit case data.

## Key results at a glance

| Metric | Pre-COVID | COVID | Post-COVID |
|---|---|---|---|
| Avg. monthly trust spend, % deviation from baseline (STL) | −0.03% | **+33.0%** | +38.6% |
| New-supplier rate (% of transactions) | 1.32% | **3.21%** | 1.71% |
| Isolation Forest anomaly rate | 0.37% | 1.82% | **2.64%** |

ML/rule-based triangulation: **3,226 / 5,142 (62.7%)** ML-flagged anomalies corroborated by ≥1 independent audit red-flag rule.

Figures (see `docs/figures/`):
- `stl_decomposition.png` — observed/trend/seasonal/residual monthly spend with COVID period shaded
- `hhi_trend.png` — supplier concentration by source over time
- `new_supplier_rate.png` — new-supplier onboarding rate, COVID spike clearly visible
- `anomaly_timeline.png` — monthly Isolation Forest anomaly rate
- `shap_top_feature_summary.png` — dominant SHAP feature across top anomalies

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. Place the 4 raw source CSVs in data/raw/ (see "Reproducing the data" below)
#    bradford_clean.csv, lincolnshire_clean.csv, nhs_england_clean.csv, contracts_clean.csv

# 2. Run the full pipeline end-to-end (~40s on 2 vCPU / 8GB RAM)
python -m src.run_pipeline

# 3. Generate report figures
python -m src.analysis.make_figures

# 4. Run the test suite
pytest tests/ -v
```

### Reproducing the data

The 4 raw CSVs (~106MB total) are gitignored to keep the repository light — they are third-party FOI/Contracts Finder exports, not code, and are fully regenerable from public sources:
- NHS England / Bradford / Lincolnshire trust spend: published under each trust's Freedom-of-Information "spend over £25k" transparency disclosures.
- UK Contracts Finder: [OCDS-format contract notices](https://www.contractsfinder.service.gov.uk/) for NHS/DHSC buyers, 2019–2024.

Place the 4 cleaned CSVs directly in `data/raw/` with the filenames referenced in `src/config.py`'s `RAW_FILES` dict, then run the pipeline.

## Data quality caveats and limitations

- United Lincolnshire Hospitals data covers only Jan 2019 – Jul 2021 (incomplete disclosure availability), limiting post-COVID comparison for that trust specifically.
- UK Contracts Finder `awards.csv`/`awards_suppliers.csv` sub-files (award-level supplier names) are sparse and were not incorporated into the anomaly-detection panel; only the `main.csv` buyer/tender-notice level is used.
- The rule-based validation in Stage 5 is a literature-derived **proxy** for genuine audit ground truth, not a substitute for it — the overlap statistic indicates triangulated plausibility, not confirmed fraud/error.
- Amount caps (>£50m trust spend, >£2bn contract notices treated as data artefacts) are a modelling judgement call; see `src/data_engineering/clean_merge.py` for the exact thresholds and rationale.

## References

- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. *Journal of Official Statistics*, 6(1), 3–73.
- Hirschman, A. O. (1945). *National Power and the Structure of Foreign Trade*. University of California Press. (Herfindahl-Hirschman Index)
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *2008 Eighth IEEE International Conference on Data Mining*, 413–422. https://doi.org/10.1109/ICDM.2008.17
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.
- Transparency International UK. (2021). *Track and Trace: Procurement for the COVID-19 Pandemic*. https://www.transparency.org.uk/sites/default/files/2024-11/Track%20and%20Trace%20-%20Transparency%20International%20UK.pdf
- Transparency International UK. (2024). *Landmark investigation finds corruption red flags in £15.3 billion of UK COVID contracts*. https://www.transparency.org.uk/news/report-landmark-investigation-finds-corruption-red-flags-ps153-billion-uk-covid-contracts
- NHS Counter Fraud Authority. (2022). *Preventing Procurement Fraud in the NHS*. https://cfa.nhs.uk/resources/downloads/documents/fraud-reports/Preventing_procurement.pdf
- NHS Counter Fraud Authority. (2026). *Buying Goods and Services (Quick Reference Guide)*. https://cfa.nhs.uk/resources/downloads/guidance/fraud-awareness/quick-reference-guides/Buying_goods-and-services.pdf
