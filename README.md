# From Manual to Machine: An Intelligent Anomaly Detection System for UK Public Sector Procurement Spend (2019–2024)

[![Tests](https://github.com/Gunjan-MRao/Gunjan_Mahadeva_Rao_7005SCN_Coursework/actions/workflows/test.yml/badge.svg)](https://github.com/Gunjan-MRao/Gunjan_Mahadeva_Rao_7005SCN_Coursework/actions/workflows/test.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository documents MSc Data Science coursework for module 7005SCN at Coventry University, supervised by Dr. Nahid Salimi.

## Research questions

> **RQ1. Spend intelligence & COVID-19 disruption.**
> What patterns exist in supplier concentration, spend distribution, and data quality across NHS procurement bodies (2019–2024), and how significantly did COVID-19 disrupt baseline spending behaviours?

> **RQ2. Anomaly detection.**
> Can unsupervised machine-learning models trained exclusively on pre-COVID NHS procurement data reliably detect anomalous spending patterns during and after the pandemic, and how does detection performance vary across methods?

> **RQ3. Explainability & actionable insight.**
> To what extent can SHAP-based explainability convert black-box anomaly scores into auditable, prioritised procurement-risk signals, and what operational recommendations follow for NHS procurement efficiency, transparency, and risk management?

The repository provides the end-to-end empirical workflow required to address these questions: data engineering, exploratory and shock analysis, unsupervised anomaly detection, SHAP-based explanation, multi-method benchmarking, supplier network analysis, composite risk scoring, and proxy validation against literature-derived audit red-flag criteria.

## What this means in practice?

This section states the corresponding operational interpretation for procurement and audit stakeholders.

**The one-sentence finding.** Procurement transactions most frequently depart from the pre-pandemic statistical baseline when they occur early in a buyer–supplier relationship. The anomaly-detection model therefore identifies relationship novelty, rather than payment size or the scale of established suppliers, as the predominant anomaly characteristic; this pattern has become more prevalent and more persistent since COVID-19, with persistence observed beyond the acute period.

**Why that specific finding, and not a different one, came out of the model.** The Isolation Forest receives no fraud or compliance labels. It estimates the distribution of pre-2020 trust-spend transactions and flags observations from 2020 onward that are dissimilar to that baseline. SHAP then attributes each flagged observation to its input features. Across the 200 highest-confidence flags, 179 of these 200 flags are primarily attributable to the transaction's position in the supplier's relationship with the buyer rather than to payment magnitude. The result therefore supplies a directly reproducible answer to the question of why an observation is statistically unusual, rather than treating the anomaly score as an uninterpreted black-box output.

**Turning the composite supplier risk score into an action, not just a number.** `src/analysis/supplier_risk_score.py` combines three signals: the frequency of flagged transactions, their anomaly-score extremity, and their overlap with an independent literature-derived audit red flag. It converts these into a 0–100 supplier-level score and allocates suppliers to four tiers. The following recommendations translate that prioritisation measure into proportionate assurance responses; they are not automated determinations, because the score is an audit-prioritisation signal rather than a fraud finding:

| Risk tier | Suppliers (of 3,682 scored) | Suggested procurement action |
|---|---|---|
| Critical | 185 | Hold pending manual review before the next payment or contract renewal; escalate to the counter-fraud/audit function rather than resolving at the procurement-desk level. |
| High | 368 | Require a second approver on new POs; request supporting documentation (registration, prior delivery evidence) before extending the relationship further. |
| Medium | 920 | No immediate action; include in the next periodic spend review rather than routine processing. |
| Low | 2,209 | Standard procurement controls are sufficient; no additional review needed. |

**The single most actionable operational implication.** Because relationship novelty, rather than payment size or supplier scale, is the dominant driver of flagged anomalies, additional assurance should not be allocated primarily to the largest suppliers. Suppliers that are structurally central, transacting with multiple trusts or at very high volume, show a *lower* per-transaction anomaly rate (1.02%) than smaller, single-relationship suppliers (2.89%); see Stage 6D below. The evidence instead supports intensified onboarding controls in the first transactions of a *new* supplier relationship, such as verification of company registration and delivery evidence before the third or fourth invoice is paid. Phase 7B supplies a complementary category-level prioritisation criterion: "Med & Surg Equip General" and "CompSwrPrch Additions" exhibit the largest substantive, rather than small-base-artefactual, absolute spend growth during COVID. A retrospective audit sample targeted to these categories during the 2020–2022 window would therefore be more informative than a random panel-wide sample.

**What the model is not saying.** A flagged transaction is not a confirmed error or fraud case. It is a statistical outlier relative to pre-pandemic norms and is independently corroborated by a literature-derived audit rule in 59.6% of cases (see Stage 6 below), a rate materially above the overlap expected from independent detection methods but short of audited ground truth. The risk tiers should consequently be used to allocate limited audit-review capacity, not as a substitute for case review.

## Repository structure

```
data/
  raw/            4 consolidated source extracts (gitignored; see "Reproducing the data" below)
  _raw_staging/   original per-month FOI / Contracts Finder files downloaded by Phase 1 (gitignored)
  processed/      pipeline outputs (small summary CSVs committed; large ones gitignored, see .gitignore)
docs/
  figures/        report figures (PNG, committed)
src/
  config.py               central configuration (paths, COVID period boundaries, model hyperparameters)
  data_engineering/
    build_raw_from_drive.py  Phase 1: fetch raw FOI/Contracts Finder archive from Google
                             Drive + consolidate it into the 4 data/raw/*_clean.csv files
    loaders.py            per-source loading + schema standardisation
    clean_merge.py         cleaning, filtering, feature engineering, merge into master panel
  analysis/
    eda.py                 data quality report, spend distribution, new-supplier rate
    hhi.py                  Herfindahl-Hirschman Index supplier concentration
    stl_shock.py            STL decomposition / COVID-19 shock quantification
    make_figures.py         generates all docs/figures/*.png
  modeling/
    isolation_forest_shap.py   Isolation Forest training/scoring + SHAP explainability
    method_comparison.py       multi-method anomaly detector comparison (IF, LOF, One-Class SVM, MLP-Autoencoder)
    synthetic_evaluation.py    synthetic anomaly injection + precision/recall/F1/PR-AUC evaluation
  network/
    supplier_network.py        bipartite buyer-supplier graph, centrality, co-occurrence community detection
  validation/
    audit_validation.py    rule-based red-flag validation + ML/rule triangulation
  analysis/
    statistical_tests.py    hypergeometric exact test, bootstrap CIs, Mann-Whitney U
    supplier_risk_score.py  composite supplier risk score (Phase 7A)
    category_deep_dive.py   category x period deep dive + COVID-shock ranking (Phase 7B)
    robustness_checks.py    threshold / period-boundary / feature-ablation sensitivity (Phase 7C)
    bi_export.py            BI-ready star-schema CSV export (Phase 7D)
    dashboard.py            interactive Plotly dashboard (Phase 7E)
  run_pipeline.py           orchestrates all phases end-to-end
tests/                      pytest unit tests (schema, cleaning logic, feature/rule correctness, Phase 6 & 6 modules)
requirements.txt
docs/
  bi_dashboard_guide.md   Tableau/Power BI build guide for the BI export CSVs
  dissertation_sections.md  drop-in methodology/results sections for the dissertation write-up
```

**Repository hygiene.** The repository commits only code, small summary or aggregate CSVs (<50KB, required to reproduce a figure or table directly), and report figures. Large per-record intermediate outputs, including the merged panel, per-record anomaly/red-flag/method-comparison scores, and the BI fact table, are each tens of MB and fully regenerable in under two minutes via `python -m src.run_pipeline`; they, the raw-source archive, and local caches (`__pycache__/`, `.pytest_cache/`) are excluded through `.gitignore` (whose comments record the rationale for each exclusion). This policy concentrates the repository on materials required for transparent examination and reproduction rather than regenerable computational by-products.

## Methodology (mapped to the original proposal's 5-stage design)

**Phase 1: Data acquisition & consolidation.** Before Stage 2, the original per-month FOI exports and Contracts Finder bulk extracts (~150+ files in mixed `.csv`/`.xls`/`.xlsx`/`.xlsm` formats) are fetched from the project's Google Drive archive and consolidated into the four `data/raw/*_clean.csv` files that Stage 2 reads. This runs automatically on first pipeline run. See ["Reproducing raw data from Google Drive"](#reproducing-raw-data-from-google-drive) below. The full methodological write-up (provenance, the data-quality issues found at this stage, and the Contracts Finder filter limitation) is in [`docs/dissertation_sections.md`](docs/dissertation_sections.md) under "Phase 1: Data Acquisition & Consolidation".

**Stage 2: Data engineering.** Four public procurement datasets are ingested and standardised into one schema (`source, entity, supplier, date, amount, category, sub_category, record_type, transaction_id`):

| Source | Rows (post-clean) | Coverage | Notes |
|---|---|---|---|
| NHS England (national trust spend, FOI ≥£25k+ disclosures) | 249,613 | 2019-01 – 2024-12 | Primary source; near-complete |
| Bradford Teaching Hospitals NHS FT | 13,526 | 2019-01 – 2024-12 | Trust-level FOI transparency data |
| United Lincolnshire Hospitals NHS Trust | 8,204 | 2019-01 – 2024-12 | Raw file mixes ISO and UK (`dd/mm/yyyy`) date formats across monthly sub-files; this is handled explicitly in `loaders._parse_mixed_dates` |
| UK Contracts Finder (OCDS contract notices) | 55,648 | 2019-01 – 2024-12 | `tender_procurementMethod` used as the single-source/direct-award signal for R1 in Stage 6 |

Merged analytical panel: **326,991 rows** (271,343 trust-spend + 55,648 contract notices), from 349,584 raw rows (93.5% retained).

Cleaning steps: drop null separator rows, parse mixed date formats, convert comma/currency-formatted amount strings to numeric, remove non-positive and extreme-outlier amounts (>£50m for trust spend, >£2bn for contract notices; treated as data-entry/framework-ceiling artefacts), normalise supplier names (case, punctuation, company-type suffixes) for cross-record matching.

**COVID-19 period boundaries** (UK government lockdown/reopening dates):
- Pre-COVID: 2019-01-01 → 2020-03-22
- COVID: 2020-03-23 → 2022-02-23
- Post-COVID: 2022-02-24 → 2024-12-31

**Stage 3: Exploratory & shock analysis.**
- *Supplier concentration (HHI)*: computed monthly per source. National NHS England aggregate HHI is naturally low (median 113, with hundreds of distinct trusts/suppliers and never above 160 in any month); the individually-tracked small trusts show genuine concentration swings that cross the HHI=2500 "highly concentrated" threshold (Hirschman, 1945) in individual months: Lincolnshire in 15 of 70 months (median 1,967, max 10,000 in a month with a single supplier), Bradford in 4 of 72 (median 602, max 5,574). The exceedances are episodic rather than sustained.
- *STL decomposition* (Cleveland et al., 1990) on monthly aggregate trust spend, baseline window 2018-07 – 2019-12 (extended pre-period for a more stable seasonal/trend fit), quantifies the COVID shock: **+33.4% average deviation from the pre-COVID baseline during COVID, +40.5% post-COVID**. Spend did not revert after the acute pandemic period.
- *New-supplier rate*: the proportion of transactions going to a supplier never seen before for that entity. A 6-month burn-in window is excluded from this flag to avoid left-censoring bias (every supplier looks "new" in the first observed months purely because there is no earlier history to compare against; this was found and corrected during development, see `clean_merge.py` comments). Result: new-supplier rate rises from **1.29% (pre-COVID) to 3.28% (COVID)**, dropping back to 1.85% post-COVID. This is consistent with the emergency/direct-award procurement risk documented in Transparency International UK's [Track and Trace](https://www.transparency.org.uk/sites/default/files/2024-11/Track%20and%20Trace%20-%20Transparency%20International%20UK.pdf) (2021) and its 2024 follow-up investigation, ["Landmark investigation finds corruption red flags in £15.3 billion of UK COVID contracts"](https://www.transparency.org.uk/news/report-landmark-investigation-finds-corruption-red-flags-ps153-billion-uk-covid-contracts), both of which flag contracts awarded to companies with no prior track record as a key corruption-risk indicator.

**Stage 4: Unsupervised anomaly detection.** Isolation Forest (Liu, Ting & Zhou, 2008; `n_estimators=300, contamination=0.02, random_state=42`) trained **only on pre-COVID trust-spend records** so it learns "normal" pre-pandemic behaviour, then scores every record. Features: log-amount, within-source-category amount z-score, days since the supplier's last transaction, transaction sequence number for that supplier, new-supplier flag, month, day-of-week.

Note: an earlier iteration also included ordinal-encoded `source`/`category` as raw features. SHAP analysis showed ~60% of top anomalies were then driven almost entirely by which (small) dataset a record came from, not genuine spend anomalies; this was an artefact of NHS England dominating the training sample. These identity features were removed; `amount_zscore_category` already captures the useful within-group signal without that bias.

Records are flagged as anomalous at the 98th percentile of anomaly score (score cut-off 0.0303). Result: **5,427 of 271,343 records (2.00%) flagged overall, but the rate rises by an order of magnitude between periods: 0.21% pre-COVID (training baseline) → 1.58% COVID → 2.79% post-COVID**, indicating growing divergence from pre-pandemic "normal" procurement behaviour that persists well after the acute crisis.

**Stage 5: Explainability.** SHAP `TreeExplainer` (Lundberg & Lee, 2017) computed on all flagged anomalies plus a contrast sample of normal records. Across the top 200 anomalies, the dominant driver is `supplier_txn_seq` (179/200), i.e. transactions very early in a new supplier relationship, consistent with the new-supplier-rate finding, followed by category-relative amount z-score (8), transaction recency (6), log-amount (4), and new-supplier status (3).

**Stage 6: Validation.** Confidential NAO / NHS Counter Fraud Authority case-level audit data is not publicly accessible, so this project uses a defensible literature-derived proxy: four explicit rule-based red flags (direct-award COVID contracts, >3× historical-median price spikes, new-supplier + large COVID contract, suspiciously round invoice amounts). These map onto the corruption-risk indicators published in Transparency International UK's [Track and Trace](https://www.transparency.org.uk/sites/default/files/2024-11/Track%20and%20Trace%20-%20Transparency%20International%20UK.pdf) (2021), which found over 20% of £18bn UK pandemic procurement contracts raised at least one red flag, predominantly single-source/direct-award and no-track-record-supplier signals. The indicators also map onto the NHS Counter Fraud Authority's published fraud-risk quick guides, including [Preventing procurement fraud in the NHS](https://cfa.nhs.uk/resources/downloads/documents/fraud-reports/Preventing_procurement.pdf) (2022) and [Buying goods and services](https://cfa.nhs.uk/resources/downloads/guidance/fraud-awareness/quick-reference-guides/Buying_goods-and-services.pdf) (2026), which list invoice manipulation, high-risk/no-track-record suppliers, and non-PO spend as key vulnerability areas. **59.6% of ML-flagged anomalies (3,237 of 5,427) are corroborated by at least one independent rule** (vs. a 17.5% base rate if rule-flags were randomly distributed; a 3.4× enrichment), providing triangulated construct validity for the unsupervised model without needing access to confidential audit case data.

**Stage 6A: Multi-method comparison.** Isolation Forest is benchmarked against three alternative unsupervised detectors trained on the same pre-COVID feature set: Local Outlier Factor (Breunig et al., 2000), One-Class SVM (Schölkopf et al., 2001), and an MLP-Autoencoder (reconstruction-error based), each calibrated to flag ~2% of records for a like-for-like comparison (`src/modeling/method_comparison.py`). Pairwise Jaccard agreement shows Isolation Forest is a clear outlier relative to the other three: IF vs. LOF = 0.093, IF vs. One-Class SVM = 0.092, IF vs. Autoencoder = 0.071, while LOF/One-Class SVM/Autoencoder cluster tightly together (pairwise Jaccard 0.42–0.67). This indicates Isolation Forest's random-partitioning mechanism captures a qualitatively different notion of anomaly (isolable via few feature splits) than the density-, boundary-, and reconstruction-based methods, which instead agree with each other on which records look unusual relative to the bulk distribution. The three alternatives also fail to reproduce the headline period ordering: LOF and One-Class SVM both flag COVID-period records (3.91% / 3.43%) *more* often than post-COVID ones (1.56% / 1.82%), and flag essentially nothing pre-COVID (0.00%), whereas Isolation Forest's pre < COVID < post ordering is monotonic. A consensus flag (≥2 of 4 methods agreeing) identifies 5,751/271,343 records (2.12%) as anomalous by multiple independent detection paradigms.

**Stage 6B: Synthetic anomaly injection evaluation.** Because no ground-truth fraud labels exist, detector sensitivity is additionally benchmarked using a synthetic-injection design (following the simulation-study approach in [Anomaly Detection Using Unsupervised ML Algorithms: A Simulation Study](https://scholarworks.utrgv.edu/mss_fac/560/)): 400 synthetic anomalies (136 invoice-inflation, 134 ghost-vendor-burst, 130 round-number-kickback) are injected into a 20,000-record held-out sample, and each method's ability to recover the known injected records is measured (`src/modeling/synthetic_evaluation.py`). Isolation Forest substantially outperforms the alternatives: precision/recall/F1 = 0.243 vs. 0.020–0.023 for LOF/One-Class SVM/Autoencoder, and PR-AUC = 0.135 vs. 0.032–0.057. The advantage is confined almost entirely to the ghost-vendor-burst injection type (a new supplier with zero transaction recency and an unusually large amount): Isolation Forest recalls 54.5% of those injected cases vs. 0.0% for all three other methods, showing its random-partitioning mechanism is far better suited to isolating a small number of jointly-extreme feature values than boundary/density/reconstruction-based approaches. On the two purely amount-based injection types it is much weaker (invoice-inflation recall 12.5%, round-number-kickback 5.4%); this is an honest limitation. The model detects *relationship-novelty* anomalies well and *amount-manipulation* anomalies poorly, which is consistent with the SHAP finding that `supplier_txn_seq` dominates its decisions.

**Stage 6C: Statistical significance testing.** Three tests quantify whether the Stage 3/4 findings are statistically robust rather than sampling noise (`src/analysis/statistical_tests.py`): (i) an exact hypergeometric test on the ML/rule-based triangulation overlap (N=326,991, K=57,060 rule-flagged, n=5,427 ML-flagged, observed overlap=3,237 vs. 947.0 expected by chance) gives log(p) ≈ −2,472.7. The overlap is astronomically unlikely to be chance; (ii) 2,000-resample bootstrap 95% confidence intervals around the Isolation Forest anomaly rate and new-supplier rate by period (e.g. COVID anomaly rate 1.58% [1.49%, 1.66%] vs. pre-COVID 0.21% [0.17%, 0.25%], with non-overlapping intervals; STL deviation COVID +33.4% [+23.8%, +42.9%] vs. pre-COVID +0.02% [−5.2%, +5.4%]); (iii) Mann-Whitney U tests on the full anomaly-score distributions between periods, all highly significant (p≈0), with rank-biserial effect sizes of 0.45 (pre-COVID vs. COVID), 0.62 (pre-COVID vs. post-COVID), and 0.23 (COVID vs. post-COVID). These results indicate a large, durable shift in the score distribution that persists after the acute pandemic period.

**Stage 6D: Supplier-buyer network analysis.** A bipartite buyer-supplier graph (8 buyer trusts, 5,597 suppliers, 6,122 edges with ≥2 transactions) is built to test whether anomalies concentrate among well-connected "hub" suppliers, a pattern associated with collusive/incumbent-favouring behaviour in procurement-fraud network literature ([Graph Data Mining for Detecting Collusions in Bidding](https://sol.sbc.org.br/index.php/sbbd_estendido/article/download/30799/30602); [Public Procurement Fraud Detection: A Review Using Network Analysis](https://www.academia.edu/125244008/Public_Procurement_Fraud_Detection_A_Review_Using_Network_Analysis)). Hub suppliers (375 multi-trust suppliers, degree≥2, unioned with 280 high-volume suppliers, top-5%-by-transaction-volume, threshold 197 transactions; n=568 total) are identified via degree and betweenness centrality (Freeman, 1977) on the bipartite projection (Borgatti & Everett, 1997), and a same-side co-occurrence projection graph (suppliers linked when they share a buyer+category+month pool) is partitioned into communities using greedy modularity maximisation (Clauset, Newman & Moore, 2004; modularity=0.650, 32 communities). Contrary to the collusion-hypothesis expectation, **hub suppliers show a markedly LOWER anomaly rate (1.02%, n=340) than non-hub suppliers (2.89%, n=3,556)**, with Mann-Whitney p=1.26×10⁻¹⁸. This suggests that the anomalies detected in this dataset concentrate among smaller, newer, single-relationship suppliers rather than large, well-established incumbents. Community-level anomaly rates vary more than 3-fold among communities of meaningful size (community 7, n=59: 3.43% vs. community 3, the largest community with n=249: 1.15%, against an overall rate of 2.00%), indicating that certain buyer/category/month supplier pools carry disproportionate anomaly risk even though the top-level hub/non-hub split does not.

## Phase 7: Composite risk scoring, category deep dive, robustness, BI delivery

Four analyses extend the Phase 6 advanced-methods work by addressing supplier-level prioritisation, category-specific exposure, specification sensitivity, and non-technical dissemination:

**7A: Composite supplier risk score.** Blends anomaly rate, mean anomaly-score magnitude, and rule-flag rate (each percentile-ranked 0–100, following the composite-indicator approach in Fazekas, Tóth & King, 2016, and IMF WP 2022/094) into a single ranked score per supplier. 3,682 of 7,713 suppliers had enough transactions to be scored: **Low = 2,209, Medium = 920, High = 368, Critical = 185**. Network hub status is deliberately excluded from the score (Stage 6D showed hubs have a *lower* raw anomaly rate) and reported only descriptively. Yet on the *composite* score, hub suppliers actually rank **higher** (64.08 vs. 48.61, p=3.6×10⁻⁵⁶), because the composite also captures accumulated rule-flag exposure across a supplier's full (much larger) transaction history, not just per-transaction anomaly propensity. Both findings are correct and are discussed together, not treated as a contradiction.

**7B: Category-level deep dive.** Disaggregates spend, anomaly rate, new-supplier rate, and HHI by `category × period`. Top COVID-shock growth categories: "CompSwrPrch Additions" (+2,259.8%, £124,429 → £2.94m), "Med & Surg Equip General" (+2,054.8%, £308,125 → £6.64m; a genuinely large absolute increase consistent with the NAO's documented PPE/medical-equipment procurement surge), and "MSSE Consumables" (+1,288.7%, £197,929 → £2.75m).

**7C: Robustness checks.** Threshold sensitivity (95th/98th/99th percentile), COVID period-boundary sensitivity (±1 month), and feature-set ablation (drop `is_new_supplier`) all confirm the pre-COVID < COVID < post-COVID anomaly-rate ordering is stable, and that the model's overall record ranking survives feature ablation (Spearman ρ=0.925) even though the specific flagged set shifts somewhat (Jaccard=0.707).

**7D/7E: BI-ready export and interactive dashboard.** A star-schema CSV export (`data/processed/bi_export/`: 1 fact + 4 dimension tables) and a self-contained interactive Plotly dashboard (`reports/nhs_procurement_dashboard.html`) present the pipeline's results in a business-intelligence format accessible to non-technical stakeholders. **Format limitation:** a native `.pbix` (Power BI) file cannot be reliably generated programmatically because it is a proprietary binary format and no Python library can write one; a `.twbx` (Tableau) file, although theoretically constructible, cannot be validated without Tableau installed. The project therefore provides clean CSVs, a functioning interactive HTML dashboard, and a field-by-field [Tableau/Power BI build guide](docs/bi_dashboard_guide.md). Implementation: `src/analysis/supplier_risk_score.py`, `category_deep_dive.py`, `robustness_checks.py`, `bi_export.py`, `dashboard.py`. Full methodology and results write-up: `docs/dissertation_sections.md`.

## Key results at a glance

| Metric | Pre-COVID | COVID | Post-COVID |
|---|---|---|---|
| Avg. monthly trust spend, % deviation from baseline (STL) | +0.02% | **+33.4%** | +40.5% |
| New-supplier rate (% of transactions) | 1.29% | **3.28%** | 1.85% |
| Isolation Forest anomaly rate | 0.21% | 1.58% | **2.79%** |

ML/rule-based triangulation: **3,237 / 5,427 (59.6%)** ML-flagged anomalies corroborated by ≥1 independent audit red-flag rule (hypergeometric exact test log(p) ≈ −2,472.7; not attributable to chance).

| Advanced-methods metric | Result |
|---|---|
| Method agreement (pairwise Jaccard), Isolation Forest vs. others | 0.07–0.09 (outlier) |
| Method agreement (pairwise Jaccard), LOF / One-Class SVM / Autoencoder | 0.42–0.67 (cluster together) |
| Consensus anomalies (≥2 of 4 methods agree) | 5,751 / 271,343 (2.12%) |
| Synthetic injection F1: Isolation Forest vs. best alternative | 0.243 vs. 0.023 (LOF / One-Class SVM) |
| Synthetic recall, ghost-vendor-burst type: Isolation Forest vs. others | 54.5% vs. 0.0% |
| Hub-supplier anomaly rate vs. non-hub | 1.02% (n=340) vs. **2.89%** (n=3,556), p=1.26×10⁻¹⁸ |
| Network communities detected (modularity) | 32 communities, Q=0.650 |

| Phase 7 metric | Result |
|---|---|
| Supplier risk tiers (of 3,682 scored suppliers) | Low=2,209, Medium=920, High=368, Critical=185 |
| Hub vs. non-hub, composite risk score (contrast with raw anomaly rate above) | 64.08 (n=334) vs. 48.61 (n=3,348), p=3.64×10⁻⁵⁶ |
| Top COVID-shock category by % growth | "CompSwrPrch Additions" +2,259.8% (£124,429 → £2.94m) |
| Largest genuine absolute-spend COVID-shock category | "Med & Surg Equip General" +2,054.8% (£308k → £6.64m) |
| Robustness: period ordering stable across thresholds | 95th/98th/99th percentile all preserve pre<COVID<post |
| Robustness: feature ablation (drop `is_new_supplier`) | Jaccard=0.707, Spearman ρ=0.925 |
| BI export | 1 fact table (326,991 rows) + 4 dimension tables, star schema |

Figures (see `docs/figures/`):
- `stl_decomposition.png`: observed/trend/seasonal/residual monthly spend with COVID period shaded
- `hhi_trend.png`: supplier concentration by source over time
- `new_supplier_rate.png`: new-supplier onboarding rate, COVID spike clearly visible
- `anomaly_timeline.png`: monthly Isolation Forest anomaly rate
- `shap_top_feature_summary.png`: dominant SHAP feature across top anomalies
- `method_agreement_heatmap.png`: pairwise Jaccard agreement between the four anomaly detectors
- `synthetic_precision_recall.png`: precision/recall/F1 on the synthetic injection benchmark, by method
- `network_hub_comparison.png`: hub vs. non-hub supplier anomaly rate, and transaction-volume vs. anomaly-rate scatter
- `community_anomaly_rate.png`: top 10 supplier co-occurrence communities ranked by mean anomaly rate
- `supplier_risk_score.png`: composite supplier risk tier distribution and top-ranked suppliers
- `category_covid_shock.png`: top categories by pre-COVID → COVID spend growth (absolute and % framing)
- `robustness_sensitivity.png`: threshold, period-boundary, and feature-ablation sensitivity results

## Data representation / BI dashboard

In addition to the CSV and figure outputs, the project provides a BI-oriented data representation: a star-schema CSV export (`data/processed/bi_export/`: `fact_transactions.csv` plus `dim_supplier.csv`, `dim_category.csv`, `dim_period.csv`, `dim_month.csv`) and a self-contained interactive dashboard (`reports/nhs_procurement_dashboard.html`) that reproduces a Tableau/Power BI-style multi-panel layout in a browser without requiring BI software.

**Stated limitation:** a native Power BI `.pbix` file is a proprietary binary format that no Python library can generate, and a Tableau `.twbx` file, while technically a buildable zip archive, cannot be validated to open correctly without Tableau installed. The project consequently supplies the CSV export, the functioning HTML dashboard, and [`docs/bi_dashboard_guide.md`](docs/bi_dashboard_guide.md). This is a field-by-field guide for reconstructing the dashboard natively in Power BI Desktop or Tableau Desktop from the exported CSVs.

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. (Optional) Place the 4 consolidated source CSVs in data/raw/ yourself:
#    bradford_clean.csv, lincolnshire_clean.csv, nhs_england_clean.csv, contracts_clean.csv
#    If any are missing, step 2 fetches and rebuilds them automatically from the
#    project's Google Drive raw archive; see "Reproducing raw data from Google Drive".

# 2. Run the full pipeline end-to-end (~105s on 2 vCPU / 8GB RAM, incl. Phases 6A-6D)
python -m src.run_pipeline

# 3. Generate report figures
python -m src.analysis.make_figures

# 4. Run the test suite
pytest tests/ -v
```

The pipeline (step 2) now also runs Phase 7A-7E and writes the BI-ready CSV export to `data/processed/bi_export/` and the interactive dashboard to `reports/nhs_procurement_dashboard.html`.

### Reproducing the data

The 4 raw CSVs (~106MB total) are gitignored because they are third-party FOI/Contracts Finder exports rather than code and are fully regenerable from public sources:
- NHS England / Bradford / Lincolnshire trust spend: published under each trust's Freedom-of-Information "spend over £25k" transparency disclosures.
- UK Contracts Finder: [OCDS-format contract notices](https://www.contractsfinder.service.gov.uk/) for NHS/DHSC buyers, 2019–2024.

Place the 4 consolidated CSVs directly in `data/raw/` with the filenames referenced in `src/config.py`'s `RAW_FILES` dict, then run the pipeline. Alternatively, let Phase 1 rebuild them for you from the original per-month source files; see the next section.

### Reproducing raw data from Google Drive

Phase 1 (`src/data_engineering/build_raw_from_drive.py`) reconstructs the four `data/raw/*_clean.csv` files from the **original** per-month FOI exports and Contracts Finder bulk extracts held in a public read-only Google Drive folder (link in `src/config.py`'s `GOOGLE_DRIVE_RAW_FOLDER_URL`). This preserves a reproducible provenance chain from original monthly publication through consolidation, cleaning, modelling, and results, using the Drive archive and a single command.

It is **automatic when required and optional when invoked independently**:

- `python -m src.run_pipeline` checks for the four files first. If any is missing it runs Phase 1, then continues into Phase 2 unchanged.
- If all four are already present, Phase 1 is skipped silently; an existing manual `data/raw/` setup behaves exactly as before.
- To run or force it standalone:

```bash
python -m src.data_engineering.build_raw_from_drive           # build only if files are missing
python -m src.data_engineering.build_raw_from_drive --force   # always rebuild
```

Notes:

- No credentials or API key are needed (the folder is shared "anyone with the link"); `gdown` is the only extra dependency.
- Only the files actually used are downloaded (~500MB): the `.csv` publication of each month rather than its `.xls`/`.xlsx` duplicate, and only `main.csv`/`awards.csv`/`awards_suppliers.csv` from each year of the Contracts Finder export (the other 10 files per year are large and unused). Downloads are cached in `data/_raw_staging/` (gitignored), so re-runs only fetch what is missing.
- Anonymous Drive access is rate-limited, so a first full run may log per-file failures and retries; individual failures are skipped rather than aborting the run, and simply re-running the module fills the gaps.
- Despite the historical `_clean.csv` filenames, Phase 1's outputs are **consolidated raw**, rather than analytically clean: the stage only concatenates monthly files and standardises column names. All cleaning (dropna, mixed date parsing, amount coercion, deduplication) remains in `loaders.py`/`clean_merge.py`.

**Contracts Finder caveat.** The Drive archive contains the *full UK national* Contracts Finder export (~50,000–77,000 notices per year across every public-sector buyer), rather than a health-only extract. Phase 1 therefore applies a keyword filter (`nhs|health|hospital|clinical|ambulance|blood|commissioning support|...` against buyer name, tender title and description) to define a health-sector subset. This is a documented **best-effort reconstruction** of the original bespoke extract rather than a byte-exact replay: it recovers 99.9% of the rows in the original `contracts_clean.csv` but is deliberately broader, admitting adjacent health buyers (Public Health England, DHSC, Health Education England, devolved health boards) excluded by the original extract. Contract-notice row counts consequently differ from the legacy file; trust-spend rows and all Isolation Forest anomaly statistics are unaffected because `load_contracts_finder()` contributes only `record_type == "contract_notice"` rows and the detector is trained and scored on trust spend only. See `docs/dissertation_sections.md` → "Phase 1" for the full quantified comparison and limitation statement.

## Data quality caveats and limitations

- United Lincolnshire Hospitals and Bradford Teaching Hospitals both now cover the full Jan 2019 – Dec 2024 study window; an earlier iteration of the raw data had these trusts truncated (Lincolnshire to Jul 2021, Bradford to Feb 2022) due to a date-parsing defect in the manually-prepared raw files, since fixed by the Phase 1 rebuild. See `docs/dissertation_sections.md` for the full before/after comparison.
- UK Contracts Finder `awards.csv`/`awards_suppliers.csv` sub-files (award-level supplier names) are sparse and were not incorporated into the anomaly-detection panel; only the `main.csv` buyer/tender-notice level is used.
- The rule-based validation in Stage 6 is a literature-derived **proxy** for genuine audit ground truth, not a substitute for it. The overlap statistic indicates triangulated plausibility, not confirmed fraud/error.
- Amount caps (>£50m trust spend, >£2bn contract notices treated as data artefacts) are a modelling judgement call; see `src/data_engineering/clean_merge.py` for the exact thresholds and rationale.
- The synthetic injection benchmark (Stage 6B) evaluates detectors against three specific, hand-designed anomaly archetypes; it demonstrates relative sensitivity to those archetypes, not a general-purpose fraud-detection accuracy estimate. Real anomalies may take forms not represented by the three injection types.
- The network hub/non-hub finding (Stage 6D) is descriptive, not causal. A lower anomaly rate among hub suppliers could reflect genuinely cleaner large-incumbent behaviour, or could reflect the Isolation Forest model's features (e.g. transaction recency, new-supplier flag) being structurally less likely to fire for high-frequency, long-established suppliers. This is flagged explicitly as a direction for further work rather than presented as evidence against a collusion hypothesis.
- Bootstrap confidence intervals on the STL % deviation figures are indicative rather than exact, since monthly time series values are autocorrelated and the naive bootstrap assumes independent observations.

## References

- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. *Journal of Official Statistics*, 6(1), 3–73.
- Hirschman, A. O. (1945). *National Power and the Structure of Foreign Trade*. University of California Press. (Herfindahl-Hirschman Index)
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *2008 Eighth IEEE International Conference on Data Mining*, 413–422. https://doi.org/10.1109/ICDM.2008.17
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.
- Transparency International UK. (2021). *Track and Trace: Procurement for the COVID-19 Pandemic*. https://www.transparency.org.uk/sites/default/files/2024-11/Track%20and%20Trace%20-%20Transparency%20International%20UK.pdf
- Transparency International UK. (2024). *Landmark investigation finds corruption red flags in £15.3 billion of UK COVID contracts*. https://www.transparency.org.uk/news/report-landmark-investigation-finds-corruption-red-flags-ps153-billion-uk-covid-contracts
- NHS Counter Fraud Authority. (2022). *Preventing Procurement Fraud in the NHS*. https://cfa.nhs.uk/resources/downloads/documents/fraud-reports/Preventing_procurement.pdf
- NHS Counter Fraud Authority. (2026). *Buying Goods and Services (Quick Reference Guide)*. https://cfa.nhs.uk/resources/downloads/guidance/fraud-awareness/quick-reference-guides/Buying_goods-and-services.pdf
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, 93–104. https://doi.org/10.1145/335191.335388
- Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J.
