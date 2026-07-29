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
    method_comparison.py       multi-method anomaly detector comparison (IF, LOF, One-Class SVM, MLP-Autoencoder)
    synthetic_evaluation.py    synthetic anomaly injection + precision/recall/F1/PR-AUC evaluation
  network/
    supplier_network.py        bipartite buyer-supplier graph, centrality, co-occurrence community detection
  validation/
    audit_validation.py    rule-based red-flag validation + ML/rule triangulation
  analysis/
    statistical_tests.py    hypergeometric exact test, bootstrap CIs, Mann-Whitney U
    supplier_risk_score.py  composite supplier risk score (Phase 6A)
    category_deep_dive.py   category x period deep dive + COVID-shock ranking (Phase 6B)
    robustness_checks.py    threshold / period-boundary / feature-ablation sensitivity (Phase 6C)
    bi_export.py            BI-ready star-schema CSV export (Phase 6D)
    dashboard.py            interactive Plotly dashboard (Phase 6E)
  run_pipeline.py           orchestrates all phases end-to-end
tests/                      pytest unit tests (schema, cleaning logic, feature/rule correctness, Phase 5 & 6 modules)
requirements.txt
docs/
  bi_dashboard_guide.md   Tableau/Power BI build guide for the BI export CSVs
  dissertation_sections.md  drop-in methodology/results sections for the dissertation write-up
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

**Stage 5A — Multi-method comparison.** Isolation Forest is benchmarked against three alternative unsupervised detectors trained on the same pre-COVID feature set: Local Outlier Factor (Breunig et al., 2000), One-Class SVM (Schölkopf et al., 2001), and an MLP-Autoencoder (reconstruction-error based), each calibrated to flag ~2% of records for a like-for-like comparison (`src/modeling/method_comparison.py`). Pairwise Jaccard agreement shows Isolation Forest is a clear outlier relative to the other three: IF vs. LOF = 0.060, IF vs. One-Class SVM = 0.058, IF vs. Autoencoder = 0.050, while LOF/One-Class SVM/Autoencoder cluster tightly together (pairwise Jaccard 0.45–0.70). This indicates Isolation Forest's random-partitioning mechanism captures a qualitatively different notion of anomaly (isolable via few feature splits) than the density-, boundary-, and reconstruction-based methods, which instead agree with each other on which records look unusual relative to the bulk distribution. A consensus flag (≥2 of 4 methods agreeing) identifies 5,410/257,095 records (2.10%) as anomalous by multiple independent detection paradigms.

**Stage 5B — Synthetic anomaly injection evaluation.** Because no ground-truth fraud labels exist, detector sensitivity is additionally benchmarked using a synthetic-injection design (following the simulation-study approach in [Anomaly Detection Using Unsupervised ML Algorithms: A Simulation Study](https://scholarworks.utrgv.edu/mss_fac/560/)): 400 synthetic anomalies (136 invoice-inflation, 134 ghost-vendor-burst, 130 round-number-kickback) are injected into a 20,000-record held-out sample, and each method's ability to recover the known injected records is measured (`src/modeling/synthetic_evaluation.py`). Isolation Forest substantially outperforms the alternatives — precision/recall/F1 = 0.315 vs. 0.018–0.040 for LOF/One-Class SVM/Autoencoder, and PR-AUC = 0.182 vs. 0.025–0.049. The advantage is most dramatic for the ghost-vendor-burst injection type (a new supplier with zero transaction recency and an unusually large amount): Isolation Forest recalls 73.1% of injected cases vs. 0–3.0% for the other three methods, showing its random-partitioning mechanism is far better suited to isolating a small number of jointly-extreme feature values than boundary/density/reconstruction-based approaches.

**Stage 5C — Statistical significance testing.** Three tests quantify whether the Stage 2/3 findings are statistically robust rather than sampling noise (`src/analysis/statistical_tests.py`): (i) an exact hypergeometric test on the ML/rule-based triangulation overlap (N=288,071, K=54,425 rule-flagged, n=5,142 ML-flagged, observed overlap=3,226 vs. 971.5 expected by chance) gives log(p) ≈ −2,445 — the overlap is astronomically unlikely to be chance; (ii) 2,000-resample bootstrap 95% confidence intervals around the Isolation Forest anomaly rate and new-supplier rate by period (e.g. COVID anomaly rate 1.82% [1.73%, 1.91%] vs. pre-COVID 0.37% [0.32%, 0.43%] — non-overlapping intervals); (iii) Mann-Whitney U tests on the full anomaly-score distributions between periods, all highly significant (p≈0), with rank-biserial effect sizes of 0.44 (pre-COVID vs. COVID), 0.61 (pre-COVID vs. post-COVID), and 0.22 (COVID vs. post-COVID) — a large, durable shift in the score distribution that persists after the acute pandemic period.

**Stage 5D — Supplier-buyer network analysis.** A bipartite buyer-supplier graph (8 buyer trusts, 5,286 suppliers, 5,691 edges with ≥2 transactions) is built to test whether anomalies concentrate among well-connected "hub" suppliers, a pattern associated with collusive/incumbent-favouring behaviour in procurement-fraud network literature ([Graph Data Mining for Detecting Collusions in Bidding](https://sol.sbc.org.br/index.php/sbbd_estendido/article/download/30799/30602); [Public Procurement Fraud Detection: A Review Using Network Analysis](https://www.academia.edu/125244008/Public_Procurement_Fraud_Detection_A_Review_Using_Network_Analysis)). Hub suppliers (multi-trust suppliers, degree≥2, OR top-5%-by-transaction-volume suppliers; n=476) are identified via degree and betweenness centrality (Freeman, 1977) on the bipartite projection (Borgatti & Everett, 1997), and a same-side co-occurrence projection graph (suppliers linked when they share a buyer+category+month pool) is partitioned into communities using greedy modularity maximisation (Clauset, Newman & Moore, 2004; modularity=0.635, 32 communities). Contrary to the collusion-hypothesis expectation, **hub suppliers show a markedly LOWER anomaly rate (1.16%, n=315) than non-hub suppliers (3.64%, n=3,535)** — Mann-Whitney p=5.4×10⁻²² — suggesting the anomalies detected in this dataset concentrate among smaller, newer, single-relationship suppliers rather than large, well-established incumbents. Community-level anomaly rates vary more than 5-fold (community 7: 8.25% vs. community 0: 1.48%, against an overall rate of 2.00%), indicating that certain buyer/category/month supplier pools carry disproportionate anomaly risk even though the top-level hub/non-hub split does not.

## Phase 6 — Composite risk scoring, category deep dive, robustness, BI delivery

Four further analyses were added on top of the Phase 5 advanced-methods work, answering "what else can we do with this dataset, and how do we hand it to a non-technical stakeholder":

**6A — Composite supplier risk score.** Blends anomaly rate, mean anomaly-score magnitude, and rule-flag rate (each percentile-ranked 0–100, following the composite-indicator approach in Fazekas, Tóth & King, 2016, and IMF WP 2022/094) into a single ranked score per supplier. 3,456 of 7,244 suppliers had enough transactions to be scored: **Low = 2,073, Medium = 864, High = 346, Critical = 173**. Network hub status is deliberately excluded from the score (Stage 5D showed hubs have a *lower* raw anomaly rate) and reported only descriptively — yet on the *composite* score, hub suppliers actually rank **higher** (64.80 vs. 48.54, p=4.9×10⁻⁵⁸), because the composite also captures accumulated rule-flag exposure across a supplier's full (much larger) transaction history, not just per-transaction anomaly propensity. Both findings are correct and are discussed together, not treated as a contradiction.

**6B — Category-level deep dive.** Disaggregates spend, anomaly rate, new-supplier rate, and HHI by `category × period`. Top COVID-shock growth categories: "Computer Hardware Purch" (+47,553.6% — flagged as a small-base artefact, pre-COVID spend was only £556.80), "Med & Surg Equip General" (+2,302.4%, a genuinely large absolute increase consistent with the NAO's documented PPE/medical-equipment procurement surge), and "CompSwrPrch Additions" (+2,259.8%).

**6C — Robustness checks.** Threshold sensitivity (95th/98th/99th percentile), COVID period-boundary sensitivity (±1 month), and feature-set ablation (drop `is_new_supplier`) all confirm the pre-COVID < COVID < post-COVID anomaly-rate ordering is stable, and that the model's overall record ranking survives feature ablation (Spearman ρ=0.919) even though the specific flagged set shifts somewhat (Jaccard=0.624).

**6D/6E — BI-ready export and interactive dashboard.** A star-schema CSV export (`data/processed/bi_export/`: 1 fact + 4 dimension tables) and a self-contained interactive Plotly dashboard (`reports/nhs_procurement_dashboard.html`) deliver the pipeline's results in a business-intelligence-style, non-technical format. **Honest limitation, stated explicitly:** a genuine `.pbix` (Power BI) file cannot be reliably hand-generated (proprietary binary format, no Python library can write one), and a `.twbx` (Tableau) file, while theoretically constructible, cannot be verified to actually open without Tableau installed. Rather than risk a broken or fabricated binary, this project ships clean CSVs, a working interactive HTML dashboard, and a field-by-field [Tableau/Power BI build guide](docs/bi_dashboard_guide.md) instead. Implementation: `src/analysis/supplier_risk_score.py`, `category_deep_dive.py`, `robustness_checks.py`, `bi_export.py`, `dashboard.py`. Full methodology and results write-up: `docs/dissertation_sections.md`.

## Key results at a glance

| Metric | Pre-COVID | COVID | Post-COVID |
|---|---|---|---|
| Avg. monthly trust spend, % deviation from baseline (STL) | −0.03% | **+33.0%** | +38.6% |
| New-supplier rate (% of transactions) | 1.32% | **3.21%** | 1.71% |
| Isolation Forest anomaly rate | 0.37% | 1.82% | **2.64%** |

ML/rule-based triangulation: **3,226 / 5,142 (62.7%)** ML-flagged anomalies corroborated by ≥1 independent audit red-flag rule (hypergeometric exact test log(p) ≈ −2,445 — not attributable to chance).

| Advanced-methods metric | Result |
|---|---|
| Method agreement (pairwise Jaccard), Isolation Forest vs. others | 0.05–0.06 (outlier) |
| Method agreement (pairwise Jaccard), LOF / One-Class SVM / Autoencoder | 0.45–0.70 (cluster together) |
| Consensus anomalies (≥2 of 4 methods agree) | 5,410 / 257,095 (2.10%) |
| Synthetic injection F1 — Isolation Forest vs. best alternative | 0.315 vs. 0.040 (One-Class SVM) |
| Synthetic recall, ghost-vendor-burst type — Isolation Forest vs. others | 73.1% vs. 0–3.0% |
| Hub-supplier anomaly rate vs. non-hub | 1.16% (n=315) vs. **3.64%** (n=3,535), p=5.4×10⁻²² |
| Network communities detected (modularity) | 32 communities, Q=0.635 |

| Phase 6 metric | Result |
|---|---|
| Supplier risk tiers (of 3,456 scored suppliers) | Low=2,073, Medium=864, High=346, Critical=173 |
| Hub vs. non-hub, composite risk score (contrast with raw anomaly rate above) | 64.80 (n=313) vs. 48.54 (n=3,143), p=4.9×10⁻⁵⁸ |
| Top COVID-shock category by % growth | "Computer Hardware Purch" +47,553.6% (small-base artefact, pre-COVID spend only £556.80) |
| Largest genuine absolute-spend COVID-shock category | "Med & Surg Equip General" +2,302.4% (£276k → £6.64m) |
| Robustness — period ordering stable across thresholds | 95th/98th/99th percentile all preserve pre<COVID<post |
| Robustness — feature ablation (drop `is_new_supplier`) | Jaccard=0.624, Spearman ρ=0.919 |
| BI export | 1 fact table (288,071 rows) + 4 dimension tables, star schema |

Figures (see `docs/figures/`):
- `stl_decomposition.png` — observed/trend/seasonal/residual monthly spend with COVID period shaded
- `hhi_trend.png` — supplier concentration by source over time
- `new_supplier_rate.png` — new-supplier onboarding rate, COVID spike clearly visible
- `anomaly_timeline.png` — monthly Isolation Forest anomaly rate
- `shap_top_feature_summary.png` — dominant SHAP feature across top anomalies
- `method_agreement_heatmap.png` — pairwise Jaccard agreement between the four anomaly detectors
- `synthetic_precision_recall.png` — precision/recall/F1 on the synthetic injection benchmark, by method
- `network_hub_comparison.png` — hub vs. non-hub supplier anomaly rate, and transaction-volume vs. anomaly-rate scatter
- `community_anomaly_rate.png` — top 10 supplier co-occurrence communities ranked by mean anomaly rate
- `supplier_risk_score.png` — composite supplier risk tier distribution and top-ranked suppliers
- `category_covid_shock.png` — top categories by pre-COVID → COVID spend growth (absolute and % framing)
- `robustness_sensitivity.png` — threshold, period-boundary, and feature-ablation sensitivity results

## Data representation / BI dashboard

Alongside the CSV/figure outputs above, this project delivers a BI-style data representation: a star-schema CSV export (`data/processed/bi_export/`: `fact_transactions.csv` plus `dim_supplier.csv`, `dim_category.csv`, `dim_period.csv`, `dim_month.csv`) and a self-contained interactive dashboard (`reports/nhs_procurement_dashboard.html`) that mimics a Tableau/Power BI multi-panel layout in the browser, no BI software required.

**Stated limitation:** a genuine Power BI `.pbix` file is a proprietary binary that no Python library can generate, and a Tableau `.twbx` file, while technically a buildable zip archive, cannot be verified to open correctly without Tableau installed to check it. Rather than submit an unverifiable or fabricated binary, this project ships the CSV export, the working HTML dashboard, and [`docs/bi_dashboard_guide.md`](docs/bi_dashboard_guide.md) — a field-by-field guide for rebuilding the same dashboard natively in Power BI Desktop or Tableau Desktop from the exported CSVs in a few minutes.

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. Place the 4 raw source CSVs in data/raw/ (see "Reproducing the data" below)
#    bradford_clean.csv, lincolnshire_clean.csv, nhs_england_clean.csv, contracts_clean.csv

# 2. Run the full pipeline end-to-end (~105s on 2 vCPU / 8GB RAM, incl. Phases 5A-5D)
python -m src.run_pipeline

# 3. Generate report figures
python -m src.analysis.make_figures

# 4. Run the test suite
pytest tests/ -v
```

The pipeline (step 2) now also runs Phase 6A-6E and writes the BI-ready CSV export to `data/processed/bi_export/` and the interactive dashboard to `reports/nhs_procurement_dashboard.html`.

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
- The synthetic injection benchmark (Stage 5B) evaluates detectors against three specific, hand-designed anomaly archetypes; it demonstrates relative sensitivity to those archetypes, not a general-purpose fraud-detection accuracy estimate. Real anomalies may take forms not represented by the three injection types.
- The network hub/non-hub finding (Stage 5D) is descriptive, not causal — a lower anomaly rate among hub suppliers could reflect genuinely cleaner large-incumbent behaviour, or could reflect the Isolation Forest model's features (e.g. transaction recency, new-supplier flag) being structurally less likely to fire for high-frequency, long-established suppliers. This is flagged explicitly as a direction for further work rather than presented as evidence against a collusion hypothesis.
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
- Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). Estimating the Support of a High-Dimensional Distribution. *Neural Computation*, 13(7), 1443–1471. https://is.mpg.de/publications/970
- Borgatti, S. P., & Everett, M. G. (1997). Network Analysis of 2-Mode Data. *Social Networks*, 19(3), 243–269. https://works.bepress.com/steveborgatti/17/
- Clauset, A., Newman, M. E. J., & Moore, C. (2004). Finding Community Structure in Very Large Networks. *Physical Review E*, 70, 066111. https://arxiv.org/abs/cond-mat/0408187
- Freeman, L. C. (1977). A Set of Measures of Centrality Based on Betweenness. *Sociometry*, 40(1), 35–41. https://doi.org/10.2307/3033543
- Agyemang, E. F. (2024). Anomaly Detection Using Unsupervised Machine Learning Algorithms: A Simulation Study. *Scientific African*. https://scholarworks.utrgv.edu/mss_fac/560/
- Victor, A. O., Sales, L. A. M., Moreira, R. S., de Moraes, C. E. C., Lima, L. G., Rocha, J. F., Contursi, B. S. N., & Meirelles, T. (2024). Graph Data Mining for Detecting Collusions in Bidding Processes: A Case Study. *Anais Estendidos do XXXIX Simpósio Brasileiro de Bancos de Dados (SBBD 2024)*. https://sol.sbc.org.br/index.php/sbbd_estendido/article/download/30799/30602
- Lyra, M. (2024). Public Procurement Fraud Detection: A Review Using Network Analysis. https://www.academia.edu/125244008/Public_Procurement_Fraud_Detection_A_Review_Using_Network_Analysis
- Fazekas, M., Tóth, I. J., & King, L. P. (2016). An Objective Corruption Risk Index Using Public Procurement Data. *European Journal on Criminal Policy and Research*, 22(3), 369–397. https://doi.org/10.1007/s10610-016-9308-z
- Abdou, A., Basdevant, O., Dávid-Barrett, E., & Fazekas, M. (2022). Assessing Vulnerabilities to Corruption in Public Procurement and Their Price Impact. IMF Working Paper No. 2022/094. https://www.imf.org/en/publications/wp/issues/2022/05/20/assessing-vulnerabilities-to-corruption-in-public-procurement-and-their-price-impact-518197
- National Audit Office. (2020). Investigation into Government Procurement during the COVID-19 Pandemic. HC 959, Session 2019–2021. https://www.nao.org.uk/wp-content/uploads/2020/11/Investigation-into-government-procurement-during-the-COVID-19-pandemic.pdf
- Rhoades, S. A. (1993). The Herfindahl-Hirschman Index. *Federal Reserve Bulletin*, 79(3), 188–189.
- Aggarwal, C. C. (2017). *Outlier Analysis* (2nd ed.). Springer. https://doi.org/10.1007/978-3-319-47578-3
- Emmott, A., Das, S., Dietterich, T., Fern, A., & Wong, W.-K. (2015). A Meta-Analysis of the Anomaly Detection Problem. arXiv:1503.01158. https://arxiv.org/abs/1503.01158
