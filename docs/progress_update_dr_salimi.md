# Project Progress Update: NHS Procurement Anomaly Detection

**To:** Dr Nahid Salimi
**From:** Gunjan Mahadeva Rao (Student ID: 15848655)
**Course:** MSc Data Science, Coventry University
**Date:** 31 July 2026
**Re:** Progress against the proposal timeline: Phases 1–4 complete, entering Phase 5

Dear Dr Salimi,

Following my earlier update confirming the availability of all four datasets, this report summarises the project's current empirical and methodological position. The workflow specified in the proposal, encompassing data engineering, exploratory analysis, Isolation Forest modelling with SHAP-based explanation, and validation against audit findings, has now been completed across all four datasets. In response to your feedback on methodological rigour and project-management detail, I have also undertaken an extended programme of validation and robustness work beyond the original scope.

The update follows the five phases of the proposal's project plan and records the principal implementation issues identified during each phase and the measures used to resolve them.

---

## Phase 1: Data Engineering Pipeline (Weeks 1–2)

**Planned:** Build a Python pipeline (Pandas, scikit-learn) to clean, standardise, and merge all four datasets into a common schema.

**Completed:** The four datasets have been retrieved, cleaned, and harmonised into a common transaction schema (date, supplier, amount, expense type, expense area, entity, department family), with every record carrying a `_dataset_source` field to maintain provenance and traceability.

| Dataset | Raw records retrieved | Source |
|---|---|---|
| NHS England Payments Over £25k | 271,598 | [england.nhs.uk](https://www.england.nhs.uk/publication/payments-over-25k-reports-2024/) |
| Bradford Teaching Hospitals | 13,925 | [bradfordhospitals.nhs.uk](https://www.bradfordhospitals.nhs.uk/our-trust/reports-and-accounts/) |
| United Lincolnshire Hospitals | 8,280 | [ulh.nhs.uk](https://www.ulh.nhs.uk/publication-of-spend-over-25000/) |
| UK Contracts Finder (NHS-relevant) | 81,482 | [data.open-contracting.org](https://data.open-contracting.org/en/publication/128) |

The Bradford, Lincolnshire, and Contracts Finder totals differ slightly from those reported in my earlier email. During the construction of a reproducible pipeline, I conducted a data-completeness audit against the originally published monthly files and identified several processing failures that had suppressed valid records without producing explicit errors:

- **Report-title rows mistaken for missing headers.** A number of the monthly exports are spreadsheet reports saved to CSV, with a report-title line sitting above the actual column headings (for example, all of Bradford's 2023 and 2024 files, and several Lincolnshire files, begin with a title string rather than a header row). Reading these with the default assumption that row one is the header produces a frame whose columns don't match the target schema at all, so the rows are silently carried through as structurally present but entirely empty. No error is thrown; the data simply vanishes. This turned out to be the single largest source of missing data: every Bradford file from 2023 and 2024 was contributing zero usable rows. I fixed it by scanning the first few rows of each file and picking the first row that actually resolves to known column names, rather than assuming row one is always the header.
- **Inconsistent date formats across monthly files.** Bradford published dates as a plain calendar date until 2021 and as a full timestamp from 2022 onwards; Lincolnshire mixes plain dates, timestamps, and UK-format `dd/mm/yyyy` across different months. The default date parser infers a single format from the first row it sees and silently turns every non-matching value into a blank date, which then gets dropped by the completeness filter; consequently, months in one format were being lost simply because a different month earlier in the file used another format. I replaced this with a parser that handles the mix explicitly, plus a dedicated fallback for Lincolnshire's more inconsistent files.
- **Column names that changed over time.** Lincolnshire renamed its spending-category and date columns several times over the six years (its 2024 files switched to labels the original mapping didn't recognise at all), which meant those months carried blank categories and amounts and were discarded outright. NHS England's final few months added a suffix to its own column headers with the same effect. I rebuilt the column-mapping step as an explicit alias table so any future renaming is easy to add without hunting through code.
- **A mislabelled file format.** One Lincolnshire month was saved with a `.csv` extension but is actually tab-separated internally, which broke the parser partway through the file. I added a step that checks the actual delimiter in each file rather than trusting the extension.
- **A month only published as a macro-enabled spreadsheet.** Bradford's January 2019 return exists only as an `.xlsm` file, with no CSV equivalent, and reading it needed an additional library (`openpyxl`) that hadn't been set up initially. Once added, this recovered 101 transactions that had been missing from the study entirely.
- **The Contracts Finder national export needed sector filtering.** Contracts Finder publishes contract notices for all public-sector buyers, not just health, so isolating the NHS-relevant subset required a keyword filter over buyer names and descriptions (matching terms like "NHS", "hospital", "clinical", "ambulance", "commissioning support", etc.). Refining this filter to properly cover all years recovered the full NHS-relevant set.
- **Two source portals briefly inaccessible.** As mentioned previously, Bradford and Lincolnshire were briefly inaccessible while I resolved download issues on their transparency pages; both are now fully retrievable, and I've kept a local, versioned copy of the original source files (shared via Google Drive) so the pipeline no longer depends on the portals staying available.

Each correction was validated by comparing rebuilt row counts, column presence, and aggregate monetary totals against the previously held files before accepting the revised figures. The NHS England figures matched exactly as a positive control (271,598 rows reconciling to the penny on total expenditure), supporting the interpretation that the Bradford, Lincolnshire, and Contracts Finder changes represent recovered records rather than newly introduced errors.

After cleaning, harmonisation, and merging with the new-supplier and COVID-period logic described below, the four sources combine into a single analytical panel of **326,991 records** spanning January 2019 to December 2024 (271,343 trust-spend transactions and 55,648 contract notices).

---

## Phase 2: Exploratory Analysis & Time-Series Decomposition (Weeks 3–5)

**Planned:** Visualise spend distributions and supplier concentration using the Herfindahl–Hirschman Index; apply STL decomposition to quantify how far COVID-19 pushed spending away from its pre-pandemic baseline.

**Completed:** Monthly and source-specific HHI measures have been calculated to characterise supplier concentration. The new-supplier onboarding rate is reported by period (1.29% pre-COVID, peaking at 3.28% during COVID, settling at 1.85% post-COVID), and STL decomposition of monthly trust-level spend estimates deviations of +33.4% during COVID and +40.5% post-COVID relative to the pre-pandemic trend.

**Issue and fix:** The initial new-supplier indicator classified a disproportionate number of suppliers as "new" at the beginning of each observation window. This reflected left-censoring: every supplier appears new on the first observed day of a truncated dataset, rather than representing a genuine onboarding increase. I addressed this bias by introducing a six-month burn-in window at the beginning of each source's coverage, during which the indicator cannot be activated. This restored a credible pre-COVID baseline and permits a comparable interpretation of the COVID-period increase.

---

## Phase 3: Isolation Forest Modelling & SHAP Explainability (Weeks 6–8)

**Planned:** Train an Isolation Forest on pre-COVID data and score all subsequent periods; use SHAP to explain which features drive each flagged anomaly.

**Completed:** The model is trained on roughly 44,000 pre-COVID trust-spend records across seven features and flags 5,427 of 271,343 scored records (2.00%) as anomalous at the 98th-percentile threshold. The anomaly rate increases monotonically across periods: 0.21% pre-COVID, 1.58% during COVID, and 2.79% post-COVID. This constitutes the project's principal empirical result. SHAP analysis of the 200 highest-confidence anomalies identifies supplier transaction sequence number, representing the transaction's position in the buyer–supplier relationship, as the dominant feature in 179 of 200 cases.

**Issue and fix:** An earlier feature specification inadvertently retained encoded identifiers for data source and spending category. As these are administrative labels rather than behavioural covariates, they permitted the model to partially distinguish observations by source/category rather than by procurement behaviour. SHAP analysis showed that roughly 60% of the top-ranked anomalies were then driven almost entirely by the small dataset from which a record originated, reflecting NHS England's dominance of the training sample rather than substantive spend anomalies. This feature leakage would have reduced the interpretability and construct validity of the anomaly signal. I removed both fields and retrained the model; the within-category amount z-score already retains the relevant within-group information without this bias, and all figures reported above use the corrected specification.

---

## Phase 4: Validation Against Audit Findings (Weeks 9–11)

**Planned:** Cross-reference ML-flagged anomalies against literature-derived audit red flags (NAO, NHS Counter Fraud Authority) and begin developing policy recommendations.

**Completed:** Four audit red-flag rules were derived from the literature: direct awards during COVID, price spikes relative to category norms, unusually large payments to brand-new suppliers during COVID, and suspiciously round transaction amounts. 59.6% of ML-flagged anomalies (3,237 of 5,427) are independently corroborated by at least one rule, and an exact hypergeometric test indicates that this overlap is extraordinarily unlikely under chance assignment (log-probability ≈ −2,472.7).

**Extending beyond the original scope.** In response to your feedback that the proposal required greater methodological rigour and more explicit project-management planning, I have developed several validation and robustness analyses. These additions provide a substantially stronger evidential basis for the dissertation's methods and results chapters:

- **Multi-method comparison.** I trained three alternative unsupervised detectors, namely Local Outlier Factor, One-Class SVM, and an autoencoder, alongside Isolation Forest, calibrated to flag the same 2% rate, to test whether the anomaly signal is specific to my chosen model or a general property of the data. Isolation Forest disagrees sharply with the other three (pairwise Jaccard 0.07–0.09) while they cluster together (0.42–0.67). To adjudicate which is "right," I built a synthetic-anomaly injection benchmark (400 anomalies across three archetypes drawn from NHS Counter Fraud Authority risk indicators, namely invoice inflation, ghost-vendor bursts, and round-number kickbacks) with known ground truth. Isolation Forest clearly outperforms the alternatives on this benchmark (F1 = 0.243 versus 0.023 for the next-best method), which justifies it as the primary detector rather than an arbitrary choice.
- **Formal statistical significance testing.** Beyond the hypergeometric test above, I added bootstrap confidence intervals (2,000 resamples) around every headline rate and Mann-Whitney U tests comparing the full anomaly-score distributions between periods. This moves the project's central claims from descriptive point estimates to statistically corroborated findings.
- **Supplier-buyer network analysis.** I built a bipartite buyer-supplier graph and a same-side supplier co-occurrence graph to test the procurement-fraud literature's hypothesis that anomalies concentrate around structurally central ("hub") suppliers. Counter-intuitively, the data shows the opposite: hub suppliers have a markedly lower per-transaction anomaly rate (1.02%) than peripheral, low-volume suppliers (2.89%, p = 1.26 × 10⁻¹⁸). This reframes the practical implication of the project away from "watch large incumbents for collusion" and toward "prioritise onboarding controls for new, low-volume suppliers"; this finding warrants prominent treatment in the discussion chapter.
- **Composite supplier risk scoring.** I combined anomaly rate, anomaly-score magnitude, and rule-flag rate into a single composite risk score per supplier and assigned each of 3,682 scored suppliers to a Low/Medium/High/Critical risk tier (2,209 / 920 / 368 / 185 respectively). This is a more operationally useful output for an audit team than a raw flag list.
- **Category-level deep dive.** Ranking spending categories by COVID-era growth identified a data-quality issue: "Computer Hardware Purch" initially appeared to have grown by over 47,000%, but inspection showed a small-base effect: its pre-COVID spend was under £1,000, so a modest absolute increase produced an uninformative percentage change. I introduced a minimum transaction-count threshold, after which the leading categories all showed substantial increases in both absolute pounds and percentage terms. The largest were "Med & Surg Equip General" (from roughly £308,000 to £6.64 million) and "CompSwrPrch Additions" (from roughly £124,000 to £2.94 million), both consistent with the PPE/medical-equipment procurement surge documented in the National Audit Office's COVID-19 procurement investigation.
- **Robustness checks.** I re-ran the anomaly-detection procedure at the 95th and 99th percentile thresholds, shifted the COVID-period boundaries by ±1 month, and retrained after removing the new-supplier feature to evaluate whether the headline findings depend on arbitrary specification choices. All three analyses confirm that the pre-COVID < COVID < post-COVID ordering is stable.
- **BI-ready export and interactive dashboard.** I consolidated all outputs into a star-schema export (one fact table plus four dimension tables) and developed a self-contained interactive dashboard, accompanied by a written guide mapping each panel to the fields required for reproduction in Tableau or Power BI. This extends the analysis to non-technical stakeholders rather than limiting access to a notebook-based presentation.

---

## Phase 5: Dissertation Writing (Weeks 12–14, in progress)

The project is now in the dissertation-writing phase. Before proceeding with further drafting, I re-ran the complete pipeline end to end in the primary working environment and reconciled every reported figure against this run. An automated test suite (33 tests covering data cleaning, feature engineering, and each analytical module) supports the reproducibility of every quantitative result reported in the dissertation, rather than reliance on an earlier one-off iteration.

The earlier phases have also been subjected to a critical methodological review focused on three areas likely to attract examiner scrutiny. This review resulted in substantive revisions rather than a purely editorial exercise:

- **Linking computation to inference.** The technical sections on data engineering, modelling, and validation were initially dense and code-oriented. I have revised them so that each stage states both the quantity estimated and the inferential conclusion it supports. For example, the SHAP analysis is identified as the evidential basis for the conclusion that new-supplier relationships drive the anomaly signal, rather than that conclusion being treated as an assumption.
- **Interpreting risk scores for non-technical readers.** The composite supplier risk score (Low/Medium/High/Critical tiers) previously reported only tier counts. I added an interpretation section that maps each tier to a graduated procurement response (for example, Critical suppliers held for manual review before the next payment, versus Low suppliers requiring no departure from standard controls). It also states the joint implication of the SHAP and network results: assurance effort should be weighted toward new-supplier onboarding rather than large established suppliers. This materially strengthens the practical interpretation of the findings for readers outside the technical detail, including a viva panel.
- **Reproducibility and repository hygiene.** I audited the project file organisation to confirm that the retained working materials comprise only code, small summary outputs needed to reproduce a figure or table, and report figures. Large intermediate outputs (millions of transaction-level scores) are fully regenerable from the four raw sources in under two minutes, so they are generated on demand rather than retained as working artefacts. This policy is now explicitly documented.

### A few questions for you before I go further into the write-up

1. Given all four datasets and the full advanced-validation work are now complete, would you like the dissertation to present all four datasets as the primary analysis (as I've now built it), or would you prefer NHS England kept as the primary dataset with the other three positioned as a supplementary robustness/generalisability check?
2. Given the word count, would you like the multi-method comparison, network analysis, and composite risk scoring presented as full Results subsections, or would it be more appropriate to summarise them in the main body and move the detailed tables to an appendix?
3. Is there a particular emphasis you'd like in the Discussion chapter, for example, more on the policy/audit-practice implications versus more on the methodological contribution (the multi-method validation and network-based reframing of the "hub supplier" hypothesis)?

I would welcome an opportunity to discuss these results before finalising the dissertation structure.

---

## Summary of headline results (for quick reference)

| Metric | Pre-COVID | COVID | Post-COVID |
|---|---|---|---|
| Avg. monthly trust spend, % deviation from baseline (STL) | +0.02% | +33.4% | +40.5% |
| New-supplier rate (% of transactions) | 1.29% | 3.28% | 1.85% |
| Isolation Forest anomaly rate | 0.21% | 1.58% | 2.79% |

- ML/rule-based triangulation: 3,237 / 5,427 (59.6%) of ML-flagged anomalies independently corroborated by an audit red flag (hypergeometric log(p) ≈ −2,472.7).
- Method agreement: Isolation Forest vs. alternatives, Jaccard 0.07–0.09; alternatives vs. each other, 0.42–0.67. Synthetic-injection F1: Isolation Forest 0.243 vs. 0.023 for the best alternative.
- Hub vs. non-hub supplier anomaly rate: 1.02% (n=340) vs. 2.89% (n=3,556), p = 1.26 × 10⁻¹⁸.
- Composite risk tiers (3,682 scored suppliers): Low 2,209 / Medium 920 / High 368 / Critical 185.
- Final analytical panel: 326,991 records (271,343 trust-spend transactions, 55,648 contract notices), January 2019 – December 2024.

---

## Key references underpinning the methodology

- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *Proceedings of the IEEE International Conference on Data Mining (ICDM)*, 413–422. https://doi.org/10.1109/ICDM.2008.17
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS, 30*. https://arxiv.org/abs/1705.07874
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics, 6*(1), 3–33.
- National Audit Office. (2020). *Investigation into government procurement during the COVID-19 pandemic*. HC 959. https://www.nao.org.uk/wp-content/uploads/2020/11/Investigation-into-government-procurement-during-the-COVID-19-pandemic.pdf
- NHS Counter Fraud Authority. (2025). *Strategic intelligence assessment 2025*. https://cfa.nhs.uk/resources/downloads/documents/corporate-publications/SIA-2025-OFFICIAL.pdf
- Rhoades, S. A. (1993). The Herfindahl-Hirschman Index. *Federal Reserve Bulletin, 79*(3), 188–189.

Thank you for your continued guidance. The project remains aligned with the proposal timeline, and I would value your advice on the questions above before finalising the dissertation structure.

Best regards,
Gunjan Mahadeva Rao
