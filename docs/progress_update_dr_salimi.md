# Project Progress Update — NHS Procurement Anomaly Detection

**To:** Dr Nahid Salimi
**From:** Gunjan Mahadeva Rao (Student ID: 15848655)
**Course:** MSc Data Science, Coventry University
**Date:** 31 July 2026
**Re:** Progress against the proposal timeline — Phases 1–4 complete, entering Phase 5

Dear Dr Salimi,

Following on from my earlier update about all four datasets becoming available, I wanted to give you a fuller picture of where the project now stands. Since then I have completed the full pipeline set out in my proposal — data engineering, exploratory analysis, Isolation Forest modelling with SHAP explainability, and validation against audit findings — across all four datasets, and I have also completed a substantial body of additional validation work that goes beyond the original scope, largely in response to your feedback on strengthening methodological rigour and project management detail.

This update is structured around the five phases from my proposal's project plan, with a short note on the specific issues encountered in each and how they were resolved.

---

## Phase 1 — Data Engineering Pipeline (Weeks 1–2)

**Planned:** Build a Python pipeline (Pandas, scikit-learn) to clean, standardise, and merge all four datasets into a common schema.

**Completed:** All four datasets have now been fully retrieved, cleaned, and standardised into a shared transaction schema (date, supplier, amount, expense type, expense area, entity, department family), with every record tagged with a `_dataset_source` column for traceability.

| Dataset | Raw records retrieved | Source |
|---|---|---|
| NHS England Payments Over £25k | 271,598 | [england.nhs.uk](https://www.england.nhs.uk/publication/payments-over-25k-reports-2024/) |
| Bradford Teaching Hospitals | 13,925 | [bradfordhospitals.nhs.uk](https://www.bradfordhospitals.nhs.uk/our-trust/reports-and-accounts/) |
| United Lincolnshire Hospitals | 8,280 | [ulh.nhs.uk](https://www.ulh.nhs.uk/publication-of-spend-over-25000/) |
| UK Contracts Finder (NHS-relevant) | 81,482 | [data.open-contracting.org](https://data.open-contracting.org/en/publication/128) |

You'll notice the Bradford, Lincolnshire, and Contracts Finder totals are slightly different from the figures in my earlier email. That's because, in preparing a clean, reproducible build of the pipeline, I ran a full data-completeness audit against the originally published monthly files and found several quiet processing issues that were suppressing legitimate records without raising any error:

- **Report-title rows mistaken for missing headers.** A number of the monthly exports are spreadsheet reports saved to CSV, with a report-title line sitting above the actual column headings (for example, all of Bradford's 2023 and 2024 files, and several Lincolnshire files, begin with a title string rather than a header row). Reading these with the default assumption that row one is the header produces a frame whose columns don't match the target schema at all, so the rows are silently carried through as structurally present but entirely empty — no error is thrown, the data simply vanishes. This turned out to be the single largest source of missing data: every Bradford file from 2023 and 2024 was contributing zero usable rows. I fixed it by scanning the first few rows of each file and picking the first row that actually resolves to known column names, rather than assuming row one is always the header.
- **Inconsistent date formats across monthly files.** Bradford published dates as a plain calendar date until 2021 and as a full timestamp from 2022 onwards; Lincolnshire mixes plain dates, timestamps, and UK-format `dd/mm/yyyy` across different months. The default date parser infers a single format from the first row it sees and silently turns every non-matching value into a blank date, which then gets dropped by the completeness filter — so months in one format were being lost simply because a different month earlier in the file used another format. I replaced this with a parser that handles the mix explicitly, plus a dedicated fallback for Lincolnshire's more inconsistent files.
- **Column names that changed over time.** Lincolnshire renamed its spending-category and date columns several times over the six years (its 2024 files switched to labels the original mapping didn't recognise at all), which meant those months carried blank categories and amounts and were discarded outright. NHS England's final few months added a suffix to its own column headers with the same effect. I rebuilt the column-mapping step as an explicit alias table so any future renaming is easy to add without hunting through code.
- **A mislabelled file format.** One Lincolnshire month was saved with a `.csv` extension but is actually tab-separated internally, which broke the parser partway through the file. I added a step that checks the actual delimiter in each file rather than trusting the extension.
- **A month only published as a macro-enabled spreadsheet.** Bradford's January 2019 return exists only as an `.xlsm` file, with no CSV equivalent, and reading it needed an additional library (`openpyxl`) that hadn't been set up initially. Once added, this recovered 101 transactions that had been missing from the study entirely.
- **The Contracts Finder national export needed sector filtering.** Contracts Finder publishes contract notices for all public-sector buyers, not just health, so isolating the NHS-relevant subset required a keyword filter over buyer names and descriptions (matching terms like "NHS", "hospital", "clinical", "ambulance", "commissioning support", etc.). Refining this filter to properly cover all years recovered the full NHS-relevant set.
- **Two source portals briefly inaccessible.** As mentioned previously, Bradford and Lincolnshire were briefly inaccessible while I resolved download issues on their transparency pages; both are now fully retrievable, and I've kept a local, versioned copy of the original source files (shared via Google Drive) so the pipeline no longer depends on the portals staying available.

Each fix was validated by comparing the rebuilt row counts, column presence, and aggregate monetary totals against the previously held files before accepting the new numbers — the NHS England figures matched exactly as a sanity check (271,598 rows reconciling to the penny on total expenditure), which gave me confidence the corrections to Bradford, Lincolnshire, and Contracts Finder were genuine recoveries rather than new errors.

After cleaning, harmonisation, and merging with the new-supplier and COVID-period logic described below, the four sources combine into a single analytical panel of **326,991 records** spanning January 2019 to December 2024 (271,343 trust-spend transactions and 55,648 contract notices).

---

## Phase 2 — Exploratory Analysis & Time-Series Decomposition (Weeks 3–5)

**Planned:** Visualise spend distributions and supplier concentration using the Herfindahl–Hirschman Index; apply STL decomposition to quantify how far COVID-19 pushed spending away from its pre-pandemic baseline.

**Completed:** Monthly and by-source HHI is computed to characterise supplier concentration; new-supplier onboarding rate is tracked by period (1.29% pre-COVID, peaking at 3.28% during COVID, settling at 1.85% post-COVID); and STL decomposition on monthly trust-level spend shows deviations of +33.4% during COVID and +40.5% post-COVID relative to the pre-pandemic trend.

**Issue and fix:** My first pass at the new-supplier indicator flagged a disproportionate number of "new" suppliers right at the start of each dataset's observation window — an artefact of the fact that every supplier necessarily looks "new" on day one of a truncated dataset, not evidence of a genuine onboarding spike. I resolved this by adding a six-month burn-in window at the start of each source's coverage, during which the flag cannot fire, which brought the pre-COVID baseline rate back down to a realistic level and made the COVID-period spike a genuine, comparable signal.

---

## Phase 3 — Isolation Forest Modelling & SHAP Explainability (Weeks 6–8)

**Planned:** Train an Isolation Forest on pre-COVID data and score all subsequent periods; use SHAP to explain which features drive each flagged anomaly.

**Completed:** The model is trained on roughly 44,000 pre-COVID trust-spend records across seven features and flags 5,427 of 271,343 scored records (2.00%) as anomalous at the 98th-percentile threshold. The anomaly rate rises monotonically across periods — 0.21% pre-COVID, 1.58% during COVID, 2.79% post-COVID — which is the central empirical finding of the project. SHAP analysis of the 200 highest-confidence anomalies shows the supplier's transaction sequence number (i.e., how early in its relationship with a buyer a transaction occurs) is by far the dominant driver, appearing as the top feature in 179 of 200 cases.

**Issue and fix:** An earlier version of the feature set had inadvertently included encoded identifiers for data source and spending category. Because these are administrative labels rather than behavioural signals, their presence let the model partly separate "anomalies" along source/category lines rather than genuine spending behaviour — SHAP analysis showed roughly 60% of the top-ranked anomalies were then driven almost entirely by which (small) dataset a record came from, an artefact of NHS England dominating the training sample, rather than genuine spend anomalies. This is a form of feature leakage that would have made the anomaly signal less meaningful and harder to justify. I removed both fields from the feature set and retrained — the within-category amount z-score feature already captures the useful signal without that bias — and all figures quoted above reflect the corrected feature set.

---

## Phase 4 — Validation Against Audit Findings (Weeks 9–11)

**Planned:** Cross-reference ML-flagged anomalies against literature-derived audit red flags (NAO, NHS Counter Fraud Authority) and begin developing policy recommendations.

**Completed:** Four audit red-flag rules were built from the literature — direct awards during COVID, price spikes relative to category norms, unusually large payments to brand-new suppliers during COVID, and suspiciously round transaction amounts. 59.6% of ML-flagged anomalies (3,237 of 5,427) are independently corroborated by at least one of these rules, and an exact hypergeometric test confirms this overlap is essentially impossible to arise by chance (log-probability ≈ −2,472.7).

**Extending beyond the original scope.** Given your feedback that the proposal needed more methodological rigour and stronger project management thinking, I used the time available in this phase to add several validation and robustness analyses that I believe substantially strengthen the eventual dissertation's methods and results chapters:

- **Multi-method comparison.** I trained three alternative unsupervised detectors — Local Outlier Factor, One-Class SVM, and an autoencoder — alongside Isolation Forest, calibrated to flag the same 2% rate, to test whether the anomaly signal is specific to my chosen model or a general property of the data. Isolation Forest disagrees sharply with the other three (pairwise Jaccard 0.07–0.09) while they cluster together (0.42–0.67). To adjudicate which is "right," I built a synthetic-anomaly injection benchmark (400 anomalies across three archetypes drawn from NHS Counter Fraud Authority risk indicators — invoice inflation, ghost-vendor bursts, and round-number kickbacks) with known ground truth. Isolation Forest clearly outperforms the alternatives on this benchmark (F1 = 0.243 versus 0.023 for the next-best method), which justifies it as the primary detector rather than an arbitrary choice.
- **Formal statistical significance testing.** Beyond the hypergeometric test above, I added bootstrap confidence intervals (2,000 resamples) around every headline rate and Mann-Whitney U tests comparing the full anomaly-score distributions between periods — moving the project's central claims from descriptive point estimates to statistically corroborated findings.
- **Supplier-buyer network analysis.** I built a bipartite buyer-supplier graph and a same-side supplier co-occurrence graph to test the procurement-fraud literature's hypothesis that anomalies concentrate around structurally central ("hub") suppliers. Counter-intuitively, the data shows the opposite: hub suppliers have a markedly lower per-transaction anomaly rate (1.02%) than peripheral, low-volume suppliers (2.89%, p = 1.26 × 10⁻¹⁸). This reframes the practical implication of the project away from "watch large incumbents for collusion" and toward "prioritise onboarding controls for new, low-volume suppliers" — a finding I think is worth foregrounding in the discussion chapter.
- **Composite supplier risk scoring.** I combined anomaly rate, anomaly-score magnitude, and rule-flag rate into a single composite risk score per supplier and assigned each of 3,682 scored suppliers to a Low/Medium/High/Critical risk tier (2,209 / 920 / 368 / 185 respectively) — a more operationally useful output for an audit team than a raw flag list.
- **Category-level deep dive.** Ranking spending categories by COVID-era growth surfaced a data-quality trap worth mentioning: a category called "Computer Hardware Purch" initially appeared to have grown by over 47,000%, but on inspection this was a "small base effect" — its pre-COVID spend was under £1,000, so a trivial absolute increase produced a meaningless percentage swing. I added a minimum transaction-count floor to the ranking, after which the top growers are all categories with genuine substantial increases in absolute pounds as well as percentage terms — the largest being "Med & Surg Equip General" (from roughly £308,000 to £6.64 million) and "CompSwrPrch Additions" (from roughly £124,000 to £2.94 million), both consistent with the PPE/medical-equipment procurement surge documented in the National Audit Office's COVID-19 procurement investigation.
- **Robustness checks.** I re-ran the anomaly detection at the 95th and 99th percentile thresholds, shifted the COVID period boundaries by ±1 month, and retrained with the new-supplier feature removed, to check none of the headline findings are artefacts of an arbitrarily chosen cut-off. All three checks confirm the pre-COVID < COVID < post-COVID ordering is stable.
- **BI-ready export and interactive dashboard.** Finally, I consolidated every output into a clean star-schema export (one fact table plus four dimension tables) and built a self-contained interactive dashboard, along with a short written guide mapping every panel to the exact fields needed to reproduce it in Tableau or Power BI — so the analysis is usable by a non-technical stakeholder, not just readable in a notebook.

---

## Phase 5 — Dissertation Writing (Weeks 12–14, in progress)

I am now moving into the writing phase. Before drafting further, I re-ran the entire pipeline end-to-end in my primary working environment and checked every reported figure against it, backed by an automated test suite (33 tests covering the data cleaning, feature engineering, and each analytical module) so that every number quoted in the dissertation is reproducible from the code and data as submitted, rather than a one-off result from an earlier iteration.

While preparing the write-up I also went back over the earlier phases with a specifically critical eye on three things, since I think they are the areas most likely to attract examiner comment, and I have made concrete changes as a result of that review rather than treating it as a purely editorial pass:

- **Bridging what was run to why it matters.** The technical sections (data engineering, modelling, validation) were originally quite dense and code-oriented in places, so I went back through and made sure every stage explicitly states not just what was computed but what conclusion it licenses and why — for example, spelling out that the SHAP explainability step is what actually proves the "new-supplier relationships drive the anomaly signal" finding, rather than that being an assumption I was making about the results.
- **Making the risk scores interpretable for a non-technical reader.** The composite supplier risk score (Low/Medium/High/Critical tiers) previously stopped at reporting the tier counts. I added an explicit interpretation section connecting each tier to a concrete, graduated procurement action (e.g. Critical suppliers held for manual review before their next payment, versus Low suppliers needing no departure from standard controls), and separately spelled out the practical implication of the SHAP/network findings together — that oversight effort is better weighted toward new-supplier onboarding than toward large, established suppliers, since the data shows the opposite of what that intuition would predict. I think this materially strengthens the "so what" of the results for anyone outside the technical detail, including for a viva panel.
- **Reproducibility and hygiene.** I audited the project's file organisation to confirm that only code, small summary outputs needed to reproduce a figure or table, and the report figures themselves are the files carried forward as the project's working materials — the large intermediate outputs (millions of individual transaction-level scores) are fully regenerable from the four raw sources in under two minutes, so I keep them out of the working set and regenerate them on demand rather than treating them as artefacts to retain. I've documented that policy explicitly rather than leaving it implicit.

### A few questions for you before I go further into the write-up

1. Given all four datasets and the full advanced-validation work are now complete, would you like the dissertation to present all four datasets as the primary analysis (as I've now built it), or would you prefer NHS England kept as the primary dataset with the other three positioned as a supplementary robustness/generalisability check?
2. Given the word count, would you like the multi-method comparison, network analysis, and composite risk scoring presented as full Results subsections, or would it be more appropriate to summarise them in the main body and move the detailed tables to an appendix?
3. Is there a particular emphasis you'd like in the Discussion chapter — e.g., more on the policy/audit-practice implications versus more on the methodological contribution (the multi-method validation and network-based reframing of the "hub supplier" hypothesis)?

I'd welcome the chance to walk through the results with you before I finalise the write-up structure.

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

Thank you again for your continued guidance — I'm on track against the timeline in my proposal and would appreciate your thoughts on the questions above before I lock in the dissertation structure.

Best regards,
Gunjan Mahadeva Rao
