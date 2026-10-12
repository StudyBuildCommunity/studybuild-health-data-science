# Fair, Interpretable, and Actionable Readmission Risk Modeling — Diabetes 130-US Hospitals

## Project Overview

This project analyzes 10 years (1999–2008) of inpatient encounter data from 130 US hospitals to understand and predict 30-day readmissions among diabetic patients. The goal is to build an interpretable baseline classifier, evaluate it honestly on clinically relevant metrics, and assess whether it performs fairly across patient subgroups.

**Clinical Problem:** Hospitals are financially penalized for high readmission rates. A model that is accurate but opaque, or accurate but systematically worse for some patient groups, is insufficient. This project prioritizes interpretability, fairness, and actionable clinical insight.

**Important:** This is an educational risk-stratification exercise, not a deployed clinical decision system. It is not intended for denying or approving care.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

**DOI:** [10.24432/C5230J](https://doi.org/10.24432/C5230J)

**License:** CC BY 4.0

**Citation:** Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, 2014, 781670.

**Dataset Scope:** 101,766 encounters, 50 attributes covering demographics, admission/discharge details, ICD-9 diagnoses, lab/procedure counts, diabetes medications, and prior utilization history. Target variable: `readmitted` (levels: `<30`, `>30`, `NO`).

**Target Definition:** Binary classification — positive class is `<30` (early readmission within 30 days). Original three-class target collapsed to binary, justified by clinical focus on early readmission and class imbalance in the `<30` group (11.16% of raw data).

---

## Variable Description

| Group | Variables | Notes |
|-------|-----------|-------|
| Identifiers | `encounter_id`, `patient_nbr` | Excluded from modeling; `patient_nbr` used for grouped splits |
| Demographics | `race`, `gender`, `age` (10-year brackets), `weight` | `weight` has heavy missingness (>96%) |
| Admission Context | `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `time_in_hospital`, `payer_code`, `medical_specialty` | |
| Utilization & Severity | `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses` | Prior utilization is a strong predictor |
| Diagnoses | `diag_1`, `diag_2`, `diag_3` (ICD-9) | Grouped into 9 categories for modeling |
| Labs & Medications | `max_glu_serum`, `A1Cresult`, 23 diabetes-medication columns, `change`, `diabetesMed` | Diabetes-specific features show weak signal |
| Target | `readmitted` | Collapsed to binary `<30` vs. not |

**Sensitive Attributes:** `race`, `gender`, `age` — deliberately handled and analyzed for fairness.

---

## Data Quality Checks & Cleaning

### Missing Values
| Variable | Missing % | Decision |
|----------|-----------|----------|
| `weight` | 96.86% | **Drop** — exceeds 80% threshold |
| `max_glu_serum` | 94.75% | **Drop** — exceeds 80% threshold |
| `A1Cresult` | 83.28% | **Drop** — exceeds 80% threshold |
| `medical_specialty` | 49.08% | **Hold** — filled with `'Missing'` label |
| `payer_code` | 39.56% | **Drop** — low logical relevance to target |
| `race` | 2.23% | **Hold** — rows removed (small %) |
| `diag_3` | 1.40% | **Hold** — filled with `'Missing'` then grouped |
| `diag_2` | 0.35% | **Hold** — filled with `'Missing'` then grouped |
| `diag_1` | 0.02% | **Hold** — filled with `'Missing'` then grouped |

### Data Cleaning Steps
1. **Drop columns** with >80% missing: `weight`, `max_glu_serum`, `A1Cresult`
2. **Drop `payer_code`** — 39.56% missing, low logical relevance
3. **Save `patient_nbr`** to separate CSV for grouped splitting (prevents leakage)
4. **Drop `encounter_id` and `patient_nbr`** from modeling features
5. **Repeat-encounter handling:** 101,766 records, 71,518 unique patients; 16,773 patients have >1 admission (max 40). All records kept, but patient-grouped splits used.
6. **No completely duplicate records** found
7. **Binary target created:** `<30` = Yes (11.16%), else No
8. **Remove death/hospice records:** `discharge_disposition_id` in [11, 13, 14, 19, 20, 21] — 2,423 records removed (patients cannot be readmitted)
9. **Remove missing race:** 2,273 records removed
10. **Remove invalid gender:** `Unknown/Invalid` — 1 record removed
11. **Final cleaned shape:** 97,108 records × 45 columns

### Categorical Variable Handling
- **`admission_type_id`:** Rare codes [4, 7, 8] → `'Other'`
- **`admission_source_id`:** Rare codes [8, 10, 11, 13, 14, 22, 25] → `'Other'`
- **`discharge_disposition_id`:** Code 18 → `'Unknown'`
- **`age`:** Grouped into `<30`, `[30,60)`, `[60,100)`
- **Diagnosis codes (`diag_1/2/3`):** Grouped into 9 clinical categories (Circulatory, Respiratory, Digestive, Genitourinary, Diabetes, Neoplasms, Musculoskeletal, Injury, Other, Missing)

### Constant Columns Dropped
13 drug columns had >99.5% one value and were dropped:
`acetohexamide`, `troglitazone`, `examide`, `citoglipton`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone`, `glipizide-metformin`, `tolbutamide`, `miglitol`, `tolazamide`, `chlorpropamide`, `acarbose`

### Outliers
Kept — fewer than 0.5% of rows exceed the 99.5th percentile on any variable; these are genuine high-utilization patients. Tree-based models are robust to outliers; capping recommended for linear models.

---

## EDA Findings (Q1–Q3)

### Q1 — What does the readmission population look like?

**Target Distribution (after cleaning):**
- No: 85,982 (88.54%)
- Yes (`<30`): 11,126 (11.46%)

**Demographics:**
| Variable | Distribution |
|----------|-------------|
| Race | Caucasian 76.43%, AfricanAmerican 19.33%, Hispanic 2.08%, Other 1.51%, Asian 0.65% |
| Gender | Female 53.89%, Male 46.11% |
| Age Group | [60,100) 66.82%, [30,60) 30.66%, <30 2.52% |

**Admission Context:**
- Admission type: Emergency (1) 52.83%, Not available (3) 18.86%, Urgent (2) 17.96%
- Admission source: Emergency Room (7) 56.65%, Physician Referral (1) 29.31%
- Discharge: Home (1) 60.52%, SNF (3) 14.02%, Home with home health (6) 13.08%

**Diagnosis Distribution (Primary):**
- Circulatory 29.87%, Other 17.87%, Respiratory 14.03%, Digestive 9.43%, Diabetes 8.72%, Injury 6.90%, Genitourinary 5.06%, Musculoskeletal 4.95%, Neoplasms 3.14%

### Q2 — Which measurements differ most across readmission groups?

**Numeric Variable Comparison (Readmitted vs. Not):**

| Variable | Mean (Yes) | Mean (No) | Mean Diff | Median Diff | Mann-Whitney p |
|----------|------------|-----------|-----------|-------------|----------------|
| `num_lab_procedures` | 44.20 | 42.71 | +1.49 | +1.0 | <0.001 |
| `num_medications` | 16.93 | 15.86 | +1.07 | +1.0 | <0.001 |
| `number_inpatient` | 1.23 | 0.56 | +0.67 | 0.0 | <0.001 |
| `time_in_hospital` | 4.77 | 4.33 | +0.44 | 0.0 | <0.001 |
| `number_diagnoses` | 7.71 | 7.38 | +0.33 | +1.0 | <0.001 |
| `number_emergency` | 0.36 | 0.18 | +0.18 | 0.0 | <0.001 |
| `number_outpatient` | 0.44 | 0.36 | +0.08 | 0.0 | <0.001 |
| `num_procedures` | 1.29 | 1.34 | −0.06 | 0.0 | 0.189 (ns) |

**Key Findings:**
- **Prior inpatient visits** (`number_inpatient`) is the strongest differentiator
- **Emergency visits** (`number_emergency`) also highly significant
- **Length of stay, number of diagnoses, number of medications** show small but significant differences
- **`num_procedures`** is **not** statistically significant (p = 0.189)
- All numeric variables except `num_procedures` show significant differences (Mann-Whitney, p < 0.05)

**Categorical Variable Comparison (Readmission Rate by Group):**
- **Age:** [60,100) has highest rate (11.99%), [30,60) has 10.32%, <30 has 11.25%
- **Race:** Differences are small (9.79%–11.53%), not practically significant
- **Gender:** Female 11.54%, Male 11.37% — negligible difference
- **Diagnosis:** Diabetes group has highest rate (13.18%), Musculoskeletal lowest (9.66%)
- **Admission type:** Emergency (1) has highest rate (11.88%), Other lowest (8.14%)

### Q3 — Which variables appear most strongly associated with 30-day readmission?

**Point-Biserial Correlation (Numeric vs. Binary Target):**

| Variable | Correlation | Absolute | p-value |
|----------|-------------|----------|---------|
| `number_inpatient` | 0.1687 | 0.1687 | <0.001 |
| `number_emergency` | 0.0610 | 0.0610 | <0.001 |
| `number_diagnoses` | 0.0536 | 0.0536 | <0.001 |
| `time_in_hospital` | 0.0471 | 0.0471 | <0.001 |
| `num_medications` | 0.0423 | 0.0423 | <0.001 |
| `num_lab_procedures` | 0.0242 | 0.0242 | <0.001 |
| `number_outpatient` | 0.0191 | 0.0191 | <0.001 |
| `num_procedures` | −0.0103 | 0.0103 | 0.002 |

**Cramér's V (Categorical vs. Binary Target):**

| Variable | Cramér's V |
|----------|------------|
| `discharge_disposition_id` | 0.1143 |
| `medical_specialty` | 0.0772 |
| `diag_3_group` | 0.0349 |
| `diag_1_group` | 0.0279 |
| `diabetesMed` | 0.0261 |
| `diag_2_group` | 0.0254 |
| `age_group` | 0.0236 |
| `admission_source_id` | 0.0222 |
| `change` | 0.0186 |
| `admission_type_id` | 0.0159 |
| `race` | 0.0055 |
| `gender` | 0.0000 |

**Key Associations:**
- **Prior inpatient utilization** is the strongest correlate (r = 0.169)
- **Discharge disposition** is the strongest categorical feature (Cramér's V = 0.114)
- **Emergency visits, number of diagnoses, time in hospital, number of medications** are moderately associated
- **Race and gender** show negligible association with readmission
- **Diabetes-specific features** (`change`, `diabetesMed`) contribute weak signal

**Correlation Matrix:** No severe multicollinearity among numeric predictors (all pairwise correlations < 0.5).

---

## Model Choice and Justification

### Baseline Models (Required)
1. **Logistic Regression** — `class_weight='balanced'`, LIBLINEAR solver. Interpretable coefficients; establishes a floor.
2. **Decision Tree (max_depth=4)** — `class_weight='balanced'`. Shallow enough to draw on a whiteboard; sanity check on LR.

### Extended Models (Team Autonomy)
3. **Random Forest** — ensemble for non-linear interactions.
4. **XGBoost** — gradient boosting; tuned via RandomizedSearchCV for best performance.

**Preprocessing Pipeline:**
- Numeric: Median imputation → StandardScaler
- Categorical: Constant fill (`'Missing'`) → OneHotEncoder (`handle_unknown='ignore'`, `min_frequency=20`)
- Derived features: `total_visits` (sum of outpatient + emergency + inpatient), `inpatient_ratio` (inpatient / total_visits + 1e-5)

**Why Start with Baselines:** If gradient boosting cannot beat logistic regression, the added complexity is not justified. Baselines also reveal whether signal is fundamentally linear or interactive.

---

## Evaluation Metrics

**Primary Metrics:**
- **ROC-AUC** — Ranking ability across thresholds
- **PR-AUC** — Honest picture on imbalanced target
- **Recall (<30)** — Fraction of true early readmissions captured
- **Precision (<30)** — Fraction of flagged cases that are true positives
- **F1-Score (<30)** — Balance of precision and recall
- **Brier Score** — Calibration quality

**Why Not Accuracy Alone:** Accuracy is trivially high when predicting majority class. With ~11.5% positives, a model can achieve 88% accuracy by predicting "no readmission" for everyone — clinically useless.

**Threshold Tuning (Q5):**
- Default 0.50 threshold is a habit from balanced problems; no clinical meaning here.
- `pick_threshold` selects the highest threshold achieving recall ≥ 0.60 on validation.
- LR baseline: threshold 0.47 → recall 0.622, precision 0.168
- Tuned XGBoost: threshold 0.11 → recall 0.636, precision 0.172

**Validation Results (Default 0.50 Threshold):**

| Model | ROC-AUC | PR-AUC | Recall (<30) | Precision (<30) |
|-------|---------|--------|--------------|-----------------|
| LR Baseline | 0.665 | 0.217 | 0.542 | 0.178 |
| Decision Tree (d=4) | 0.652 | 0.196 | 0.633 | 0.166 |
| Random Forest | 0.668 | 0.222 | — | — |
| XGBoost | 0.670 | 0.229 | 0.574 | 0.183 |
| Tuned XGBoost | 0.672 | 0.226 | 0.636 | 0.172 |

**Test Set (Tuned XGBoost, threshold=0.11):**
- ROC-AUC: 0.672, PR-AUC: 0.226
- Recall: 0.636, Precision: 0.172, F1: 0.271
- Brier: 0.095

---

## Error Analysis (Q6)

**False Negative Structure:**
- 398 false negatives in test set
- Median LOS: 3 days; median medications: 13
- FN rate highest in shorter-stay bins (1–2 days: 0.045, 3–4 days: 0.050)
- Diagnosis-group differences exist (Digestive: 0.048, Injury: 0.022) but smaller than LOS effect
- Short, uncomplicated stays are the hardest cases — few labs, few procedures, unremarkable vitals

**Implications:**
- Model is a triage aid, not autonomous decision-maker
- Short-stay patients need post-discharge signal not in dataset
- Aggregate recall masks subgroup variation
- Human review should focus on short, uncomplicated stays

---

## Feature Interpretation (Q7)

### Logistic Regression Coefficients (Top 20)
- **Prior utilization dominates:** `number_inpatient`, `number_emergency`, `total_visits`, `inpatient_ratio`
- **Discharge destination matters:** `discharge_disposition_id` proxies post-discharge support
- **Encounter intensity:** `num_medications`, `num_lab_procedures`, `time_in_hospital`
- **Diabetes-specific features weak:** `change`, `diabetesMed`, `A1Cresult` contribute little

### XGBoost Gain Importance (Top 10)
| Feature | Gain |
|---------|------|
| `number_inpatient` | 64.54 |
| `discharge_disposition_id_1` | 46.56 |
| `inpatient_ratio` | 44.05 |
| `discharge_disposition_id_22` | 36.88 |
| `total_visits` | 25.08 |
| `discharge_disposition_id_3` | 19.83 |
| `discharge_disposition_id_15` | 12.50 |
| `discharge_disposition_id_5` | 12.40 |
| `discharge_disposition_id_2` | 11.96 |
| `diag_1_group_Other` | 10.93 |

**Clinical Meaning:**
- Top features available at discharge — actionable at point of care
- Features are modifiable/addressable (discharge planning, follow-up scheduling)
- Weak diabetes-specific signal is a dataset limitation, not evidence diabetes management doesn't matter

---

## Fairness Findings (Q8)

**Methodology:**
- Recall on `<30` class measured separately across `race`, `gender`, `age_group`
- Wilson 95% confidence intervals for subgroup recall
- Chi-square test on FN/TP contingency table across subgroups
- Two views: held-out test split and out-of-fold predictions (GroupKFold on `patient_nbr`)

**Test Split Results (XGBoost, threshold=0.11):**

| Subgroup | n_pos | Recall | FNR | 95% CI |
|----------|-------|--------|-----|--------|
| Female | 614 | 0.666 | 0.334 | [0.628, 0.702] |
| Male | 478 | 0.596 | 0.404 | [0.552, 0.639] |
| <30 | 36 | 0.750 | 0.250 | [0.589, 0.862] |
| [30,60) | 310 | 0.577 | 0.423 | [0.522, 0.631] |
| [60,100) | 746 | 0.654 | 0.346 | [0.619, 0.687] |

**Key Findings:**
- Recall differences across race (chi²=13.35, p=0.0097), gender (chi²=0.000, p=1.000), age (chi²=17.38, p=0.0002)
- Some differences are large enough to matter for resource allocation
- Small subgroups (e.g., <30 age bracket) have wide confidence intervals — noisy estimates
- Out-of-fold predictions show near-perfect recall (~0.999) — likely optimistic due to repeated patients, but largest unbiased sample

**Implications:**
- Before operational use, re-estimate on cohort with adequate subgroup sizes
- Resolve residual disparity with clinical stakeholders (model adjustment, group-specific thresholds, or human-in-the-loop)

---

## Clinical/Operational Implications (Q9)

**Main Patterns:**
- Prior inpatient utilization, discharge destination, emergency-visit history, and overall utilization burden are strongest correlates of 30-day readmission
- Diabetes-specific medication changes carry weak signal

**Model Performance:**
- LR baseline: ROC-AUC ≈ 0.665, PR-AUC ≈ 0.217 (validation, thr=0.47)
- Tuned XGBoost: ROC-AUC ≈ 0.672, PR-AUC ≈ 0.226 (test, thr=0.11)
- At default 0.50 threshold, would miss ~99.4% of early readmissions
- To catch 60% of `<30` readmissions, threshold must be lowered — real cost in precision
- **This model is a triage aid, not a discharge decision.**

**Fairness:**
- Recall differs across race, gender, and age subgroups
- Differences visible and some large enough to matter if resources allocated by model
- Wilson CIs and chi-square tests reported due to small subgroup sizes

**Limitations:**
- Data from 1999–2008, ICD-9 coded, 130 US hospitals
- No social determinants, post-discharge information, or lab values (only flags)
- `<30` class is small minority — noisy estimates
- No external validation performed
- Outliers kept — genuine high-utilization patients; capping preferable to removal for linear models

**Recommendations:**
- Use as triage aid, not autonomous decision system
- Do not use to deny care or make discharge decisions
- Add post-discharge signal for short-stay patients
- Re-estimate fairness on larger cohort before operational use
- Consider group-specific thresholds or human-in-the-loop review

---

## Required Visualizations

1. **Binary Target Distribution** — Bar chart showing 88.54% No vs. 11.46% Yes
2. **Missing Values by Variable** — Horizontal bar chart with 80% threshold line
3. **Readmission Rate by Age Group** — Bar chart showing [60,100) highest
4. **Readmission Rate by Race** — Horizontal bar chart showing small differences
5. **Readmission Rate by Primary Diagnosis** — Bar chart showing Diabetes highest
6. **Numeric Variable Distributions** — Histograms and boxplots
7. **Correlation Matrix** — Heatmap of numeric variables
8. **Confusion Matrix** — For LR baseline (thr=0.50 and 0.47) and Tuned XGBoost (thr=0.11)
9. **ROC and PR Curves** — For all models
10. **Calibration Curves** — For LR baseline and Tuned XGBoost
11. **Feature Importance** — LR coefficients and XGBoost gain
12. **Fairness Subgroup Recall** — By race, gender, and age group
13. **FN Rate by Length of Stay** — Bar chart
14. **FN Rate by Diagnosis Group** — Bar chart
15. **Model Comparison** — ROC and PR curves for all models on validation

---

## Team Roles and Work Division

| Member | Role | Contributions |
|--------|------|---------------|
| Poone Kamyabi & NIAZ | Data & EDA Lead | Data loading, quality checks, EDA, repeat-encounter handling |
| Iliya Hakani | Modeling Lead | Baseline models, extended models, hyperparameter tuning |
| Iliya Hakani | Evaluation Lead | Metrics, threshold tuning, error analysis, calibration |
| Iliya Hakani & Poone Kamyabi | Fairness & Interpretation Lead | Subgroup analysis, feature importance, clinical interpretation |
| Iliya Hakani & Poone Kamyabi | Documentation & Reproducibility Lead | README, requirements.txt, notebook organization |

*Adjust based on actual team composition and contributions.*

---

## Repository Structure

```
readmission-risk-modeling/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── Cleaning & EDA.ipynb          # Part 1: EDA and preprocessing
│   └── Machine Learning Analysis and Evaluation.ipynb                  # Part 2: Modeling and evaluation
├── src/
│   └── model.py (optional)
├── figures/
├── report/
│   ├── part1&2 - report.docx
│   └── part1&2 - report.pdf
├── data/
│   ├── diabetic_data.csv
│   ├── diabetic_cleaned.csv
│   ├── patient_ids_for_split.csv
│   ├── IDS_mapping.csv
│   └── README.md
└──

```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
scipy
jupyter
```

Install with: `pip install -r requirements.txt`

---

## Instructions for Running the Code

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd readmission-risk-modeling
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Download from [UCI Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
   - Place `diabetic_data.csv` in `./data/`
   - Run the EDA notebook first to generate `diabetic_cleaned.csv` and `patient_ids_for_split.csv`

4. **Run Part 1 — EDA Notebook:**
   ```bash
   jupyter notebook notebooks/Cleaning & EDA.ipynb
   ```
   - Run all cells sequentially
   - Generates cleaned dataset and patient IDs for split
   - Outputs EDA figures

5. **Run Part 2 — Modeling Notebook:**
   ```bash
   jupyter notebook notebooks/Machine Learning Analysis and Evaluation.ipynb
   ```
   - Run all cells sequentially
   - Figures saved to `./figures/`
   - Logs printed to console

6. **Reproducibility:**
   - Fixed random seed: `SEED = 42`
   - All preprocessing fitted on training data only
   - Patient-grouped splits to prevent leakage

---

## Limitations

- **Temporal:** Data from 1999–2008; clinical practice and coding may have changed
- **Geographic:** 130 US hospitals; not representative of all settings
- **Coding:** ICD-9; transition to ICD-10 may affect feature mapping
- **Missing Data:** Weight >96% missing; no lab values (only flags)
- **Class Imbalance:** `<30` is ~11.5% of encounters; estimates noisy for small subgroups
- **No External Validation:** All results on internal splits; no temporal or geographic validation
- **Fairness:** Subgroup differences exist; small sample sizes for some groups limit precision of estimates
- **Causation:** Associations are not causal; feature importance is not clinical causation

---

## Scientific & Quality Requirements

- ✅ Not treated as deployed decision system
- ✅ Not reporting Accuracy alone
- ✅ Confusion matrix and class-wise Recall/F1 included
- ✅ Data leakage prevented (patient-grouped splits)
- ✅ Preprocessing fitted on training data only
- ✅ Model choice justified (baseline + extended)
- ✅ Interpret cautiously (association ≠ causation)
- ✅ Fairness checked, not assumed
- ✅ Subgroup performance reported honestly
- ✅ Limitations documented
- ✅ Reproducible (fixed seed, requirements.txt)
- ✅ Required visualizations included (≥5 figures)
- ✅ Individual submissions from each team member

---
