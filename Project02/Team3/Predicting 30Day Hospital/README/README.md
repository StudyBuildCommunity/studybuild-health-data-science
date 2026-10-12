# Fair, Interpretable, and Actionable Readmission Risk Modeling

**Predicting 30-day hospital readmission among patients with diabetes, with patient-level leakage control and subgroup fairness evaluation**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-informational)
![Domain](https://img.shields.io/badge/Domain-Digital%20Health-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-lightgrey)

> Persian version: [`README_FA.md`](README_FA.md)

## Authors

- Atefe Asadi
- Iman Jamshidi
- Mahdiyeh Mirzaei Valojerdie

## Dataset

*Diabetes 130-US Hospitals for Years 1999–2008*, UCI Machine Learning Repository :
<https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>

The data are licensed CC BY 4.0. The two files used are placed in `Data/`.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Results at a Glance](#2-key-results-at-a-glance)
3. [Objectives](#3-objectives)
4. [Dataset Description](#4-dataset-description)
5. [Repository Structure](#5-repository-structure)
6. [Data Preparation](#6-data-preparation)
7. [Exploratory Data Analysis](#7-exploratory-data-analysis)
8. [Statistical Analysis](#8-statistical-analysis)
9. [Modeling](#9-modeling)
10. [Results](#10-results)
11. [Key Findings](#11-key-findings)
12. [Practical Implications](#12-practical-implications)
13. [Limitations](#13-limitations)
14. [Conclusion](#14-conclusion)
15. [Technologies](#15-technologies)
16. [Reproducibility](#16-reproducibility)
17. [Figure Index](#17-figure-index)

---

## At a Glance

- **What:** an end-to-end machine-learning and statistics project that predicts whether a hospital encounter of a patient with diabetes will be followed by a readmission within 30 days.
- **Data:** 101,766 encounters, 71,518 patients, 50 variables, 130 US hospitals, 1999–2008.
- **How:** documented data cleaning, effect-size-based exploratory analysis, four classifiers, a patient-grouped train/test design, a validation-tuned decision threshold, and a subgroup fairness audit.
- **Main message:** the choice of decision threshold matters more than the choice of algorithm, overall predictive power is modest, and the models fail badly for the youngest age groups.
- **Intended reader:** anyone who wants to see how a clinical risk model can be built, evaluated, and criticised honestly, rather than only how to maximise a score.

## 1. Project Overview

Unplanned hospital readmission is costly for health systems and is a recognised marker of care quality. This project analyses **101,766 hospital encounters** of patients with diabetes from 130 US hospitals and delivery networks (1999–2008) and asks one focused question:

> **Can an encounter that will be followed by a readmission within 30 days be identified at the time of discharge, and is the resulting model reliable across patient subgroups?**

The work has three connected parts:

1. **Data cleaning and quality audit**: duplicate and identifier checks, missing-data decisions, administrative-code mapping, and outlier screening.
2. **Exploratory analysis**: outcome distribution, group differences, effect sizes, and associations with the 30-day readmission outcome.
3. **Predictive modeling with a fairness lens**: four classifiers compared under a patient-grouped train/test design, recall-oriented threshold tuning, hyperparameter search for XGBoost, and subgroup error analysis by race, gender, and age.

The analysis is framed as **decision support**, not automated decision-making. Predictive performance is modest, and subgroup results are unstable where positive cases are scarce.

### 1.1 Why this problem matters

A patient who returns to hospital soon after discharge usually signals that something went wrong: an incomplete recovery, a medication problem, or missing follow-up care. Hospitals and payers therefore track early readmission closely. If a model could flag high-risk patients at discharge, staff could focus follow-up calls, appointments, or home-care referrals on them. The difficulty is that early readmission is relatively rare (about 11% of encounters), so a naive model looks accurate while helping nobody, and a model that works on average can still fail for a particular group of patients. This project addresses both difficulties directly.

### 1.2 Workflow at a glance

```text
Raw data (101,766 encounters)
   │
   ├─ Quality audit: duplicates, missing values, constant columns, outliers
   │
   ├─ Binary target: early readmission (<30 days) = 1, otherwise 0
   │
   ├─ Cohort: remove death/hospice discharges  →  99,343 encounters
   │
   ├─ Patient-grouped split (80/20, zero patient overlap)
   │        ├─ Train  79,395 encounters
   │        └─ Test   19,948 encounters
   │
   ├─ Four models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
   │
   ├─ Threshold tuning on a validation split inside the training data (F2)
   │
   ├─ Evaluation on the test set: Recall, Precision, F1, ROC-AUC, PR-AUC
   │
   └─ Fairness audit by race, gender, and age; interpretation and limitations
```

### 1.3 How to read the metrics

| Metric | Plain-language meaning | Why it matters here |
|---|---|---|
| **Accuracy** | Share of all predictions that are correct | Misleading for rare outcomes: predicting "no early readmission" for everyone would score about 89% |
| **Recall** (sensitivity) | Of the true early readmissions, the share the model catches | The main goal: missing a high-risk patient is the costly error |
| **Precision** | Of the encounters the model flags, the share that really were readmitted early | Tells how many alerts are false alarms; flagging everyone gives a precision equal to the base rate (about 0.113, derived) |
| **F1** | Balance of Precision and Recall | Single summary when both errors matter equally |
| **F2** | Like F1 but weights Recall four times as much as Precision in the harmonic mean | Used to tune the threshold, because a missed readmission is considered worse than a false alarm |
| **Balanced accuracy** | Average of Recall and Specificity | Accuracy corrected for class imbalance |
| **Specificity** | Of the encounters without early readmission, the share correctly left unflagged | Shows how many alerts the model avoids |
| **NPV** | Of the encounters left unflagged, the share truly without early readmission | Tells how reassuring a "low risk" result is |
| **ROC-AUC** | Probability that a random positive encounter receives a higher risk score than a random negative one (0.5 = chance, 1.0 = perfect) | Measures ranking quality independent of any threshold |
| **PR-AUC** | Area under the Precision–Recall curve | More informative than ROC-AUC for rare outcomes; the chance level equals the base rate (about 0.113) |
| **FNR** | Share of true early readmissions the model misses (1 − Recall) | The quantity compared across subgroups in the fairness audit |
| **SMD** | Difference in group means divided by the pooled standard deviation | Effect size independent of sample size |
| **Cramér's V** | Strength of association between a categorical variable and the outcome (0 to 1) | Effect size for categorical variables |

## 2. Key Results at a Glance

All values below are from the held-out, patient-separated test set (19,948 encounters, 2,260 early readmissions).

| Result | Value |
|---|---|
| **Task** | Binary classification: early readmission (`<30` days) = 1, otherwise 0 |
| **Best model** | XGBoost (ROC-AUC 0.683, PR-AUC 0.238) |
| **Decision threshold** | Threshold **0.41** (tuned with F2 on a validation set) raises XGBoost Recall from **0.598** (default 0.50) to **0.797**; ROC-AUC is unchanged |
| **Cost of that threshold** | Precision 0.152 (vs 0.188), Accuracy 0.474 (vs 0.663); about 59% of encounters are flagged |
| **Critical fairness finding** | Severe age-subgroup disparity: FNR gap 0.885–0.967 in all four models; at least one age band has FNR = 1.00 |
| **Strongest predictor** | Prior inpatient visits (standardised mean difference 0.415) |

![XGBoost with vs without threshold](Figures/XGBoost%20-%20With%20vs%20Without%20Threshold.png)

*Figure 13. XGBoost (stage A) at the tuned threshold 0.41 versus the default 0.50. ROC-AUC and PR-AUC do not depend on the threshold.*

## 3. Objectives

1. Characterise the encounter population and the distribution of the readmission outcome.
2. Audit data quality (duplicates, missingness, constant features, implausible or extreme values) and document every cleaning decision.
3. Identify which patient and utilisation variables differ most between early-readmission encounters and all others.
4. Build and compare four classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) without patient-level data leakage.
5. Show why overall Accuracy is inadequate for an imbalanced outcome, and evaluate Recall, Precision, F1, ROC-AUC, and PR-AUC instead.
6. Select the decision threshold on validation data and measure its effect on the test set.
7. Explain which features drive each model.
8. Evaluate subgroup behaviour (race, gender, age) using Recall and false-negative rate (FNR).
9. Translate the results into practical guidance for a hospital administrator or clinician, including conditions under which the model should not be used.

## 4. Dataset Description

| Item | Description |
|---|---|
| **Source** | [UCI Machine Learning Repository, dataset 296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| **Files** | `Data/diabetic_data.csv` (encounter-level data), `Data/IDS_mapping.csv` (labels for admission type, discharge disposition, admission source) |
| **Observations** | 101,766 encounters, each with a unique `encounter_id` |
| **Unique patients** | 71,518 (`patient_nbr`); 54,745 patients have one encounter, 16,773 have more than one; maximum 40 encounters for one patient |
| **Variables** | 50 columns (13 integer, 37 text/categorical) |
| **Target** | `readmitted` has three levels (`NO`, `>30`, `<30`) and is converted to the binary target `early_readmission` (1 = `<30`, 0 = `NO` or `>30`). All modeling, metrics, thresholds, and fairness results use this binary target. |

**Variable groups**

| Group | Examples |
|---|---|
| Identifiers | `encounter_id`, `patient_nbr` |
| Demographics | `race`, `gender`, `age` (10-year bands) |
| Administrative codes | `admission_type_id`, `discharge_disposition_id`, `admission_source_id` |
| Stay and utilisation counts | `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses` |
| Diagnoses | `diag_1`, `diag_2`, `diag_3` (ICD-9 codes) |
| Laboratory results | `max_glu_serum`, `A1Cresult` |
| Medications | 23 medication columns, plus `change` and `diabetesMed` |
| Outcome | `readmitted` |

**Outcome distribution (full dataset, n = 101,766)**

| Original category | Encounters | Share | Binary target |
|---|---:|---:|---|
| `NO` | 54,864 | 53.91% | 0 |
| `>30` | 35,545 | 34.93% | 0 |
| `<30` | 11,357 | 11.16% | **1** |

## 5. Repository Structure

```text
project-04/
├── Dashboard/
│   └── Diabetes_Dashboard.html
├── Data/
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
├── Figures/                      # 17 figures (see Section 17)
├── Notebook/
│   ├── project-04.ipynb
│   └── new_figures.py            # generates Figures 1, 13 and 16
├── article/                      # LaTeX/PDF manuscript
├── README.md                     # English README (this file)
├── README_FA.md                  # Persian README
├── Project_Report.md             # detailed English report
└── Project_Report_FA.md          # detailed Persian report
```

## 6. Data Preparation

Every step is supported by code and executed output in the notebook.

| Step | What was done | Result / decision |
|---|---|---|
| **Loading** | Files read with `keep_default_na=False` so the literal string `None` in `A1Cresult` and `max_glu_serum` stays a valid category; `?` is treated as missing | 101,766 rows × 50 columns |
| **Duplicates** | Checked `encounter_id` and exact duplicate rows | None found; no rows removed |
| **Repeat encounters** | Counted encounters per patient | 16,773 patients with several encounters; retained, and `patient_nbr` is used for grouped splitting |
| **Missing values** | `weight` 96.86%, `medical_specialty` 49.08%, `payer_code` 39.56%, `race` 2.23%, `diag_3` 1.40%, `diag_2` 0.35%, `diag_1` 0.02% | `weight` excluded; the others recoded as an explicit `Missing` category; 3 `Unknown/Invalid` gender values recoded to `Missing` |
| **Constant features** | `examide` and `citoglipton` have one unique value | Excluded |
| **Outliers** | 1.5 × IQR fences used for screening only | Nothing removed: count variables are zero-inflated (for example 83.55% of `number_outpatient` is zero) |
| **Code mapping** | Mapped admission type, admission source, and discharge disposition to readable labels | No unmapped values |
| **Feature engineering** | Binary target; `death_or_hospice` flag; ICD-9 grouping of `diag_1–3` into broad categories | 2,423 death/hospice encounters (2.38%) flagged |
| **Modeling cohort** | Death/hospice encounters removed, because these patients cannot be readmitted | **99,343 encounters**, 11,314 early readmissions (**11.39%**) |
| **Encoding and scaling** | Numeric: median imputation and `StandardScaler`. Categorical: most-frequent imputation and one-hot encoding. Everything sits inside a scikit-learn `Pipeline`, so it is fitted on training data only | 11 numeric and 40 categorical predictors |

**Not performed:** resampling (SMOTE or under-sampling), formal feature selection, log or power transforms, medication-column aggregation, and collinearity filtering.

**Known preparation caveats**

- Nominal ID codes (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`) were also treated as numeric by automatic type detection.
- Raw `diag_1–3` columns (717–790 distinct codes each) remain in the predictors next to the grouped versions, which makes the one-hot matrix very wide.
- `death_or_hospice` is constant after the cohort restriction and carries no information.

## 7. Exploratory Data Analysis

EDA was run on the full 101,766-encounter dataset, before the death/hospice restriction.

### 7.1 Outcome distribution

Early readmission is the minority class at every stage. A model that predicts "no early readmission" for everyone would be right about 89% of the time and be useless, which is why Accuracy is not the headline metric.

![Distribution of binary target](Figures/Distribution%20of%20Binary%20Target%20-%20Early%20Readmission.png)

*Figure 1. Binary target in the full dataset, the modeling cohort, and the test set (11.16% / 11.39% / 11.33% positives).*

![Distribution of readmission outcomes](Figures/Distribution%20of%20Readmission%20Outcomes.png)

*Figure 2. The three original categories (`NO`, `>30`, `<30`). Descriptive only; not the modeling target.*

### 7.2 Age

Encounters concentrate in older adults: the 70–80 band is the largest (26,068), followed by 60–70 (22,483) and 80–90 (17,197).

| Age | Encounters | Early readmissions | Rate |
|---|---:|---:|---:|
| [0–10) | 161 | 3 | 1.86% |
| [10–20) | 691 | 40 | 5.79% |
| [20–30) | 1,657 | 236 | 14.24% |
| [30–40) | 3,775 | 424 | 11.23% |
| [40–50) | 9,685 | 1,027 | 10.60% |
| [50–60) | 17,256 | 1,668 | 9.67% |
| [60–70) | 22,483 | 2,502 | 11.13% |
| [70–80) | 26,068 | 3,069 | 11.77% |
| [80–90) | 17,197 | 2,078 | 12.08% |
| [90–100) | 2,793 | 310 | 11.10% |

Adult rates are broadly similar (about 10–12%). The highest rate (ages 20–30, 14.24%) comes from a small group and should not be read as an age effect.

![Age distribution](Figures/Age%20Distribution%20of%20Hospital%20Encounters.png)

*Figure 3. Encounters per 10-year age band. The youngest bands are very small, which matters for the fairness analysis.*

![Early readmission rate by age](Figures/Early%20Readmission%20Rate%20by%20Age%20Group.png)

*Figure 4. Early-readmission rate per age band.*

### 7.3 Utilisation differences between groups

| Variable (mean) | `<30` | `>30` | `NO` |
|---|---:|---:|---:|
| `number_inpatient` | **1.22** | 0.84 | 0.38 |
| `number_emergency` | 0.36 | 0.28 | 0.11 |
| `time_in_hospital` (days) | 4.77 | 4.50 | 4.25 |
| `num_medications` | 16.90 | 16.28 | 15.67 |
| `num_lab_procedures` | 44.23 | 43.84 | 42.38 |

The clearest separation is in prior inpatient visits. Length of stay differs only slightly and the distributions overlap heavily.

![Prior inpatient visits](Figures/Prior%20Inpatient%20Visits%20by%20Readmission%20Group.png)

*Figure 5. Prior inpatient visits by readmission group.*

![Time in hospital](Figures/Time%20in%20Hospital%20by%20Readmission%20Group.png)

*Figure 6. Time in hospital by readmission group.*

### 7.4 Effect sizes

| Variable | Absolute SMD |
|---|---:|
| `number_inpatient` | **0.415** |
| `number_diagnoses` | 0.163 |
| `number_emergency` | 0.157 |
| `time_in_hospital` | 0.140 |
| `num_medications` | 0.122 |
| `num_lab_procedures` | 0.065 |
| `number_outpatient` | 0.059 |

![Standardized differences](Figures/Standardized%20Differences%20-%20Early%20Readmission%20vs%20Other%20Encounters.png)

*Figure 7. Absolute standardised mean differences, early readmission versus all other encounters.*

### 7.5 Correlations among numeric variables

Pairwise Pearson correlations are weak to moderate. The largest is 0.47 (`time_in_hospital` and `num_medications`); `time_in_hospital`–`num_lab_procedures` is 0.32 and `number_inpatient`–`number_emergency` is 0.27.

![Correlation heatmap](Figures/Correlation%20Between%20Numeric%20Utilization%20Variables.png)

*Figure 8. Correlation between numeric utilisation variables.*

### 7.6 Categorical associations (Cramér's V with early readmission)

| Variable | Cramér's V |
|---|---:|
| `discharge_disposition` | **0.125** |
| `insulin` | 0.043 |
| `age` | 0.034 |
| `diabetesMed` | 0.027 |
| `admission_source` | 0.023 |
| `change`, `A1Cresult` | 0.019 each |
| `admission_type`, `race` | 0.016 each |
| `max_glu_serum` | 0.012 |
| `gender` | 0.003 |

Diagnosis-group variables are weakly associated with the outcome (V = 0.031, 0.023, and 0.032 for the primary, secondary, and tertiary diagnosis group). `discharge_disposition` has the largest association, but it mixes clinical destination with administrative categories, so it should not be read as a simple clinical risk effect.

### 7.7 Race, gender, and diagnosis group

| Race | Encounters | Early-readmission rate |
|---|---:|---:|
| Caucasian | 76,099 | 11.29% |
| African American | 19,210 | 11.22% |
| Hispanic | 2,037 | 10.41% |
| Asian | 641 | 10.14% |
| Other | 1,506 | 9.63% |
| Missing | 2,273 | 8.27% |

Gender: Female 11.25% (n = 54,708), Male 11.06% (n = 47,055). Among primary diagnosis groups, Diabetes has the highest rate among sizeable groups (12.98%, n = 8,757), followed by Injury (12.25%); Respiratory (9.73%) and Musculoskeletal (9.50%) are lowest. The `Missing` group (23.81%) has only 21 encounters and is not interpretable.

![Rate by race](Figures/Early%20Readmission%20Rate%20by%20Race.png)

*Figure 9. Early-readmission rate by race. Small groups are less stable.*

![Rate by gender](Figures/Early%20Readmission%20Rate%20by%20Gender.png)

*Figure 10. Early-readmission rate by gender.*

![Rate by diagnosis group](Figures/Early%20Readmission%20Rate%20by%20Primary%20Diagnosis%20Group.png)

*Figure 11. Early-readmission rate by primary diagnosis group.*

### 7.8 Anomalies

- Extreme utilisation counts exist (up to 76 emergency and 42 outpatient visits for one encounter). They were retained because they may represent genuine high-utilisation patients.
- 3,691 encounters (3.63%) have a `Missing` discharge disposition, and 5,291 admission types are coded `NULL`.

## 8. Statistical Analysis

The notebook uses **effect-size-based descriptive statistics**. No hypothesis tests or p-values were run, and none are reported here.

| Method | Purpose | Why selected |
|---|---|---|
| Standardised mean difference (SMD) | Compare numeric variables between `<30` and all other encounters | Unit-free effect size, independent of sample size |
| Cramér's V | Association between a categorical variable and the binary outcome | Handles variables with different numbers of levels; bounded between 0 and 1 |
| Pearson correlation | Relationships among numeric variables | Simple screening of linear association and redundancy |
| 1.5 × IQR rule | Screen extreme values | A screening device, not a deletion rule |
| Group rates with counts | Compare readmission percentages across groups | Transparent; group sizes are shown to avoid over-reading small groups |

**Standardised mean difference** (group 1 = early readmission, group 0 = all others):

$$
\mathrm{SMD} = \frac{\bar{x}_1 - \bar{x}_0}{\sqrt{\tfrac{1}{2}\left(s_1^2 + s_0^2\right)}}
$$

**Cramér's V** (Pearson chi-square statistic of the contingency table, with $n$ observations, $r$ rows, and $k$ columns):

$$
V = \sqrt{\frac{\chi^2 / n}{\min(k-1,\; r-1)}}
$$

**Interpretation**

- With roughly 100,000 encounters, even trivial differences produce very small p-values; effect sizes show whether a difference is large enough to matter.
- `number_inpatient` (SMD = 0.415) is a small-to-moderate difference by common rules of thumb, and still leaves large overlap between groups.
- All Cramér's V values are small (largest 0.125).
- The data are observational, so association is not causation.

## 9. Modeling

### 9.1 Problem setup

| Item | Choice |
|---|---|
| **Problem type** | Binary classification: `early_readmission` = 1 if `readmitted == "<30"`, else 0 |
| **Cohort** | 99,343 non-death/hospice encounters; positive rate 11.39% |
| **Predictors** | 51 columns (11 numeric, 40 categorical); no separate feature-selection step |
| **Train/test split** | `GroupShuffleSplit` by `patient_nbr`, 80/20, `random_state=123`. Train: 79,395 encounters / 55,992 patients. Test: 19,948 encounters / 13,998 patients. **Patient overlap: 0.** Test set: 2,260 positives (11.33%) and 17,688 negatives |
| **Imbalance handling** | `class_weight="balanced"` (Logistic Regression, Decision Tree), `"balanced_subsample"` (Random Forest), `scale_pos_weight` = negatives / positives in training (XGBoost) |
| **Preprocessing** | Median / most-frequent imputation, standardisation, and one-hot encoding inside a `Pipeline` fitted on training data only |

### 9.2 Leakage control

Because 16,773 patients contribute several encounters, a random row-level split would put the same patient in both training and test data. The project controls this at three levels:

1. The train/test split is grouped by `patient_nbr` (zero patient overlap).
2. Threshold selection uses a second grouped 80/20 split inside the training data; the test set is never used to choose a threshold.
3. Hyperparameter search uses patient-grouped cross-validation (`StratifiedGroupKFold`, 3 folds).

### 9.3 Models and settings (stage A comparison)

| Model | Key settings |
|---|---|
| Logistic Regression | `max_iter=2000`, balanced class weights |
| Decision Tree | `max_depth=8`, `min_samples_leaf=50`, balanced class weights |
| Random Forest | 250 trees, `max_depth=16`, `min_samples_leaf=10`, balanced-subsample weights |
| XGBoost | 250 trees, `max_depth=5`, `learning_rate=0.05`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `scale_pos_weight` as above |

### 9.4 Threshold selection

A probabilistic classifier outputs a risk score; the decision threshold decides at which score an encounter is flagged. With an 11% outcome, the default cut-off of 0.50 is a poor operating point when a missed early readmission is costlier than a false alarm. For each model, the threshold in [0.05, 0.50] (step 0.01) that maximises **F2** on a patient-separated validation set is selected, and the model is then refit on the full training set.

$$
F_\beta = (1+\beta^2)\,\frac{\text{Precision}\cdot\text{Recall}}{\beta^2\,\text{Precision}+\text{Recall}},\qquad \beta = 2
$$

$$
\mathrm{FNR} = \frac{FN}{FN + TP} = 1 - \text{Recall}
$$

### 9.5 Hyperparameter search (stage B, supplementary)

| Item | Value |
|---|---|
| Method | `RandomizedSearchCV`, 12 sampled combinations, `StratifiedGroupKFold` with 3 folds (grouped by patient), scoring = F2, `refit=True` |
| Search space | `n_estimators` {150, 200, 250}; `max_depth` {3, 4, 5}; `learning_rate` {0.03, 0.05, 0.08}; `min_child_weight` {3, 5, 8}; `subsample` {0.8, 1.0}; `colsample_bytree` {0.8, 1.0}; `reg_lambda` {1.0, 3.0, 5.0} |
| Best parameters | `n_estimators=200`, `max_depth=3`, `learning_rate=0.05`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=1.0`, `reg_lambda=1.0` |
| Best cross-validated F2 | **0.406** |

Stage B is evaluated at a fixed threshold of 0.50 and is supplementary. The model that carries the threshold result is the stage A XGBoost at 0.41.

### 9.6 Why these four models

The four models were chosen to cover a range of complexity and interpretability:

| Model | Role in the comparison | Strength | Weakness for this task |
|---|---|---|---|
| Logistic Regression | Transparent linear baseline | Simple, coefficients can be read directly | Cannot capture interactions; very wide one-hot input makes coefficients noisy |
| Decision Tree | Interpretable non-linear baseline | Easy to explain, shows clear splits | Unstable and prone to overfitting even when depth is limited |
| Random Forest | Averaged ensemble of trees | Robust to noise, captures interactions | Harder to interpret; highest alert volume at its tuned threshold |
| XGBoost | Gradient-boosted trees | Best ranking quality (ROC-AUC, PR-AUC) here | Needs tuning; importances are not causal |

Because the differences in ROC-AUC are small (0.659–0.683), the project does not claim a decisive winner on discrimination alone. XGBoost is preferred because it is best on ROC-AUC, PR-AUC, balanced accuracy, and F1 together, while the more important design choice is the validated decision threshold.

## 10. Results

### 10.1 Model comparison at the default threshold (0.50)

| Model | Accuracy | Balanced acc. | Recall (`<30`) | Precision | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.655 | 0.617 | 0.568 | 0.178 | 0.272 | 0.664 | 0.221 |
| Decision Tree | 0.590 | 0.620 | 0.659 | 0.167 | 0.267 | 0.659 | 0.212 |
| Random Forest | 0.643 | 0.626 | 0.603 | 0.180 | 0.277 | 0.674 | 0.207 |
| XGBoost | 0.663 | 0.634 | 0.598 | 0.188 | 0.287 | 0.683 | 0.238 |

### 10.2 Model comparison at recall-oriented thresholds (tuned on validation with F2)

| Model | Threshold | Accuracy | Balanced acc. | Recall (`<30`) | Precision | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.39 | 0.448 | 0.597 | 0.791 | 0.145 | 0.245 | 0.664 | 0.221 |
| Decision Tree | 0.38 | 0.444 | 0.593 | 0.786 | 0.143 | 0.243 | 0.659 | 0.212 |
| Random Forest | 0.45 | 0.426 | 0.602 | **0.830** | 0.145 | 0.247 | 0.674 | 0.207 |
| XGBoost | 0.41 | 0.474 | **0.615** | 0.797 | **0.152** | **0.255** | **0.683** | **0.238** |

XGBoost has the best ROC-AUC, PR-AUC, balanced accuracy, and F1; Random Forest has the highest Recall at its tuned threshold. ROC-AUC differences across models are small (0.659–0.683).

### 10.3 Error counts at tuned thresholds (test set)

| Model | TP | FN | FP | TN | FNR | Specificity* | NPV* | Share flagged* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 1,787 | 473 | 10,539 | 7,149 | 0.209 | 0.404 | 0.938 | 61.8% |
| Decision Tree | 1,776 | 484 | 10,608 | 7,080 | 0.214 | 0.400 | 0.936 | 62.1% |
| Random Forest | 1,876 | 384 | 11,069 | 6,619 | 0.170 | 0.374 | 0.945 | 64.9% |
| XGBoost | 1,802 | 458 | 10,044 | 7,644 | 0.203 | 0.432 | 0.943 | 59.4% |

\* Derived from the confusion-matrix counts (specificity = TN/(TN+FP), NPV = TN/(TN+FN), share flagged = (TP+FP)/19,948).

![Confusion matrices](Figures/Confusion%20Matrices%20-%20All%20Models.png)

*Figure 12. Confusion matrices for all four models at their tuned thresholds.*

Raising Recall to roughly 0.79–0.83 costs a large drop in Accuracy (to 0.43–0.47) and Precision (to about 0.14–0.15). Ranked by Accuracy alone, the models would appear in the opposite order to their usefulness for catching early readmissions.

### 10.4 The decision threshold (XGBoost, stage A)

| Metric | Threshold 0.41 | Default 0.50 | Change |
|---|---:|---:|---:|
| Recall (`<30`) | **0.797** | 0.598 | +0.199 (+33% relative) |
| Precision (`<30`) | 0.152 | 0.188 | −0.036 |
| F1 (`<30`) | 0.255 | 0.287 | −0.032 |
| Balanced accuracy | 0.615 | 0.634 | −0.019 |
| Accuracy | 0.474 | 0.663 | −0.189 |
| ROC-AUC | 0.683 | 0.683 | 0 (threshold-independent) |
| PR-AUC | 0.238 | 0.238 | 0 (threshold-independent) |
| F2 (derived) | 0.431 | 0.416 | +0.015 |
| True early readmissions caught (of 2,260) | **1,802** | ≈ 1,351 (derived) | ≈ +451 |
| Encounters flagged (of 19,948) | 11,846 (59.4%) | ≈ 7,190 (≈ 36%) (derived) | ≈ +4,650 |

Moving from 0.50 to 0.41 does not change how well the model ranks patients (ROC-AUC and PR-AUC are identical); it changes where the alarm is set. The lower threshold recovers about 450 additional true early readmissions at the price of about ten additional alerts per additional case caught (derived). The right threshold depends on review capacity: a hospital that can review fewer alerts should choose a higher one.

### 10.5 Feature importance

| Model | Strongest features (as reported) |
|---|---|
| Decision Tree | `number_inpatient` (0.425), discharge disposition "Home" (0.135), discharge disposition "Rehabilitation" (0.055) |
| Random Forest | `number_inpatient` (0.132), `discharge_disposition_id` (0.054), discharge disposition "Home" (0.051) |
| XGBoost | discharge disposition "Home" (0.020), `number_inpatient` (0.019), discharge disposition "Rehabilitation" (0.009) |
| Logistic Regression (absolute coefficient) | Rare individual diagnosis codes (for example `diag_2_136`, `diag_3_537`) |

![Top features](Figures/Top%20Features%20-%20All%20Models.png)

*Figure 14. Top-5 importances (absolute coefficients for Logistic Regression) per model.*

Across the three tree-based models, prior inpatient visits and discharge disposition dominate. The top Logistic Regression coefficients are rare diagnosis codes, which is more consistent with noise from a very wide one-hot matrix than with a stable clinical signal. Importances describe fitted models, not causal effects.

### 10.6 Fairness evaluation (stage A models, tuned thresholds)

Recall and FNR were computed within race, gender, and age subgroups on the test set and summarised as the gap between the best and worst subgroup.

| Model | Race: Recall range | Race: FNR gap | **Age: FNR gap** | Gender: FNR gap |
|---|---|---:|---:|---:|
| Logistic Regression | 0.641–0.929 | 0.288 | **0.910** | 0.006 |
| Decision Tree | 0.714–0.798 | 0.083 | **0.933** | 0.031 |
| Random Forest | 0.718–0.836 | 0.118 | **0.967** | 0.024 |
| XGBoost | 0.692–0.805 | 0.112 | **0.885** | 0.029 |

![FNR gap across subgroups](Figures/FNR%20Gap%20Across%20Patient%20Subgroups.png)

*Figure 15. FNR gap (maximum minus minimum) within race, gender, and age for all four models.*

> **Critical finding: severe age-related disparity.** In all four models, at least one age band has FNR = 1.00: not a single true early readmission in that band is detected. The age FNR gap (0.885–0.967) is about 3–11 times the race gaps and 30–150 times the gender gaps. The model does not perform equally across age, and this is the most serious fairness problem in the project.

The zero-recall bands are the youngest, where positive cases are extremely scarce:

| Age band | Test encounters | True early readmissions |
|---|---:|---:|
| [0–10) | 27 | **1** |
| [10–20) | 119 | **6** |
| [20–30) | 396 | 78 |
| [30–40) | 743 | 83 |
| [40–50) | 1,837 | 172 |
| [50–60) | 3,387 | 298 |
| [60–70) | 4,504 | 513 |
| [70–80) | 5,127 | 611 |
| [80–90) | 3,281 | 438 |
| [90–100) | 527 | 60 |
| **Total** | **19,948** | **2,260** |

![Age subgroup disparity](Figures/Age%20Subgroup%20Disparity%20and%20Support.png)

*Figure 16. Left: true early readmissions per age band (red = fewer than 10). Right: FNR gap by attribute (XGBoost, stage A).*

With only 1 and 6 positives in the two youngest bands, an FNR of 1.00 can arise from a single missed case, so the exact magnitude of the gap is uncertain and subgroup confidence intervals were not computed. This qualifies the size of the estimate; it does not remove the finding. The model gives no evidence that it can detect early readmissions in patients under 20.

Race gaps are moderate (largest for Logistic Regression, 0.288) and gender gaps are small (0.006–0.031). The `min_Recall = 0.000` for gender comes from a `Missing` gender group of one encounter with no positives and is not a model failure.

### 10.7 Supplementary tuned XGBoost (stage B, threshold 0.50)

| Metric | Value |
|---|---:|
| Accuracy | 0.640 |
| Recall (`<30`) | 0.619 |
| Precision (`<30`) | 0.181 |
| F1 (`<30`) | 0.280 |
| ROC-AUC | ≈ 0.68 (read from the ROC figure) |
| Precision / Recall for `Not <30` | 0.930 / 0.643 |
| Macro-average precision / recall / F1 | 0.555 / 0.631 / 0.520 |

![ROC curve](Figures/ROC%20Curve%20-%20XGBoost.png)

*Figure 17. ROC curve of the hyperparameter-searched XGBoost (stage B) on the test set; AUC ≈ 0.68.*

The threshold was not re-tuned for this model, so its operating point (higher precision, lower recall) differs from stage A at 0.41. PR-AUC and fairness metrics were not recomputed for stage B.

### 10.8 Practical meaning of the performance

- **Modest discrimination.** ROC-AUC of about 0.68 indicates some, but limited, ability to rank encounters by risk.
- **Lift over base rate (derived).** Test prevalence is 11.33%. Precision of 0.152 (threshold 0.41) is about 1.3 times the base rate; 0.188 (threshold 0.50) is about 1.7 times. The threshold buys Recall, not Precision.
- **Alert volume (derived).** At threshold 0.41, 11,846 of 19,948 test encounters (59.4%) are flagged to catch 79.7% of true early readmissions. The performance is real but operationally costly.

### 10.9 Putting the results together

1. **Ranking quality is similar across models.** ROC-AUC between 0.659 and 0.683 means every model separates high-risk from low-risk encounters only moderately well. Choosing a different algorithm changes little; the information in the available variables is the limiting factor.
2. **The operating point is a policy decision.** Tuning the threshold moved XGBoost from catching 59.8% to 79.7% of early readmissions without changing its ranking ability. Where to set it depends on how many alerts a hospital can review and how costly a missed readmission is compared with an unnecessary review.
3. **High Recall is expensive.** At the tuned thresholds roughly 59–65% of all encounters are flagged and about 85% of the alerts are false alarms (1 minus Precision, derived). The model is a prioritisation aid, not a diagnosis.
4. **The signal is concentrated in utilisation history.** Prior inpatient visits dominate both the exploratory analysis and the tree-based models; demographic variables add little.
5. **Fairness problems can hide behind good averages.** Overall metrics look acceptable while one group, the youngest patients, receives no correct early-readmission alerts at all. This is why subgroup evaluation is part of the project and not an afterthought.

## 11. Key Findings

1. **The task is binary and imbalanced:** 11.16% of the 101,766 encounters were followed by readmission within 30 days (11.39% in the modeling cohort; 11.33% in the test set).
2. **The decision threshold is the model's key strength.** XGBoost at 0.41 (tuned on validation data) reaches Recall 0.797 versus 0.598 at 0.50, with identical ROC-AUC (0.683) and PR-AUC (0.238), at the price of Precision 0.152 and Accuracy 0.474.
3. **Severe age-subgroup disparity** (FNR gap 0.885–0.967, FNR = 1.00 in at least one band) in all four models; the affected bands hold only 1 and 6 positives, so the exact size is uncertain.
4. **Prior inpatient utilisation is the clearest signal** (SMD 0.415; mean 1.22 vs 0.38 prior visits), and the top or second feature in all tree-based models.
5. **Most other associations are weak:** all Cramér's V values are ≤ 0.125.
6. **Discharge disposition is informative but hard to interpret** because it mixes clinical destination with administrative categories.
7. **Accuracy is misleading:** raising Recall to about 0.80 reduces Accuracy to about 0.43–0.47.
8. **Models are close in discrimination** (ROC-AUC 0.659–0.683); XGBoost is best on ROC-AUC and PR-AUC.
9. **Race and gender gaps are moderate and small** (FNR gap 0.083–0.288 and 0.006–0.031).
10. **Patient-level leakage was controlled:** no patient appears in both training and test sets, and the threshold was chosen without the test set.

## 12. Practical Implications

These implications come from associations; the design does not support causal claims.

- **Use as triage, not automation.** With Precision of roughly 15–18%, most alerts are false alarms. The model suits prioritising a manual review queue, not triggering automatic interventions.
- **Choose the threshold to match capacity.** At 0.41 the model flags about 59% of discharges and catches 79.7% of early readmissions; at 0.50 it flags roughly 36% (derived) and catches 59.8%.
- **Prior hospitalisation history is an actionable flag.** Follow-up resources could be prioritised for patients with several previous inpatient visits; the data show association, not that such actions would reduce readmission.
- **Interpret discharge destination carefully** because of the death/hospice and administrative categories.
- **Do not deploy without addressing age.** Exclude or separately validate patients under 20, report age-band Recall with confidence intervals, and consider age-aware calibration or thresholds before any pilot.
- **Revalidate before use.** The data end in 2008; any real use needs validation on current data.

### Intended use and disclaimer

This repository is an educational and portfolio project built on a public, de-identified dataset. It is not a medical device, has not been validated on current clinical data, and must not be used to make decisions about real patients. Any clinical deployment would require validation on recent local data, prospective testing, fairness review, and clinical governance.

## 13. Limitations

- **Temporal validity:** data are from 1999–2008; clinical practice, coding, and care pathways have changed.
- **Single split, no uncertainty estimates:** one grouped train/test split; no bootstrap confidence intervals or repeated splits.
- **Age-subgroup disparity (critical):** see Section 10.6.
- **Small subgroups:** several age and race groups have very few positives, giving unstable Recall/FNR.
- **Scoring convention:** groups with zero positives are scored Recall = 0 (`zero_division=0`), which inflates apparent gaps.
- **Fairness scope:** only stage A models, only Recall/FNR, and only race, gender, and age; no calibration or predictive-parity measures.
- **Stage B threshold:** not re-tuned; PR-AUC and subgroup metrics not recomputed.
- **Threshold uncertainty:** chosen on one validation split; its stability was not assessed.
- **Feature handling:** nominal ID codes treated as numeric, high-cardinality diagnosis codes one-hot encoded, and a constant flag left in the predictors.
- **Missing information:** `weight` unusable (96.86% missing); `payer_code` and `medical_specialty` handled as a `Missing` category.
- **Outcome definition:** readmissions outside the source system are not captured.
- **Representation:** predominantly Caucasian (74.78%) and African American (18.88%); other groups are small.
- **Association, not causation.**
- **Modest performance:** ROC-AUC about 0.68 and precision below 20% limit direct clinical usefulness.

## 14. Conclusion

This project builds a complete, leakage-aware workflow for 30-day readmission among hospitalised patients with diabetes: documented data cleaning, effect-size-based EDA, a four-model comparison, validation-based threshold selection, patient-grouped hyperparameter search, and subgroup error analysis. Three results matter most. First, the decision threshold is the model's key strength: XGBoost at 0.41 detects 79.7% of early readmissions versus 59.8% at the default 0.50, with unchanged ROC-AUC, at the cost of precision and a high alert volume. Second, all four models show a severe age-subgroup disparity that must be resolved or controlled before use, even though the affected bands hold very few positives. Third, prior inpatient utilisation and discharge disposition are the most consistent predictive signals, while overall discrimination remains modest (ROC-AUC about 0.68). The appropriate use is as a human-reviewed triage aid, after revalidation on current data, with a threshold chosen for review capacity and with age-specific safeguards.

## 15. Technologies

| Category | Tools |
|---|---|
| Language | Python (kernel reported as 3.13.5) |
| Data handling | pandas, NumPy |
| Statistics | SciPy (`chi2_contingency`) |
| Machine learning | scikit-learn (`Pipeline`, `ColumnTransformer`, `GroupShuffleSplit`, `StratifiedGroupKFold`, `RandomizedSearchCV`, metrics), XGBoost |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebook, Anaconda |

## 16. Reproducibility

**Requirements:** Python 3.10 or later and Jupyter Notebook or JupyterLab. Package versions were not recorded in the project; pin them once you freeze your environment.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn jupyter
pip freeze > requirements.txt      # optional: record exact versions
```

**Run the analysis**

```bash
cd Notebook
jupyter lab project-04.ipynb
```

Run all cells from top to bottom; figures are written to `../Figures/`. Then run `python new_figures.py` from the same folder to create Figures 1, 13, and 16.

**Notes**

- The notebook reads data with a Windows-style path, `Path(r"..\Data")`. On macOS or Linux change it to `Path("../Data")`.
- Run the notebook from inside `Notebook/` so relative paths resolve, and make sure `Figures/` exists.
- The random seed is `123` for the splits, models, and hyperparameter search; small differences can still occur across library versions.
- Expected checkpoints: 101,766 × 50 raw data; modeling cohort of 99,343 encounters; train 79,395 / test 19,948; zero patient overlap.

### Quick start

1. Download or clone this repository and open a terminal in the project folder.
2. Create and activate a virtual environment and install the packages listed above.
3. Place `diabetic_data.csv` and `IDS_mapping.csv` in `Data/` (the files are already there if you cloned the full project; otherwise download them from the UCI link at the top).
4. Open `Notebook/project-04.ipynb` and run all cells from top to bottom.
5. Run `python new_figures.py` from inside `Notebook/` to regenerate Figures 1, 13, and 16.
6. Open `Dashboard/Diabetes_Dashboard.html` in a browser for the interactive view.

### Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError` for the data files | The notebook uses a Windows-style path or is run from the wrong folder | Run from inside `Notebook/` and change the path to `Path("../Data")` on macOS/Linux |
| `ModuleNotFoundError: xgboost` | Package not installed in the active environment | `pip install xgboost` |
| Figures are not saved | `Figures/` folder is missing | Create the folder before running |
| Slightly different numbers | Library versions differ | Record versions with `pip freeze`; the seed is fixed at 123 |

### Frequently asked questions

**Why a binary target instead of three classes?** The question of interest is early readmission specifically, so `>30` and `NO` are merged into the negative class. The three-class distribution is shown only for transparency (Figure 2).

**Why is Accuracy so low at the tuned thresholds?** Because the threshold is deliberately set to catch more true cases, which flags many more encounters. Accuracy rewards the opposite behaviour, so it is not used to judge the model.

**Why was resampling such as SMOTE not used?** Class imbalance was handled with class weights and threshold tuning. Resampling was not performed in this project.

**Can the model be used in practice?** Not as it stands. See the intended-use note and the age-subgroup finding in Section 10.6.

## 17. Figure Index

| Fig. | Title | File |
|---:|---|---|
| 1 | Distribution of Binary Target - Early Readmission | `Figures/Distribution of Binary Target - Early Readmission.png` |
| 2 | Distribution of Readmission Outcomes | `Figures/Distribution of Readmission Outcomes.png` |
| 3 | Age Distribution of Hospital Encounters | `Figures/Age Distribution of Hospital Encounters.png` |
| 4 | Early Readmission Rate by Age Group | `Figures/Early Readmission Rate by Age Group.png` |
| 5 | Prior Inpatient Visits by Readmission Group | `Figures/Prior Inpatient Visits by Readmission Group.png` |
| 6 | Time in Hospital by Readmission Group | `Figures/Time in Hospital by Readmission Group.png` |
| 7 | Standardized Differences - Early Readmission vs Other Encounters | `Figures/Standardized Differences - Early Readmission vs Other Encounters.png` |
| 8 | Correlation Between Numeric Utilization Variables | `Figures/Correlation Between Numeric Utilization Variables.png` |
| 9 | Early Readmission Rate by Race | `Figures/Early Readmission Rate by Race.png` |
| 10 | Early Readmission Rate by Gender | `Figures/Early Readmission Rate by Gender.png` |
| 11 | Early Readmission Rate by Primary Diagnosis Group | `Figures/Early Readmission Rate by Primary Diagnosis Group.png` |
| 12 | Confusion Matrices - All Models | `Figures/Confusion Matrices - All Models.png` |
| 13 | XGBoost - With vs Without Threshold | `Figures/XGBoost - With vs Without Threshold.png` |
| 14 | Top Features - All Models | `Figures/Top Features - All Models.png` |
| 15 | FNR Gap Across Patient Subgroups | `Figures/FNR Gap Across Patient Subgroups.png` |
| 16 | Age Subgroup Disparity and Support | `Figures/Age Subgroup Disparity and Support.png` |
| 17 | ROC Curve - XGBoost (stage B) | `Figures/ROC Curve - XGBoost.png` |
