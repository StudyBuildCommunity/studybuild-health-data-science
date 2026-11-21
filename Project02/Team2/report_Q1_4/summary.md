# Readmission Risk Modeling: EDA and Baseline Pipeline Report
**Digital Health & Medical Data Science Track**

## Abstract
This technical report addresses questions Q1 through Q4 for the diabetic hospital readmission dataset (101,766 encounters spanning 1999–2008). It covers exploratory data analysis, data quality assessment, feature association analysis, and a rigorous baseline machine learning pipeline designed to prevent data leakage.

---

## Q1: Population Overview & Data Quality Assessment

* **Dataset Scope:** The dataset contains 101,766 inpatient encounters across 130 US hospitals.
* **Target Distribution:** The target variable `readmitted` shows significant class imbalance:
  * No Readmission (`NO`): 54,864 encounters (53.9%)
  * Readmission after 30 days (`>30`): 35,545 encounters (34.9%)
  * Early Readmission within 30 days (`<30`): 11,357 encounters (11.2%)
* **Data Quality Issues:** Severe missingness was detected in specific columns, notably `weight` (>96% missing values), justifying its exclusion from predictive modeling.

---

## Q2: Clinical Measurements Across Readmission Groups

* Comparative statistical analysis using group aggregations indicates that prior utilization metrics—specifically `number_inpatient`—exhibit the most striking differences across readmission groups.
* Patients readmitted within 30 days present a higher mean number of prior inpatient visits, longer hospital stays, and increased medication counts compared to non-readmitted cohorts.

---

## Q3: Association Analysis & Clinical Causation

* A binary target (`target_early_readmit`) was formulated to isolate early readmissions (`<30`).
* **Important Caveat:** While variables such as prior inpatient visits strongly correlate with early readmissions, observed statistical associations do not prove direct causation. These features act as clinical risk proxies reflecting chronic disease severity and outpatient care gaps.

---

## Q4: Preprocessing Pipeline & Baseline Model

* **Data Leakage Mitigation:** Data splitting was strictly executed at the unique patient level (`patient_nbr`) to ensure encounters from the same patient do not span across both training and testing sets.
* **Pipeline Architecture:** Scikit-Learn pipelines were built incorporating median imputation and standard scaling for numerical features, alongside one-hot encoding for categorical attributes.
* **Baseline Performance:** A Logistic Regression model configured with balanced class weights served as the baseline predictor, establishing benchmark evaluation metrics.
