[README_DATA.md](https://github.com/user-attachments/files/33246851/README_DATA.md)
# README: Diabetic Data Dataset

## Dataset Overview

This dataset contains clinical and demographic information about patients with diabetes who were admitted to hospitals across different time periods. The primary objective of this dataset is to analyze the factors influencing **hospital readmission** of diabetic patients within specific timeframes (less than 30 days, more than 30 days, or no readmission).

This dataset is one of the most comprehensive and widely used datasets in the field of **Medical Machine Learning** and **Healthcare Analytics**. It is commonly used for binary or multi-class classification problems related to predicting hospital readmission.

---

## File Content

- **File Name:** `diabetic_data.csv`
- **File Format:** CSV (Comma-Separated Values)
- **Data Type:** Structured Data
- **Data Language:** English
- **Source:** Hospital data related to diabetic patients (typically derived from clinical databases such as HCUP or teaching hospitals)

---

## Dataset Objectives

1. **Predict hospital readmission** of diabetic patients within different timeframes
2. **Analyze factors influencing patient return** to the hospital
3. **Examine the impact of diabetes medications** on readmission rates
4. **Compare demographic and clinical characteristics** of readmitted vs. non-readmitted patients
5. **Application in Machine Learning and Medical Data Mining research**

---

## Dataset Structure

The dataset contains **50 columns (Features)** and **101,766 records**. Each record represents a hospital encounter for a diabetic patient.

### Column Descriptions

| #   | Column Name                | Data Type   | Description                                   |
| --- | -------------------------- | ----------- | --------------------------------------------- |
| 1   | `encounter_id`             | Numeric     | Unique identifier for each hospital encounter |
| 2   | `patient_nbr`              | Numeric     | Unique identifier for each patient            |
| 3   | `race`                     | Categorical | Patient's race                                |
| 4   | `gender`                   | Categorical | Patient's gender                              |
| 5   | `age`                      | Categorical | Patient's age group as intervals              |
| 6   | `weight`                   | Categorical | Patient's weight as intervals                 |
| 7   | `admission_type_id`        | Numeric     | Admission type identifier                     |
| 8   | `discharge_disposition_id` | Numeric     | Discharge disposition identifier              |
| 9   | `admission_source_id`      | Numeric     | Admission source identifier                   |
| 10  | `time_in_hospital`         | Numeric     | Length of hospital stay in days               |
| 11  | `payer_code`               | Categorical | Payer code                                    |
| 12  | `medical_specialty`        | Categorical | Admitting physician's specialty               |
| 13  | `num_lab_procedures`       | Numeric     | Number of lab tests performed                 |
| 14  | `num_procedures`           | Numeric     | Number of procedures performed                |
| 15  | `num_medications`          | Numeric     | Number of medications administered            |
| 16  | `number_outpatient`        | Numeric     | Number of outpatient visits in the past year  |
| 17  | `number_emergency`         | Numeric     | Number of emergency visits in the past year   |
| 18  | `number_inpatient`         | Numeric     | Number of inpatient visits in the past year   |
| 19  | `diag_1`                   | Categorical | Primary diagnosis (ICD-9 code)                |
| 20  | `diag_2`                   | Categorical | Secondary diagnosis (ICD-9 code)              |
| 21  | `diag_3`                   | Categorical | Tertiary diagnosis (ICD-9 code)               |
| 22  | `number_diagnoses`         | Numeric     | Total number of diagnoses recorded            |
| 23  | `max_glu_serum`            | Categorical | Blood glucose test result                     |
| 24  | `A1Cresult`                | Categorical | Hemoglobin A1C test result                    |
| 25  | `metformin`                | Categorical | Metformin prescription status                 |
| 26  | `repaglinide`              | Categorical | Repaglinide prescription status               |
| 27  | `nateglinide`              | Categorical | Nateglinide prescription status               |
| 28  | `chlorpropamide`           | Categorical | Chlorpropamide prescription status            |
| 29  | `glimepiride`              | Categorical | Glimepiride prescription status               |
| 30  | `acetohexamide`            | Categorical | Acetohexamide prescription status             |
| 31  | `glipizide`                | Categorical | Glipizide prescription status                 |
| 32  | `glyburide`                | Categorical | Glyburide prescription status                 |
| 33  | `tolbutamide`              | Categorical | Tolbutamide prescription status               |
| 34  | `pioglitazone`             | Categorical | Pioglitazone prescription status              |
| 35  | `rosiglitazone`            | Categorical | Rosiglitazone prescription status             |
| 36  | `acarbose`                 | Categorical | Acarbose prescription status                  |
| 37  | `miglitol`                 | Categorical | Miglitol prescription status                  |
| 38  | `troglitazone`             | Categorical | Troglitazone prescription status              |
| 39  | `tolazamide`               | Categorical | Tolazamide prescription status                |
| 40  | `examide`                  | Categorical | Examide prescription status                   |
| 41  | `citoglipton`              | Categorical | Citoglipton prescription status               |
| 42  | `insulin`                  | Categorical | Insulin prescription status                   |
| 43  | `glyburide-metformin`      | Categorical | Glyburide-metformin combination status        |
| 44  | `glipizide-metformin`      | Categorical | Glipizide-metformin combination status        |
| 45  | `glimepiride-pioglitazone` | Categorical | Glimepiride-pioglitazone combination status   |
| 46  | `metformin-rosiglitazone`  | Categorical | Metformin-rosiglitazone combination status    |
| 47  | `metformin-pioglitazone`   | Categorical | Metformin-pioglitazone combination status     |
| 48  | `change`                   | Categorical | Change in diabetes medication dosage          |
| 49  | `diabetesMed`              | Categorical | Was diabetes medication prescribed?           |
| 50  | `readmitted`               | Categorical | Target Variable: Readmission status           |

---

## Target Variable: `readmitted`

The target variable in this dataset is the `readmitted` column, which takes three values:

| Value | Description                         | Approximate Percentage |
| ----- | ----------------------------------- | ---------------------- |
| `NO`  | Patient was not readmitted          | ~54%                   |
| `>30` | Readmitted after 30 days            | ~35%                   |
| `<30` | Readmitted within less than 30 days | ~11%                   |

> **Note:** This dataset has Class Imbalance, which must be addressed during the modeling process.

---

## Required Preprocessing

1. Handling Missing Values
2. Categorical Encoding
3. Feature Engineering
4. Scaling
5. Handling Class Imbalance
6. Removing Unnecessary Columns

---

## Suggested Applications

- Binary Classification
- Multi-class Classification
- Feature Importance Analysis
- Machine Learning Models
- Statistical Analyses

---

## Important Notes

- This dataset contains real hospital data and may contain demographic biases.
- A single patient can have multiple records.
- ICD-9 codes are used for diagnoses.
- The `readmitted` variable is three-class.
- Suitable for educational and research purposes only.

---

## References

- UCI Machine Learning Repository
- Strack, B., et al. (2014)
- American Diabetes Association (ADA) Guidelines
- HCUP Databases

---

## Conclusion

The `diabetic_data.csv` dataset is a rich and comprehensive resource for analyzing hospital readmission of diabetic patients. With 50 clinical, demographic, and medication-related features, this dataset enables in-depth analysis and the development of powerful predictive models.

---

