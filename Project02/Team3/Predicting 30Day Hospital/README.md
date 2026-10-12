# 🏥 Medical Readmission Risk Prediction

### Data-Driven Clinical Analytics & Machine Learning

A reproducible medical analytics project focused on **30-day hospital readmission risk prediction** using statistical analysis, exploratory data analysis (EDA), machine learning, model evaluation, and subgroup/fairness assessment.

The project is designed as a **clinical decision-support research workflow**, not as a replacement for physician judgment or an autonomous diagnostic system.


## 📌 Project Overview

Hospital readmission within 30 days is an important quality and resource-management indicator. This project analyzes patient-level clinical and demographic information to identify patterns associated with readmission and evaluate machine-learning models for risk prediction.

The workflow combines:

- 🩺 Clinical data analysis
- 📊 Exploratory and descriptive statistics
- 🔬 Statistical hypothesis testing
- 🤖 Machine learning classification
- 📈 Model performance evaluation
- ⚖️ Subgroup and fairness analysis
- 📋 Clinical interpretation and limitations
- 🧪 Reproducible research reporting


## 🔑 Headline Results

The task is **binary** classification (early readmission `<30` days = 1 vs not = 0).

| Result | Value (held-out, patient-separated test set) |
|---|---|
| Best model | XGBoost — ROC-AUC 0.683, PR-AUC 0.238 |
| **Decision threshold (key strength)** | Validation-tuned threshold **0.41** raises Recall from 0.598 (default 0.50) to **0.797**; ROC-AUC unchanged. Cost: Precision 0.152, Accuracy 0.474 |
| **Critical fairness finding** | **Severe age-subgroup disparity** — FNR gap 0.885–0.967 in all four models, FNR = 1.00 in at least one age band (bands hold only 1 and 6 positives, so the exact size is uncertain) |
| Strongest predictor | Prior inpatient visits (SMD 0.415) |

📄 Full reports: [`Project_Report.md`](Project_Report.md) (English) · [`Project_Report_FA.md`](Project_Report_FA.md) (فارسی) · 📊 Dashboard: `Dashboard/Diabetes_Dashboard.html` · 🖼️ 17 figures with exact paths: Section 10.1 of the report.


## 🎯 Objectives

| Objective | Description |
|---|---|
| 🏥 Readmission Analysis | Investigate factors associated with 30-day readmission |
| 📊 Statistical Analysis | Summarize distributions, associations, and group differences |
| 🤖 Prediction | Develop and compare classification models |
| 📈 Evaluation | Assess discrimination and classification performance |
| ⚖️ Fairness | Examine model behavior across demographic subgroups |
| 🩺 Clinical Insight | Translate analytical findings into clinically meaningful observations |
| 🔁 Reproducibility | Provide a structured and documented analytical workflow |


## 🧬 Analytical Workflow

```text
Raw Clinical Data
       ↓
Data Cleaning & Validation
       ↓
Exploratory Data Analysis
       ↓
Descriptive & Inferential Statistics
       ↓
Feature Preparation
       ↓
Machine Learning Models
       ↓
Model Evaluation
       ↓
Subgroup / Fairness Analysis
       ↓
Clinical Interpretation
       ↓
Research Report
```


## 📊 Methods

### Statistical Methods

The project uses statistical techniques appropriate for clinical data analysis, including:

- Descriptive statistics
- Distribution and missing-data assessment
- Interquartile range (IQR) analysis
- Standardized Mean Difference (SMD)
- Cramér's V for categorical association
- Correlation analysis
- Confidence intervals
- Group-level comparisons

### 🤖 Machine Learning

Several classification approaches are evaluated, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Model performance is examined using clinically relevant classification metrics rather than relying on accuracy alone.


## 📈 Evaluation Metrics

| Metric | Purpose |
|---|---|
| ROC-AUC | Overall ranking/discrimination ability |
| PR-AUC | Performance under class imbalance |
| Precision | Proportion of predicted positives that are true positives |
| Recall / Sensitivity | Ability to identify positive cases |
| F1-Score | Balance between precision and recall |
| F2-Score | Places greater emphasis on recall |
| Specificity | Ability to identify negative cases |
| False Negative Rate | Important for assessing missed high-risk patients |
| Confusion Matrix | Detailed classification outcomes |


## 🧠 Clinical Perspective

The project emphasizes that predictive performance should be interpreted in a **clinical context**.

A model with a high discrimination score does not automatically mean that it is clinically useful. In a readmission setting, false negatives can be particularly important because patients at elevated risk may be missed.

Therefore, the analysis considers:

- Model discrimination
- Sensitivity and recall
- False-negative behavior
- Class imbalance
- Subgroup performance
- Clinical interpretability
- Data limitations


## ⚖️ Fairness & Subgroup Analysis

Model behavior is examined across relevant demographic groups to identify potential performance differences.

The analysis considers variables such as:

- 👩 Gender
- 👥 Race / ethnicity
- 🎂 Age groups

Subgroup analysis is used to identify differences in predictive performance and should not be interpreted as proof of causality or absence of bias.

---

## 📁 Repository Structure

```text
medical-readmission-risk/
│
├── 📄 README.md
├── 📄 medical_readmission_article.tex
├── 📄 medical_readmission_article.pdf
│
├── 📂 Figures/
│   ├── 📊 exploratory_analysis/
│   ├── 📈 model_evaluation/
│   └── ⚖️ subgroup_analysis/
│
├── 📂 data/
│   └── clinical_dataset.*
│
├── 📂 src/
│   ├── data_preprocessing.*
│   ├── statistical_analysis.*
│   ├── modeling.*
│   └── evaluation.*
│
├── 📂 notebooks/
│   └── analysis.*
│
└── 📂 results/
    ├── tables/
    └── model_outputs/
```

> File and folder names can be adjusted to match the actual repository implementation.


## 📚 Research Report

A complete academic report accompanies the project and documents the methodology, statistical analysis, machine-learning experiments, clinical interpretation, fairness assessment, limitations, and reproducibility considerations.

📄 **Main report:** `medical_readmission_article.pdf`

📝 **LaTeX source:** `medical_readmission_article.tex`


## 🛠️ Technologies

| Category | Tools |
|---|---|
| 🐍 Programming | Python |
| 📊 Data Analysis | Pandas, NumPy |
| 🔬 Statistics | SciPy, statistical testing |
| 🤖 Machine Learning | Scikit-learn, XGBoost |
| 📈 Visualization | Matplotlib, Seaborn |
| 📄 Reporting | LaTeX / XeLaTeX |
| 🗂️ Version Control | Git / GitHub |


## 🔬 Reproducibility

The project follows a reproducible research structure:

1. Prepare and validate the dataset.
2. Perform exploratory analysis.
3. Apply statistical methods.
4. Prepare model features.
5. Train classification models.
6. Evaluate models using multiple metrics.
7. Perform subgroup analysis.
8. Document findings and limitations.
9. Generate the final research report.

Random seeds, preprocessing steps, evaluation procedures, and analytical assumptions should be documented alongside the code to support reproducibility.


## ⚠️ Limitations

This project should be interpreted within the limitations of the available dataset and study design.

Important considerations include:

- Observational data do not establish causality.
- Dataset characteristics may limit generalizability.
- Class imbalance can affect predictive metrics.
- Demographic variables may reflect complex social and clinical factors.
- Model performance does not guarantee clinical utility.
- External validation is required before real-world clinical deployment.
- Predictions should not replace professional medical assessment.


## 🩺 Clinical Use Disclaimer

> **Research and educational use only.**
>
> This project is intended for data-science research, education, and methodological demonstration. It is **not a medical device**, does not provide medical advice, and should not be used independently to diagnose, treat, triage, or make clinical decisions about individual patients.

Any real-world deployment would require appropriate clinical validation, external validation, governance, privacy review, monitoring, and regulatory assessment.


## 📖 References

The accompanying research article provides the detailed bibliography and methodological references supporting the statistical and machine-learning approaches used in this project.

See:

📄 `medical_readmission_article.pdf`

📄 `medical_readmission_article.tex`


