# Maternal Health Risk Stratification – Interpretable Machine Learning with Python

**StudyBuild – Health & Medical Data Science Track – Project 01**

## Clinical Problem

Maternal mortality remains a critical concern in low- and middle-income countries. This educational project explores a public dataset collected via an IoT-based risk-monitoring system in rural Bangladesh. The goal is to identify clinically relevant patterns associated with Low, Mid and High maternal risk levels and to build a simple, interpretable baseline classifier.

**Important:** This is an educational risk-stratification exercise, **not** a diagnostic tool.

## Dataset

- **Source:** UCI Machine Learning Repository – Maternal Health Risk  
  https://archive.ics.uci.edu/dataset/863/pregnant+health+risk  
- **DOI:** https://doi.org/10.24432/C5DP5D  
- **License:** CC BY 4.0  
- **Paper:** Ahmed et al. (2020) – Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT)

**Variables:** Age, SystolicBP, DiastolicBP, BS (blood sugar), BodyTemp, HeartRate, RiskLevel (target).

After rigorous data-quality checks we use a cleaned version of **451 unique records** (Version B).

## Repository Structure

```
maternal-health-risk/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── Maternal Health Risk Data Set.csv
├── notebooks/
│   ├── analysis.ipynb                          # Full reproducible analysis
├── src/
│   └── model.py                                # Reusable model helper functions
├── figures/
│   ├── q2_risk_feature_distributions.png
│   ├── q3_scatter_bs_vs_sysbp.png
│   ├── q4_confusion_matrices.png
│   ├── q5_confusion_matrix_detailed.png
│   ├── q6_error_distribution.png
│   ├── q7_feature_importance.png
│   └── q8_executive_summary_chart.png
└── report/
    ├── summary.pdf                             # Clinical analysis report (PDF)
```



## Key Findings (Summary)

| Question | Main Result |
|----------|-------------|
| Q1 | After removing 562 duplicates and 2 implausible HeartRate records → 451 clean samples. Class balance: Low 51.7 %, High 24.8 %, Mid 23.5 %. |
| Q2 | Blood sugar (BS) and SystolicBP show the clearest separation across risk groups. |
| Q3 | BS has the strongest association with High Risk (Pearson r ≈ 0.57). Association ≠ causation. |
| Q4 | Simple Decision Tree (max_depth=4) slightly outperforms Logistic Regression (68.1 % vs 67.0 % accuracy). |
| Q5 | Overall accuracy is misleading. High-Risk Recall is only 0.61; Mid-Risk Recall is ~0.10. |
| Q6 | Majority of errors are Mid → Low (62 %) and High → Low (28 %). Overlap in feature space is the main cause. |
| Q7 | Feature importance: BS (56.8 %), SystolicBP (26.4 %), BodyTemp (10.2 %), Age (6.7 %). DiastolicBP & HeartRate unused by the tree. |
| Q8 | Model may serve only as a preliminary screening aid. Additional variables (proteinuria, gestational age, BMI) are required before any real-world use. |

## Limitations

- Small sample size after cleaning (n = 451).
- High original duplication rate (55 %).
- Geographic limitation (rural Bangladesh).
- No external validation set.
- Missing important clinical covariates.

## License & Citation

Dataset: CC BY 4.0 (UCI).  
Please cite the original UCI page and the 2020 introductory paper when using this analysis.
