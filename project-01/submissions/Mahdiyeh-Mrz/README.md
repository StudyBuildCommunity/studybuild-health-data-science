<div align="center">

# 🩺 Maternal Health Risk Stratification

**Interpretable Machine Learning for Maternal Risk Screening**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-SQLite-07405E?logo=sqlite&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

*Statistical analysis, exploratory data analysis, and interpretable ML on the UCI Maternal Health Risk dataset*

</div>

---

## 📌 Overview

Digital health systems and low-cost IoT sensors now make it possible to monitor blood pressure, glucose, temperature, and heart rate at scale. This project analyzes real-world sensor data collected from hospitals and community clinics in **rural Bangladesh**, asking:

> Can a small set of routine physiological variables distinguish **Low / Mid / High** maternal risk — and can a simple, interpretable model flag High-Risk patients with useful sensitivity?

The goal isn't the highest possible accuracy — it's a transparent, explainable **screening baseline** that a health worker could actually reason about.

## 📊 Dataset

| Variable | Unit | Description |
|---|---|---|
| `Age` | years | Maternal age |
| `SystolicBP` / `DiastolicBP` | mmHg | Blood pressure |
| `BS` | mmol/L | Blood sugar |
| `BodyTemp` | °F | Body temperature |
| `HeartRate` | bpm | Heart rate |
| `RiskLevel` | — | Target: Low / Mid / High |

**Cleaning summary:**

| Step | Result |
|---|---|
| Raw records | 1,014 |
| Missing values | 0 — none found |
| Exact duplicates | 562 — kept (plausible sensor rounding) |
| Invalid `HeartRate` (= 7 bpm) | 2 — removed |
| **Final dataset** | **1,012** patients (Low 404 · Mid 336 · High 272) |

- **Parallel SQL layer:** 9 documented SQLite queries reproduce every group summary and threshold check directly on the cleaned table — no pandas required.
- **Leakage-safe split:** stratified 80/20 train/test (`random_state=42`), `StandardScaler` fit on training data only.
- **Two models trained and compared:**

```python
# Multinomial Logistic Regression
scaler = StandardScaler().fit(X_train)
logreg = LogisticRegression(max_iter=2000, random_state=42)
logreg.fit(scaler.transform(X_train), y_train)

# Decision Tree — final model (shallow & interpretable)
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=42)
tree.fit(X_train, y_train)
```

The tree stays shallow (`max_depth=4`) so every split — minimizing Gini impurity `1 − Σpₖ²` — can be read as a plain-language threshold rule.

## 🔍 Key Findings

- **Blood Sugar is the strongest signal** — mean 12.12 mmol/L (High) vs. 7.22 (Low); a simple `BS ≥ 11` rule catches 62.5% of High-Risk patients vs. just 1–9.5% of Low/Mid.
- **Blood pressure & age rise with risk**, but Low/Mid groups overlap significantly.
- **Heart Rate barely separates the classes** — no single variable is a perfect predictor on its own.
- The **BS × Systolic BP** relationship shows High-Risk patients clustering at high values of both, hinting that `RiskLevel` may itself be partly threshold-derived.

## 🏆 Results

| Model | Accuracy | Macro F1 | High-Risk Recall | High-Risk Precision |
|---|---|---|---|---|
| Logistic Regression | 0.586 | 0.584 | 0.782 | 0.768 |
| **Decision Tree ✅ (selected)** | **0.680** | **0.670** | **0.836** | **0.920** |

**Per-class performance (Decision Tree):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Low Risk | 0.614 | 0.864 | 0.718 |
| Mid Risk | 0.564 | 0.328 | 0.415 |
| High Risk | 0.920 | 0.836 | 0.876 |

**Feature importance:** Blood Sugar (~50%) and Systolic BP (~33%) drive most of the model's decisions — fully consistent with the EDA. The main confusion is between Low and Mid classes, not High-Risk detection.

## 💡 Conclusion

Across three independent methods — descriptive statistics, SQL queries, and machine learning — **Blood Sugar and blood pressure consistently emerged as the dominant risk drivers.** The final Decision Tree reached **83.6% recall** and **92% precision** on High-Risk patients: a strong result for an interpretable baseline.

> ⚠️ This is a transparent research baseline — **not a validated clinical decision tool.**

## ⚠️ Limitations

- Rural Bangladesh cohort only — limited generalizability
- Only 6 predictors (no obstetric history, labs, comorbidities)
- 562 duplicate rows reduce effective sample diversity
- No external validation cohort
- Small High-Risk test set (n = 55) → recall has sampling variability

## 🚀 Getting Started

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

python pipeline.py                          # run the full analysis
sqlite3 maternal_health.db < queries.sql    # run the SQL layer
```

## 👤 Author

**Mahdiyeh Mirzaei**  
🏢 StudyBuild  
🔗 [github.com/mahdiyeh-mirzaei-v2](https://github.com/mahdiyeh-mirzaei-v2)

<div align="center">
<sub>Educational research project — not intended for clinical diagnosis or decision-making.</sub>
</div>
