# Maternal Health Risk Stratification — Interpretable Machine Learning with Python

**Health & Medical Data Science Track — Project 01**

> ⚠️ **Educational project only.** This is a risk-stratification exercise for practicing
> health-data analysis, baseline modeling, and honest evaluation. It is **not** a diagnostic or
> clinical decision-making tool and must not be used as one.

> 📋 For a full declaration of every method/technique used in this project and why it was
> chosen (cleaning, EDA, SQL, modeling, evaluation, reproducibility), see
> **[`METHODOLOGY.md`](METHODOLOGY.md)**.

## 1. Clinical Problem

A maternal-health analytics team wants to understand patterns associated with maternal risk
level (Low / Mid / High) from five simple vital-sign measurements, and to test whether a
transparent, simple baseline model can support risk stratification.

## 2. Dataset

- **Source:** [UCI Maternal Health Risk](https://archive.ics.uci.edu/dataset/863/pregnant+health+risk) — DOI [10.24432/C5DP5D](https://doi.org/10.24432/C5DP5D), License **CC BY 4.0**
- Collected via an IoT-based risk-monitoring system in hospitals, community clinics, and
  maternal-health settings in **rural Bangladesh**.
- 1,013 instances (UCI), 6 input features, 1 categorical target (`RiskLevel`).
- See [`data/README.md`](data/README.md) for full variable definitions and citation.

| Variable | Meaning |
|---|---|
| `Age` | Maternal age (years) |
| `SystolicBP` / `DiastolicBP` | Blood pressure (mmHg) |
| `BS` | Blood sugar (mmol/L) |
| `BodyTemp` | Body temperature (°F) |
| `HeartRate` | Heart rate (bpm) |
| `RiskLevel` | Target: `low risk` / `mid risk` / `high risk` |

## 3. Data Quality Checks

| Check | Finding |
|---|---|
| Missing values | 0 |
| Exact duplicate rows | 562 / 1,014 (55.4%) |
| Implausible values | 2 rows with `HeartRate = 7` bpm (physiologically implausible) |

**Duplicates were kept**, not dropped: the source sensors round BP/BS/BodyTemp coarsely, so many
different patients legitimately share identical readings — dropping duplicates would silently
distort the class balance. This heavy rounding is instead documented as a **limitation** (lower
effective information content than the raw row count suggests).

**The 2 rows with `HeartRate = 7` bpm were removed** — a resting heart rate that low is not
compatible with a monitored, ambulatory patient and is treated as a sensor/entry error, not a
real reading. Final analysis dataset: **1,012 rows**.

## 4. EDA Findings (Q1–Q3)

- Class balance after cleaning: Low 404 (40%), Mid 336 (33%), High 272 (27%) — moderately
  imbalanced, which is why accuracy alone is not used to judge model quality (see §6).
- **Blood Sugar (BS)** shows the clearest, most monotonic separation across risk groups — High
  Risk patients have a markedly higher median and wider spread.
- **Systolic and Diastolic BP** step up from Low → Mid → High, with more overlap than BS.
- **Age** shows a weaker, noisier relationship with risk group; several very young (≤14) / older
  (≥55) ages appear and are flagged as a possible data-quality curiosity, not corrected.
- **Heart Rate** shows almost no separation across risk groups.
- In a 2-D view (BS vs Systolic BP), High Risk patients cluster toward higher values of both,
  but no single threshold cleanly separates the classes — **association, not proof of
  causation**: the labels may themselves have been assigned partly from BP/BS thresholds by the
  original clinicians, which can inflate this apparent relationship.

All figures are in [`figures/`](figures/) with titles, axis labels, and interpretations in the
notebook.

## 5. Model (Q4)

Two simple, interpretable baselines were trained on an 80/20 stratified train/test split
(`random_state=42`), with `StandardScaler` fit on the **training set only** to prevent leakage:

- **Logistic Regression** (multinomial, standardized features)
- **Decision Tree** (`max_depth=4`, `min_samples_leaf=15` — kept shallow on purpose)

## 6. Evaluation (Q5) — Accuracy Alone Is Not Enough

Because the dataset is imbalanced and **missing a High Risk patient is more clinically costly
than confusing Low/Mid Risk**, both models are compared on accuracy, macro F1, and — with
priority — **High Risk Recall**.

| Model | Accuracy | Macro F1 | High-Risk Recall | High-Risk Precision |
|---|---|---|---|---|
| Logistic Regression | 0.586 | 0.584 | 0.782 | 0.768 |
| **Decision Tree (depth=4)** | **0.680** | **0.670** | **0.836** | **0.920** |

**Decision Tree selected** as the final baseline — better on every metric, and its High Risk
class has the *best* precision/recall of the three classes for both models (reassuring from a
safety standpoint; the harder problem is Low vs. Mid).

Full class-wise report (Decision Tree, test set, n=203):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low Risk | 0.614 | 0.864 | 0.718 | 81 |
| Mid Risk | 0.564 | 0.328 | 0.415 | 67 |
| High Risk | 0.920 | 0.836 | 0.876 | 55 |

Confusion matrices for both models: [`figures/07a_confusion_logreg.png`](figures/07a_confusion_logreg.png), [`figures/07b_confusion_tree.png`](figures/07b_confusion_tree.png).

## 7. Error Analysis (Q6)

- 65 / 203 test cases misclassified (32.0%).
- Dominant error: **Mid Risk predicted as Low Risk (42 of 65 errors)** — consistent with the
  heavy Low/Mid overlap seen in EDA.
- **9 of 55** true High Risk test cases were predicted as a lower risk class (2 as Low, 7 as
  Mid) — these are the most clinically important errors (false negatives on the highest-risk
  class) and the first target for future improvement (e.g., class weighting, threshold tuning).

## 8. Feature Interpretation (Q7)

Decision Tree Gini-based feature importance:

| Feature | Importance |
|---|---|
| BS | 0.502 |
| SystolicBP | 0.330 |
| BodyTemp | 0.115 |
| Age | 0.045 |
| DiastolicBP | 0.008 |
| HeartRate | 0.000 |

Blood Sugar dominates the model's splits, consistent with the EDA. **Importance describes what
the model used to separate the given labels — it is not evidence of clinical causation**, and
may partly reflect that the labels themselves were assigned using similar thresholds.

## 9. Clinical Interpretation & Limitations (Q8)

Blood Sugar and Systolic BP are the variables most associated with High Risk in this sample;
Heart Rate adds little value. The Decision Tree baseline reaches ~68% accuracy and ~84% High
Risk Recall — useful as a teaching example, but **not reliable enough to stand alone**, given:

- **Sample size & geography** — ~1,000 patients from rural-Bangladesh IoT settings; may not
  generalize elsewhere.
- **Feature availability** — only 6 vital-sign features; no gestational age, obstetric history,
  labs, or comorbidities.
- **Label quality** — `RiskLevel` provenance is unknown and may already encode BP/BS thresholds.
- **Measurement precision** — heavy rounding produced ~55% duplicate rows, reducing effective
  data diversity.
- **No external validation** — never tested outside this single dataset, and never reviewed by
  a clinician.

Before any real use: a larger, more diverse validated cohort, richer clinical features,
prospective comparison against clinician judgment, a cost-sensitive decision rule that further
prioritizes catching High Risk cases, and formal clinical/regulatory review.

## 10. Repository Structure

```
maternal-health-risk/
├── README.md                    # this file
├── METHODOLOGY.md               # full declaration: every method used, for what, and why
├── requirements.txt
├── notebooks/
│   └── analysis.ipynb           # full, executed, reproducible analysis (Q1–Q8)
├── src/
│   └── model.py                 # same pipeline as a standalone script
├── sql/
│   ├── queries.sql              # 9 documented SQL queries (EDA reproduced in pure SQL)
│   ├── run_queries.py           # executes queries.sql, writes sql_results.md
│   └── sql_results.md           # generated output of every query
├── figures/                     # all 10 saved charts (PNG, titled + labeled)
├── report/
│   ├── summary.md                # short clinical-analysis summary
│   └── summary.pdf               # same summary, PDF
└── data/
    ├── Maternal_Health_Risk_Data_Set.csv
    ├── maternal_health.db        # SQLite DB of the cleaned data (used by sql/)
    └── README.md                 # dataset source, license, citation
```

## 10a. SQL Analysis

In addition to the Python/pandas pipeline, the core EDA (Q1–Q3) is reproduced in pure SQL
against a SQLite database built from the same cleaned data:

```bash
# rebuild the database from the cleaned CSV (optional - already included)
python - <<'PY'
import pandas as pd, sqlite3
df = pd.read_csv("data/Maternal_Health_Risk_Data_Set.csv", encoding="utf-8-sig")
df["RiskLevel"] = df["RiskLevel"].str.strip().str.lower()
df = df[df["HeartRate"] >= 30]
sqlite3.connect("data/maternal_health.db").close()
conn = sqlite3.connect("data/maternal_health.db")
df.to_sql("maternal_health", conn, index=False, if_exists="replace")
PY

# run every query and regenerate sql/sql_results.md
python sql/run_queries.py
```

`sql/queries.sql` covers: class distribution, summary statistics, a data-quality re-check,
per-group mean/spread comparisons (`GROUP BY`, window functions), threshold-based High-Risk
separation checks (`CASE WHEN`), and an age-band cross-tab — each with an inline `-- Purpose:`
comment. See [`METHODOLOGY.md`](METHODOLOGY.md) §7 for the full reasoning, including a documented
caveat about one query's unit-dependent ranking.

## 11. How to Run

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Option A: run the full pipeline as a script (prints all results, saves all figures)
python src/model.py

# Option B: run the notebook end-to-end
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
# or open notebooks/analysis.ipynb in Jupyter/Colab and Run All
```

A fixed random seed (`random_state=42`) is used throughout for reproducibility; both entry
points (`src/model.py` and `notebooks/analysis.ipynb`) implement the identical pipeline and
produce identical figures/metrics.


## Final data consistency

The final analysis is standardized on **1,012 observations**. The supplied raw export contains 1,014 observations; exactly two records with `HeartRate = 7` bpm are excluded as physiologically implausible. Final class counts are **Low Risk = 404, Mid Risk = 336, High Risk = 272**. The cleaned CSV, SQLite database, SQL results, figures, dashboards, and article are aligned to these final counts.

## Article

The `article/` folder contains the complete English LaTeX article, compiled PDF, bibliography/reference material embedded in the `.tex`, and all figures used in the article.
