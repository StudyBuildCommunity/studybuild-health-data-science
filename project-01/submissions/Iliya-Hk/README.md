# Maternal Health Risk Analysis

An exploratory data analysis and classification project on the **Maternal Health Risk Data Set**, looking at which clinical measurements are most associated with high-risk pregnancies, and how well a simple, interpretable model (a Decision Tree) can flag those cases.

I went into this wanting an answer to a pretty practical question: *if we only had a handful of vitals — blood pressure, blood sugar, body temp, heart rate, age — could we build something interpretable enough for a clinician to trust, but still useful as a screening signal?* Short answer: yes, with caveats. Long answer below.

## Dataset

- **Source file:** `data/Maternal Health Risk Data Set.csv`
- **Raw shape:** 1,014 records × 7 columns
- **Columns:** `Age`, `SystolicBP`, `DiastolicBP`, `BS` (blood sugar), `BodyTemp`, `HeartRate`, `RiskLevel` (target: low / mid / high risk)
- No missing values in any column.

Raw class distribution:

| RiskLevel | Count | % |
|---|---|---|
| Low risk | 406 | 40.0% |
| Mid risk | 336 | 33.1% |
| High risk | 272 | 26.8% |

## Data Cleaning

The raw data had a couple of issues worth flagging before doing anything else:

- **562 exact duplicate rows** — more than half the dataset. These were dropped.
- **2 records with `HeartRate = 7`**, which is obviously not a real human heart rate (min in the rest of the data is 60). These were removed as data entry errors.
- No implausible values were found for Age, SystolicBP, DiastolicBP, BS, or BodyTemp when checked against reasonable clinical ranges.
- No records had `DiastolicBP > SystolicBP` (which would be physiologically impossible).

After deduplication and removing the bad heart rate readings: **451 records** remained.

On top of that, an **Isolation Forest** (contamination = 3%) was used to catch multivariate outliers that a single-column range check wouldn't flag — combinations of values that are individually plausible but jointly unusual. This trimmed the dataset down to a final analysis set of **437 records**.

Final class distribution (post-cleaning):

| RiskLevel | Count | % |
|---|---|---|
| Low risk | 231 | 52.9% |
| Mid risk | 105 | 24.0% |
| High risk | 101 | 23.1% |

(Note: cleaning shifted the class balance somewhat vs. the raw data — worth keeping in mind if you compare results against the uncleaned set.)

## Exploratory Analysis

For each clinical variable, I compared distributions across the three risk groups using boxplots, violin plots, a normalized heatmap of group means, and a radar chart. A one-way ANOVA confirmed that **all six clinical variables differ significantly across risk groups** (p < 0.001 for every variable), with blood sugar showing the largest F-statistic by a wide margin (F = 114.15).

To quantify *how much* high-risk and low-risk patients actually differ, I calculated Cohen's d for each variable (high risk vs. low risk):

| Variable | Mean (High Risk) | Mean (Low Risk) | Cohen's d | Effect size |
|---|---|---|---|---|
| BS | 11.17 | 7.20 | 1.734 | Large |
| SystolicBP | 119.49 | 105.37 | 0.808 | Large |
| BodyTemp | 99.23 | 98.36 | 0.650 | Medium |
| DiastolicBP | 81.54 | 72.72 | 0.646 | Medium |
| Age | 33.73 | 27.37 | 0.462 | Small |
| HeartRate | 76.48 | 73.04 | 0.456 | Small |

Blood sugar is the clear standout here — the gap between high- and low-risk patients is nearly two standard deviations, which is a substantial effect by any convention.

A correlation heatmap also showed the expected strong positive relationship between systolic and diastolic BP (r = 0.78), and a PCA projection of the feature space showed real overlap between risk groups rather than clean separation — a hint that this isn't going to be a trivial classification problem.

## Modeling

**Target:** Binary `HighRisk` flag (High Risk = 1, Low/Mid Risk = 0), since correctly catching high-risk patients is the clinically important task.

**Model:** `DecisionTreeClassifier` with `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5`, `class_weight='balanced'` — deliberately kept shallow and simple so the splits stay human-readable, at some cost to raw predictive power.

**Split:** 70/30 train-test split, stratified on the `HighRisk` flag (305 training records, 132 test records).

### Test set performance

| Metric | Score |
|---|---|
| Accuracy | 0.8712 |
| Precision (High Risk) | 0.6591 |
| Recall / Sensitivity (High Risk) | 0.9355 |
| F1-Score (High Risk) | 0.7733 |
| ROC-AUC | 0.9225 |

5-fold cross-validation on the training set gave a mean accuracy of **0.7672 (± 0.0706)** — noticeably lower and more variable than the single test-set split, which is worth keeping in mind (437 records is a small dataset to be splitting three ways).

### Confusion matrix (High Risk vs. Low/Mid Risk)

|  | Predicted Low/Mid | Predicted High |
|---|---|---|
| **Actual Low/Mid** | 86 | 15 |
| **Actual High** | 2 | 29 |

Only **2 high-risk patients** were missed by the model on the test set — recall of 93.6%. Precision is lower (65.9%), meaning the model over-flags some low/mid-risk patients as high risk. That trade-off is intentional: in a screening context, a false alarm costs a follow-up check, while a missed high-risk case can cost a lot more.

### Feature importance

| Feature | Importance |
|---|---|
| BS (blood sugar) | 0.568 |
| SystolicBP | 0.144 |
| BodyTemp | 0.123 |
| Age | 0.079 |
| HeartRate | 0.044 |
| DiastolicBP | 0.041 |

Blood sugar dominates the tree — it's the root split (BS > 7.95) and drives the majority of the model's decision-making. This lines up with the effect size analysis above and with clinical intuition: elevated blood sugar is a hallmark of gestational diabetes, a well-known pregnancy risk factor.

### 3-class confusion matrix (for reference)

A second tree was trained on the full 3-class target (`low / mid / high risk`) to see where the confusion actually happens:

|  | Pred. Low | Pred. Mid | Pred. High |
|---|---|---|---|
| **Actual Low** | 50 | 14 | 2 |
| **Actual Mid** | 20 | 5 | 10 |
| **Actual High** | 1 | 5 | 25 |

Most of the confusion sits between **Low and Mid risk** (Mid→Low: 20 cases, Low→Mid: 14 cases) — which makes sense given "mid risk" is inherently a fuzzier, in-between category. High-risk vs. low-risk confusion is rare (only 1 High→Low, 2 Low→High), which is reassuring: the model is much better at telling apart the two extremes than it is at nailing the middle category.

## Key Takeaways

1. **Blood sugar is the single strongest signal** for high-risk pregnancy classification, both in raw effect size (Cohen's d = 1.73) and in the trained model's feature importance (57%).
2. Systolic blood pressure, body temperature, and diastolic blood pressure all show medium-to-large effect sizes and contribute meaningfully to the model.
3. A simple, shallow decision tree gets you **93.6% recall** on high-risk cases — good for a screening tool, though at the cost of some false positives (66% precision).
4. Cross-validation results (76.7% mean accuracy) suggest the single test-set numbers above are a bit optimistic; take the headline metrics with a grain of salt given the small sample size.
5. The dataset had significant quality issues (55%+ duplicate rate) that needed cleaning before any of this analysis was meaningful.

## Important Caveats

This project is an **educational data science exercise**, not a validated clinical tool. A few things to flag explicitly:

- No prospective clinical validation has been done — this is retrospective data.
- The dataset has no gestational age, medical history, or demographic information beyond age.
- Correlation ≠ causation: feature importance tells you what the model *uses*, not what physiologically *causes* risk.
- 437 records post-cleaning is a small sample for training and evaluating a model, and results should be interpreted with appropriate humility.
- This should never be used for actual clinical decisions without proper regulatory review, external validation, and clinician oversight.

## Tech Stack

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn` (IsolationForest, DecisionTreeClassifier, SelectKBest, PCA, train/test split & metrics)

## Repo Structure

```
├── data/
│   └── Maternal Health Risk Data Set.csv
│   └── README.md
├── figures/
│   ├── risk_distribution.png
│   ├── clinical_variables_by_risk.png
│   ├── heatmap_means.png
│   ├── violin_plots.png
│   ├── radar_chart.png
│   ├── Maternal Age Groups.png
│   ├── Multivariate Pair Plot.png
│   ├── Correlation of Clinical Variables.png
│   ├── high_risk_comparison.png
│   ├── feature_importance_anova.png
│   ├── decision_tree_full.png
│   ├── decision_tree_simple.png
│   ├── confusion_matrix_dt.png
│   ├── confusion_matrix_3class_dt.png
│   ├── pca_risk_groups_dt.png
│   └── feature_importance_and_roc_dt.png
├── Notebooks/
│   └── analysis.ipynb
├── Report/
│   └── analysis.pdf
└── README.md
```
