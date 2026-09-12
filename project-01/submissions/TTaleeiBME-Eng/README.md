# Maternal Health Risk Stratification – v3 (Improved High-Risk Recall)

**StudyBuild Project 01 – still 3-class (Low / Mid / High)**

## What changed in v3

| Improvement | Detail |
|-------------|--------|
| `class_weight="balanced"` | Both Logistic Regression and Decision Tree |
| Limited GridSearchCV | max_depth ∈ {3,4,5,6}, min_samples_leaf ∈ {3,5,8}, criterion ∈ {gini,entropy} |
| Scoring | `recall_macro` (raises High-risk Recall without leaving 3-class) |
| Pipeline | StandardScaler inside LR Pipeline; no data leakage |
| Result | **High-risk Recall 0.78** (was ~0.61) · Accuracy 0.70 |

We did **not** convert the problem to binary High vs Rest (that would inflate metrics but violate the project brief).

## Main test-set results (n = 91)

| Model | Accuracy | High-Risk Recall | High-Risk F1 |
|-------|----------|------------------|--------------|
| Decision Tree (tuned, balanced) | **70.3 %** | **0.78** | 0.77 |
| Logistic Regression (balanced) | 61.5 % | 0.57 | 0.67 |

Best DT params found by GridSearch: `criterion=entropy`, `max_depth=4`, `min_samples_leaf=5`.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
# or
python run_analysis.py
```

## Repository layout

```
maternal-health-risk-v3/
├── README.md
├── requirements.txt
├── data/
├── notebooks/analysis.ipynb
├── src/model.py
├── figures/   (10 plots)
└── report/summary.pdf
```

## Limitations (unchanged)
- Original data 55 % exact duplicates
- Small cleaned n = 451, single geography
- Mid-risk class remains hard to separate
- No external validation

**Disclaimer:** Educational only – not a diagnostic tool.
