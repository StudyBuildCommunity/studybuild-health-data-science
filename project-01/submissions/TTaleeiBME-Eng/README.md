# Maternal Health Risk Stratification – v4 (Label-noise aware)

**StudyBuild Project 01 – Health & Medical Data Science Track**

## What changed in v4

| Improvement | Detail |
|-------------|--------|
| Label-noise handling | Explicitly removed **215 rows** that share identical features but have conflicting RiskLevel labels (35 unique feature groups) |
| Exact duplicates | Still removed (standard practice) |
| HeartRate anomalies | 2 records with HeartRate = 7 removed |
| Final clean size | **380** high-quality records |
| Pipeline + class_weight | Kept (professional & improves High-risk Recall) |
| Still 3-class | Low / Mid / High (as required by the project brief) |

## Test-set results (n = 76)

| Model | Accuracy | High-Risk Recall | High-Risk F1 |
|-------|----------|------------------|--------------|
| **Decision Tree (tuned, balanced)** | **80.3 %** | **0.95** | **0.95** |
| Logistic Regression (balanced) | 71.1 % | 0.75 | 0.79 |

Best DT params: `criterion=entropy`, `max_depth=4`, `min_samples_leaf=3`.

## Cleaning rationale (important)

- **Exact duplicates** (same features + same label): Removed. They add no new information and risk data leakage.
- **True conflicts** (same features + different labels): Removed. These are real label noise; keeping them forces the model to learn contradictory signals.
- This directly addresses the valid peer-feedback point about conflicting labels.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

## Repository layout

```
maternal-health-risk-v4/
├── README.md
├── requirements.txt
├── data/
│   ├── Maternal Health Risk Data Set.csv   # original
│   ├── maternal_health_clean.csv           # cleaned (380 rows)
│   └── README.md
├── notebooks/analysis.ipynb
├── src/model.py
├── figures/          # 10 plots
└── report/summary.pdf
```

## Limitations
- Small final sample (n = 380) after rigorous cleaning
- Single geographic origin (rural Bangladesh)
- Important clinical covariates missing
- No external validation

**Disclaimer:** Educational risk-stratification exercise only – **not** a diagnostic tool.
