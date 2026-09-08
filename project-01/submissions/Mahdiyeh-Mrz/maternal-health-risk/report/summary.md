# Maternal Health Risk Stratification — Clinical Analysis Summary

*Educational risk-stratification exercise — not a diagnostic tool.*

## Purpose

The clinical analytics team asked for an exploration of patterns in a maternal-health dataset
(age, blood pressure, blood sugar, body temperature, heart rate) and a simple baseline model to
separate Low, Mid and High Risk patients, with an emphasis on catching High Risk cases and
interpreting results cautiously.

## Data

1,014 records in the project CSV from rural-Bangladesh clinics (UCI Maternal Health Risk dataset,
CC BY 4.0). No missing values. Two rows with a physiologically implausible heart rate (7 bpm)
were removed as likely sensor errors. Duplicate rows (55%) were kept, since the dataset's coarse
measurement rounding means many patients legitimately share identical readings — this is
recorded as a limitation rather than "corrected." Final analytical dataset: 1,012 patients (Low 404 /
Mid 336 / High 272).

## Key Findings

1. **Blood Sugar is the strongest single marker of High Risk**, showing a clear, wide
   separation from Low/Mid Risk. Systolic and Diastolic BP are secondary markers. Age, Body
   Temperature and especially Heart Rate carry little independent signal.
2. **Low and Mid Risk overlap heavily** on every measured variable — this is the hardest
   distinction in the data, not High Risk vs. the rest.
3. A shallow **Decision Tree** (max depth 4) outperformed Logistic Regression on every metric
   tracked and was selected as the baseline model:

   | Metric | Value |
   |---|---|
   | Accuracy | 0.680 |
   | Macro F1 | 0.670 |
   | High-Risk Recall | **0.836** |
   | High-Risk Precision | 0.920 |

4. **32% of test cases were misclassified**, dominated by Mid Risk patients predicted as Low
   Risk (42 of 65 errors). **9 of 55** true High Risk patients were predicted as a lower risk
   class — the most clinically significant error type, since it represents a risk patient who
   would not be flagged.
5. Feature importance confirms Blood Sugar (50%) and Systolic BP (33%) as the model's primary
   decision drivers, with Diastolic BP, Age and Heart Rate contributing little to nothing.

## Interpretation for Decision-Makers

These results are consistent with Blood Sugar and Blood Pressure being clinically meaningful
markers of maternal risk in this population — but the dataset cannot establish that they *cause*
risk, only that they are statistically associated with the risk labels assigned in this specific
sample. The model correctly flags most High Risk patients (Recall 0.84) but still misses roughly
1 in 6, and confuses Low/Mid Risk often enough that it should be treated as a **screening aid
at most, never a diagnostic or triage tool**, and always alongside clinical judgment.

## Limitations

- Small, geographically narrow sample (rural Bangladesh) collected via low-precision IoT
  sensors — heavy value rounding produced many duplicate rows, reducing effective data
  diversity.
- Only six vital-sign features; no obstetric history, gestational age, labs, or comorbidities.
- Unknown provenance of the `RiskLevel` label, which may already be threshold-derived from the
  same BP/BS variables used as predictors.
- No external validation cohort and no clinician review of model outputs.

## What Would Be Needed Before Real Clinical Use

A larger and more diverse validated patient cohort; richer clinical features (history,
gestational stage, labs); prospective testing against clinician judgment; an explicit
cost-sensitive threshold that further prioritizes catching High Risk cases even at the cost of
more false alarms; and formal clinical and regulatory review.

*Full analysis, code, figures and reproducibility instructions: see the project* `README.md`
*and* `notebooks/analysis.ipynb`.
