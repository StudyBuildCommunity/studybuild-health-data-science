
# Maternal Health Risk - Complete Workflow Summary

## 1. Dataset Understanding
- **Source**: UCI Maternal Health Risk Dataset
- **Samples**: 414 after cleaning (from 1014 original)
- **Features**: 6 clinical measurements
- **Target**: RiskLevel (low, mid, high)

## 2. Quality Checks
- [OK] No missing values
- [OK] 562 duplicates removed
- [OK] Clinical validation applied (Age: 10-70, HR: 40-120)

## 3. Focused EDA
Key findings:
- Risk distribution: Low (51.7%), Mid (23.4%), High (24.9%)
- Blood sugar and systolic BP are the strongest discriminators
- Body temperature shows no variation

## 4. Compare Risk Groups
| Feature | Low Risk | Mid Risk | High Risk |
|---------|----------|----------|-----------|
| BS (mmol/L) | 6.9 | 7.5 | 10.2 |
| Systolic BP | 110 | 120 | 138 |
| Age | 28 | 31 | 34 |

## 5. Model Selection
- **Model**: Logistic Regression
- **Why**: Interpretable coefficients, handles class imbalance
- **Class Weights**: Balanced

## 6. Train/Test Split
- Train: 70% (290 samples)
- Test: 30% (124 samples)
- Stratified: Yes

## 7. Model Performance
- Accuracy: 0.75
- Recall (High Risk): 0.78
- Precision (High Risk): 0.72
- F1 (High Risk): 0.75

## 8. Error Analysis
- Most confusion: Mid <-> High risk
- False negatives (High Risk): 22% missed
- False positives (High Risk): 28% incorrectly flagged

## 9. Feature Interpretation
Top predictors (coefficients):
1. Blood Sugar: +2.15
2. Systolic BP: +1.87
3. Age: +0.45

## 10. Limitations
- Small sample size
- Geographic limitation
- Missing clinical variables
- No external validation

## 11. Conclusion
This analysis demonstrates that blood sugar and blood pressure are key indicators of high-risk pregnancy in this population. However, the model should be used as a screening tool only and requires clinical validation before deployment.
