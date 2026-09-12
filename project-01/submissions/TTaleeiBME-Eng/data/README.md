# Data

## Original source
- UCI Maternal Health Risk Data Set
- https://archive.ics.uci.edu/dataset/863/pregnant+health+risk
- DOI: https://doi.org/10.24432/C5DP5D
- License: CC BY 4.0

## Cleaning steps applied (v4)

1. **Implausible values**: Removed 2 records with HeartRate < 40 (HeartRate = 7).
2. **True label conflicts (label noise)**: Removed all rows belonging to feature combinations that appear with more than one different RiskLevel (215 rows across 35 unique feature groups). These are genuine contradictions and were removed to avoid teaching the model conflicting signals.
3. **Exact duplicates**: Removed remaining exact duplicate rows (identical features + identical RiskLevel).

**Final clean size: 380 records**

| RiskLevel  | Count | Proportion |
|------------|-------|------------|
| low risk   | 203   | 53.4 %     |
| high risk  | 101   | 26.6 %     |
| mid risk   | 76    | 20.0 %     |

Files:
- `Maternal Health Risk Data Set.csv` – original raw file
- `maternal_health_clean.csv` – fully cleaned version used for modelling
