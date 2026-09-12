# Data Directory

## Source
- UCI Maternal Health Risk: https://archive.ics.uci.edu/dataset/863/pregnant+health+risk
- DOI: 10.24432/C5DP5D · License: CC BY 4.0
- Paper: Ahmed et al. (2020)

## Cleaning (Version B)
1. Removed 562 exact duplicate rows (no conflicting labels).
2. Removed 2 records with HeartRate = 7.
3. Converted RiskLevel to ordered categorical.
4. Final size: **451 unique records**.

| Risk Level | Count | Proportion |
|------------|-------|------------|
| low risk   | 233   | 51.7 %     |
| high risk  | 112   | 24.8 %     |
| mid risk   | 106   | 23.5 %     |

Cleaned file: `maternal_health_clean.csv`
