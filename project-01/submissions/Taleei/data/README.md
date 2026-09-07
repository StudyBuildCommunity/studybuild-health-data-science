# Data Directory

## Source Dataset

**File:** `Maternal Health Risk Data Set.csv`

- **Official source:** UCI Machine Learning Repository  
  https://archive.ics.uci.edu/dataset/863/pregnant+health+risk
- **DOI:** https://doi.org/10.24432/C5DP5D
- **License:** CC BY 4.0
- **Introductory paper:** Ahmed, M., Kashem, M.A., Rahman, M., Khatun, S. (2020). Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT). Lecture Notes in Electrical Engineering, vol 632.

## Dataset Description

Data collected from hospitals, community clinics and maternal-health care settings in rural Bangladesh through an IoT-based risk monitoring system.

| Variable    | Role   | Type        | Description                                      | Units  |
|-------------|--------|-------------|--------------------------------------------------|--------|
| Age         | Feature| Integer     | Maternal age in years                            | years  |
| SystolicBP  | Feature| Integer     | Upper value of blood pressure                    | mmHg   |
| DiastolicBP | Feature| Integer     | Lower value of blood pressure                    | mmHg   |
| BS          | Feature| Continuous  | Blood sugar (glucose) level                      | mmol/L |
| BodyTemp    | Feature| Continuous  | Body temperature                                 | °F     |
| HeartRate   | Feature| Integer     | Resting heart rate                               | bpm    |
| RiskLevel   | Target | Categorical | Predicted risk intensity (low / mid / high risk) | -      |

- Official UCI instances: 1,013  
- File used in this project: 1,014 rows (minor discrepancy documented)  
- Missing values: None

## Cleaning Applied (Version B – recommended)

1. Removed 562 exact duplicate rows.
2. Removed 2 physiologically implausible records with `HeartRate = 7`.
3. Final cleaned size: **451 unique records**.

Class distribution after cleaning:
- low risk: 233 (51.7%)
- high risk: 112 (24.8%)
- mid risk: 106 (23.5%)

The cleaned file is saved as `maternal_health_clean.csv` by the analysis notebook.
