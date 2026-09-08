# Dataset: UCI Maternal Health Risk

- **Source:** UCI Machine Learning Repository — Maternal Health Risk Data Set
- **Official page:** https://archive.ics.uci.edu/dataset/863/pregnant+health+risk
- **DOI:** https://doi.org/10.24432/C5DP5D
- **License:** CC BY 4.0
- **Introductory paper:** *Review and Analysis of Risk Factor of Maternal Health in Remote Area
  Using the Internet of Things (IoT)*, 2020.
- **Collection context:** hospitals, community clinics, and maternal-health care settings in
  rural Bangladesh, collected via an IoT-based risk-monitoring system.
- **Instances / features:** 1,013 instances (1,014 rows in this CSV export, including header),
  6 input features, 1 categorical target (`RiskLevel`). UCI reports no missing values.

## Variables

| Column        | Description                              |
|---------------|-------------------------------------------|
| `Age`         | Maternal age (years)                       |
| `SystolicBP`  | Systolic blood pressure (mmHg)             |
| `DiastolicBP` | Diastolic blood pressure (mmHg)            |
| `BS`          | Blood sugar (mmol/L)                       |
| `BodyTemp`    | Body temperature (°F)                      |
| `HeartRate`   | Heart rate (bpm)                           |
| `RiskLevel`   | Categorical maternal risk level (target): `low risk` / `mid risk` / `high risk` |

## Citation

If you use this dataset, please cite the original UCI source:

> Ahmed, M. (2020). Maternal Health Risk [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5DP5D

## File in this repo

`Maternal_Health_Risk_Data_Set.csv` — the **final cleaned analysis dataset (1,012 observations)**.
`Maternal_Health_Risk_Data_Set_raw_1014.csv` — the supplied raw export (1,014 observations), retained unchanged for reproducibility.
Two observations with `HeartRate = 7` bpm are excluded from the final analysis dataset. Exact duplicates are retained.
