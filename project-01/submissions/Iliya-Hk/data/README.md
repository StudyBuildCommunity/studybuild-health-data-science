# Maternal Health Risk Data Set

## Overview
This dataset contains health parameters of pregnant women and their associated risk levels during pregnancy. It is designed for machine learning and statistical analysis to predict maternal health risks based on various physiological measurements.

## Dataset Description
The dataset includes **1,014 records** of maternal health indicators collected during pregnancy. Each record contains six health parameters and one target variable indicating the risk level.

## Features

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| **Age** | Age of the pregnant woman | Years | 10-70 |
| **SystolicBP** | Systolic Blood Pressure | mmHg | 70-160 |
| **DiastolicBP** | Diastolic Blood Pressure | mmHg | 49-100 |
| **BS** | Blood Sugar Level | mmol/L | 6.0-19.0 |
| **BodyTemp** | Body Temperature | °F | 98.0-103.0 |
| **HeartRate** | Heart Rate | bpm | 7-90 |
| **RiskLevel** | Risk Category (Target) | Categorical | low risk, mid risk, high risk |

## Target Variable Distribution
The target variable `RiskLevel` has three classes:
- **Low Risk**: Normal health parameters, minimal concern
- **Mid Risk**: Some parameters outside normal range, requires monitoring
- **High Risk**: Multiple parameters in concerning ranges, requires immediate attention

## Key Statistics

### Blood Pressure (mmHg)
- **SystolicBP**: Range 70-160, normal range typically 90-120
- **DiastolicBP**: Range 49-100, normal range typically 60-80

### Blood Sugar (mmol/L)
- **Range**: 6.0-19.0
- Normal fasting: < 5.6 mmol/L
- Values > 7.0 may indicate gestational diabetes risk

### Age
- **Range**: 10-70 years
- Most frequent age groups: 20-35 years

### Heart Rate
- **Range**: 7-90 bpm
- Typical range: 60-80 bpm

## Usage Examples

### Classification Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('Maternal Health Risk Data Set.csv')

# Separate features and target
X = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))