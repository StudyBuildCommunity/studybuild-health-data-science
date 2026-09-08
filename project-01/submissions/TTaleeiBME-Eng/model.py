"""
Optional helper module for Maternal Health Risk Stratification.
Contains reusable functions for training and evaluating the baseline models.
"""

from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"
RISK_ORDER = ["high risk", "low risk", "mid risk"]


def load_and_prepare(path: str) -> pd.DataFrame:
    """Load the cleaned CSV and ensure consistent RiskLevel labels."""
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].str.strip().str.lower()
    return df


def train_test_split_stratified(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
) -> Tuple:
    """Stratified train/test split."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def fit_logistic_regression(X_train, y_train):
    """Fit a scaled Logistic Regression baseline."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def fit_decision_tree(X_train, y_train, max_depth: int = 4):
    """Fit a shallow Decision Tree baseline."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_pred, labels=None):
    """Print classification report and return confusion matrix."""
    if labels is None:
        labels = RISK_ORDER
    print(classification_report(y_true, y_pred, labels=labels))
    return confusion_matrix(y_true, y_pred, labels=labels)
