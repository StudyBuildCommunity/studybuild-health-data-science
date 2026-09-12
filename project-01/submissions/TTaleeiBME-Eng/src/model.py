"""
Helper module – Maternal Health Risk Stratification (v3)
- 3-class classification (Low / Mid / High)
- sklearn Pipeline (no leakage)
- class_weight='balanced' to improve High-risk Recall
"""

from typing import Tuple, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"
RISK_ORDER = ["low risk", "mid risk", "high risk"]


def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].str.strip().str.lower().astype("category")
    df[TARGET] = df[TARGET].cat.reorder_categories(RISK_ORDER, ordered=True)
    return df


def stratified_split(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def make_logistic_pipeline(random_state: int = 42) -> Pipeline:
    """Logistic Regression with StandardScaler + balanced class weights."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=random_state,
            max_iter=2000,
            class_weight="balanced",
        )),
    ])


def make_tree_pipeline(max_depth: int = 5, random_state: int = 42) -> Pipeline:
    """Decision Tree with balanced class weights."""
    return Pipeline([
        ("clf", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])


def tune_tree(X_train, y_train, random_state: int = 42) -> Pipeline:
    """
    Limited GridSearch on Decision Tree, optimising macro-recall
    (improves High-risk Recall while staying 3-class).
    """
    base = Pipeline([
        ("clf", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        )),
    ])
    param_grid = {
        "clf__max_depth": [3, 4, 5, 6],
        "clf__min_samples_leaf": [3, 5, 8],
        "clf__criterion": ["gini", "entropy"],
    }
    grid = GridSearchCV(
        base,
        param_grid,
        cv=5,
        scoring="recall_macro",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def evaluate(y_true, y_pred, labels=None):
    if labels is None:
        labels = RISK_ORDER
    print(classification_report(y_true, y_pred, labels=labels, digits=3))
    return confusion_matrix(y_true, y_pred, labels=labels)
