"""
Maternal Health Risk Stratification
------------------------------------
End-to-end, reproducible pipeline:
  1. Load & clean the UCI Maternal Health Risk dataset
  2. Exploratory data analysis (EDA) with saved figures
  3. Train/test split (stratified, leakage-safe)
  4. Baseline models: Logistic Regression + shallow Decision Tree
  5. Class-wise evaluation (confusion matrix, Recall, F1 - High Risk focus)
  6. Error analysis
  7. Feature interpretation (coefficients / importances)

Run:
    python src/model.py

All figures are written to ../figures/ (relative to this file) and all
numeric summaries are printed to stdout, which is exactly what the
Jupyter notebook (notebooks/analysis.ipynb) also does cell-by-cell.

Author: Clinical Analytics Track - Project 01
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "Maternal_Health_Risk_Data_Set.csv")
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RISK_ORDER = ["low risk", "mid risk", "high risk"]
RISK_COLORS = {"low risk": "#4C9A6A", "mid risk": "#E0A93A", "high risk": "#C0392B"}


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved figures/{name}")


# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------------------
def load_and_clean(path=DATA_PATH, verbose=True):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["RiskLevel"] = df["RiskLevel"].str.strip().str.lower()

    n_raw = len(df)
    n_dupes = df.duplicated().sum()
    n_missing = df.isna().sum().sum()

    # Clinically implausible heart-rate values (HR = 7 bpm is not compatible
    # with a monitored, ambulatory patient - almost certainly a sensor/entry
    # error in the original IoT capture pipeline). We remove these rather
    # than impute, since we cannot know the true reading.
    implausible_hr = df[df["HeartRate"] < 30]
    df_clean = df[df["HeartRate"] >= 30].copy()

    # NOTE on duplicates: this dataset is captured from a small number of
    # low-precision IoT sensors (BP rounded to 5/10 mmHg, BS to 0.1 mmol/L,
    # BodyTemp to 1 F). Many patients legitimately share identical rounded
    # readings, so duplicate rows are NOT dropped by default - dropping them
    # would silently change the class balance and understate how common
    # these measurement combinations are. We flag duplicate counts in the
    # EDA instead of removing them.

    if verbose:
        print("=== Data Quality Summary ===")
        print(f"Raw rows: {n_raw}")
        print(f"Missing values (any column): {n_missing}")
        print(f"Exact duplicate rows: {n_dupes} ({n_dupes/n_raw:.1%} of raw rows)")
        print(f"Implausible HeartRate (<30 bpm) rows removed: {len(implausible_hr)}")
        if len(implausible_hr):
            print(implausible_hr.to_string(index=False))
        print(f"Rows after cleaning: {len(df_clean)}")
        print()
        print("Class balance after cleaning:")
        print(df_clean["RiskLevel"].value_counts().reindex(RISK_ORDER))
        print()

    return df_clean


# ---------------------------------------------------------------------------
# 2. EDA FIGURES
# ---------------------------------------------------------------------------
def make_eda_figures(df):
    print("=== Generating EDA figures ===")

    # Fig 1: RiskLevel distribution
    counts = df["RiskLevel"].value_counts().reindex(RISK_ORDER)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(
        [c.replace(" risk", "").title() for c in RISK_ORDER],
        counts.values,
        color=[RISK_COLORS[c] for c in RISK_ORDER],
    )
    for b, v in zip(bars, counts.values):
        plt.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", fontsize=10)
    plt.title("Distribution of Maternal Risk Level (n=%d)" % len(df))
    plt.xlabel("Risk Level")
    plt.ylabel("Number of Patients")
    savefig("01_risklevel_distribution.png")

    # Fig 2: Age by risk group (boxplot)
    plt.figure(figsize=(6, 4.5))
    data = [df.loc[df["RiskLevel"] == r, "Age"] for r in RISK_ORDER]
    bp = plt.boxplot(data, tick_labels=[r.replace(" risk", "").title() for r in RISK_ORDER],
                      patch_artist=True)
    for patch, r in zip(bp["boxes"], RISK_ORDER):
        patch.set_facecolor(RISK_COLORS[r])
        patch.set_alpha(0.7)
    plt.title("Maternal Age by Risk Group")
    plt.xlabel("Risk Level")
    plt.ylabel("Age (years)")
    savefig("02_age_by_risk.png")

    # Fig 3: Blood sugar by risk group (boxplot)
    plt.figure(figsize=(6, 4.5))
    data = [df.loc[df["RiskLevel"] == r, "BS"] for r in RISK_ORDER]
    bp = plt.boxplot(data, tick_labels=[r.replace(" risk", "").title() for r in RISK_ORDER],
                      patch_artist=True)
    for patch, r in zip(bp["boxes"], RISK_ORDER):
        patch.set_facecolor(RISK_COLORS[r])
        patch.set_alpha(0.7)
    plt.title("Blood Sugar (BS) by Risk Group")
    plt.xlabel("Risk Level")
    plt.ylabel("Blood Sugar (mmol/L)")
    savefig("03_bloodsugar_by_risk.png")

    # Fig 4: Systolic & Diastolic BP by risk group (grouped boxplots)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    for ax, col, title in zip(axes, ["SystolicBP", "DiastolicBP"],
                               ["Systolic BP", "Diastolic BP"]):
        data = [df.loc[df["RiskLevel"] == r, col] for r in RISK_ORDER]
        bp = ax.boxplot(data, tick_labels=[r.replace(" risk", "").title() for r in RISK_ORDER],
                         patch_artist=True)
        for patch, r in zip(bp["boxes"], RISK_ORDER):
            patch.set_facecolor(RISK_COLORS[r])
            patch.set_alpha(0.7)
        ax.set_title(f"{title} by Risk Group")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel(f"{title} (mmHg)")
    savefig("04_bloodpressure_by_risk.png")

    # Fig 5: Scatter - BS vs SystolicBP colored by risk
    plt.figure(figsize=(6.5, 5))
    for r in RISK_ORDER:
        sub = df[df["RiskLevel"] == r]
        plt.scatter(sub["BS"], sub["SystolicBP"], label=r.replace(" risk", "").title(),
                    alpha=0.5, s=25, color=RISK_COLORS[r])
    plt.title("Blood Sugar vs Systolic BP by Risk Group")
    plt.xlabel("Blood Sugar (mmol/L)")
    plt.ylabel("Systolic BP (mmHg)")
    plt.legend(title="Risk Level")
    savefig("05_scatter_bs_vs_systolicbp.png")

    # Fig 6: HeartRate by risk group (boxplot)
    plt.figure(figsize=(6, 4.5))
    data = [df.loc[df["RiskLevel"] == r, "HeartRate"] for r in RISK_ORDER]
    bp = plt.boxplot(data, tick_labels=[r.replace(" risk", "").title() for r in RISK_ORDER],
                      patch_artist=True)
    for patch, r in zip(bp["boxes"], RISK_ORDER):
        patch.set_facecolor(RISK_COLORS[r])
        patch.set_alpha(0.7)
    plt.title("Heart Rate by Risk Group")
    plt.xlabel("Risk Level")
    plt.ylabel("Heart Rate (bpm)")
    savefig("06_heartrate_by_risk.png")

    print()


# ---------------------------------------------------------------------------
# 3-5. MODEL TRAIN + EVALUATE
# ---------------------------------------------------------------------------
FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]


def prepare_split(df):
    X = df[FEATURES].copy()
    y = df["RiskLevel"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    # Fit scaler on TRAIN ONLY to avoid leakage; apply to both splits.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=FEATURES, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(name, y_test, y_pred, labels=RISK_ORDER):
    print(f"--- {name} ---")
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels)
    high_recall = recall_score(y_test, y_pred, labels=["high risk"], average="macro")
    print(f"Accuracy: {acc:.3f}  |  Macro F1: {macro_f1:.3f}  |  High-Risk Recall: {high_recall:.3f}")
    print(classification_report(y_test, y_pred, labels=labels, digits=3))
    return {"accuracy": acc, "macro_f1": macro_f1, "high_risk_recall": high_recall}


def plot_confusion(y_test, y_pred, name, filename):
    cm = confusion_matrix(y_test, y_pred, labels=RISK_ORDER)
    disp = ConfusionMatrixDisplay(cm, display_labels=[r.replace(" risk", "").title() for r in RISK_ORDER])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    savefig(filename)
    return cm


def run_models(df):
    print("=== Train/Test Split & Modeling ===")
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = prepare_split(df)
    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")
    print("Train class balance:\n", y_train.value_counts().reindex(RISK_ORDER))
    print("Test class balance:\n", y_test.value_counts().reindex(RISK_ORDER))
    print()

    # --- Logistic Regression (needs scaled features) ---
    logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    logreg.fit(X_train_s, y_train)
    y_pred_lr = logreg.predict(X_test_s)
    lr_metrics = evaluate_model("Logistic Regression", y_test, y_pred_lr)
    plot_confusion(y_test, y_pred_lr, "Logistic Regression", "07a_confusion_logreg.png")

    # --- Shallow Decision Tree (no scaling needed) ---
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    tree_metrics = evaluate_model("Decision Tree (depth=4)", y_test, y_pred_tree)
    cm_tree = plot_confusion(y_test, y_pred_tree, "Decision Tree", "07b_confusion_tree.png")

    # Choose final model = the one with better High-Risk recall, tie-break macro F1
    if tree_metrics["high_risk_recall"] >= lr_metrics["high_risk_recall"]:
        final_name, final_model, final_pred, final_metrics = "Decision Tree", tree, y_pred_tree, tree_metrics
        X_test_final = X_test
    else:
        final_name, final_model, final_pred, final_metrics = "Logistic Regression", logreg, y_pred_lr, lr_metrics
        X_test_final = X_test_s

    print(f"\n>>> Selected baseline model: {final_name} "
          f"(High-Risk Recall={final_metrics['high_risk_recall']:.3f}, "
          f"Macro F1={final_metrics['macro_f1']:.3f})\n")

    # --- Feature importance / coefficients ---
    print("=== Feature Interpretation ===")
    if final_name == "Decision Tree":
        importances = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print(importances)
        plt.figure(figsize=(6, 4.5))
        importances.sort_values().plot(kind="barh", color="#3E6E9E")
        plt.title(f"Feature Importance - {final_name}")
        plt.xlabel("Importance (Gini-based)")
        savefig("08_feature_importance.png")

        plt.figure(figsize=(16, 8))
        plot_tree(final_model, feature_names=FEATURES, class_names=[c.replace(" risk", "").title() for c in final_model.classes_],
                  filled=True, rounded=True, fontsize=8, max_depth=3)
        plt.title("Decision Tree Structure (max_depth=4)")
        savefig("09_decision_tree_structure.png")
    else:
        coef_df = pd.DataFrame(final_model.coef_, index=final_model.classes_, columns=FEATURES)
        print(coef_df.T)
        plt.figure(figsize=(7, 4.5))
        coef_df.T.plot(kind="barh", figsize=(7, 4.5))
        plt.title(f"Logistic Regression Coefficients by Class")
        plt.xlabel("Coefficient (standardized features)")
        savefig("08_feature_importance.png")

    # --- Error analysis ---
    print("\n=== Error Analysis ===")
    X_test_readable = X_test.copy()
    X_test_readable["true"] = y_test.values
    X_test_readable["pred"] = final_pred
    errors = X_test_readable[X_test_readable["true"] != X_test_readable["pred"]]
    print(f"Total test errors: {len(errors)} / {len(X_test_readable)} "
          f"({len(errors)/len(X_test_readable):.1%})")
    error_pairs = errors.groupby(["true", "pred"]).size().sort_values(ascending=False)
    print("\nMost common misclassification pairs (true -> predicted):")
    print(error_pairs)

    high_risk_missed = errors[errors["true"] == "high risk"]
    print(f"\nHigh-Risk cases MISSED (predicted as lower risk): {len(high_risk_missed)}")
    if len(high_risk_missed):
        print(high_risk_missed[FEATURES + ["true", "pred"]].to_string(index=False))

    return {
        "logreg": lr_metrics,
        "tree": tree_metrics,
        "final_model_name": final_name,
        "final_metrics": final_metrics,
        "n_errors": len(errors),
        "error_pairs": error_pairs,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df = load_and_clean()
    make_eda_figures(df)
    results = run_models(df)
    print("\n=== DONE ===")
    print(f"Figures written to: {FIG_DIR}")
    return df, results


if __name__ == "__main__":
    main()
