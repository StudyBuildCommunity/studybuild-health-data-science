# METHODOLOGY — Methods Used, and Why

This document is a full declaration of every method, technique, and tool used in this project,
what it was used for, and the reasoning behind choosing it. It complements `README.md` (which
focuses on findings) by focusing on **process and justification**.

---

## 1. Data Loading & Encoding

| Method | Used for | Why chosen |
|---|---|---|
| `pandas.read_csv(..., encoding="utf-8-sig")` | Loading the raw CSV | The source file begins with a UTF-8 BOM (byte-order mark); reading with plain `utf-8` leaves a hidden `\ufeff` character stuck to the first column name (`Age`), which silently breaks every later reference to that column. `utf-8-sig` strips it automatically. |
| `.str.strip().str.lower()` on `RiskLevel` | Normalizing the target labels | Guards against stray whitespace/case differences (e.g. `"High risk "` vs `"high risk"`) that would otherwise be treated as different classes. |

## 2. Data Quality & Cleaning

| Method | Used for | Why chosen |
|---|---|---|
| `isna().sum()` | Missing-value audit | Standard first check before any analysis; confirmed 0 missing values (matches UCI's documentation). |
| `duplicated().sum()` | Duplicate-row audit | Found 562 exact duplicate rows (55.4%). **Decision: kept, not dropped.** The dataset's six features are captured by low-precision IoT sensors (BP rounded to 5–10 mmHg, BS to 0.1 mmol/L, BodyTemp to whole/half degrees), so many genuinely different patients can share identical rounded readings by chance. Silently dropping "duplicates" would remove real patients and distort the class balance without evidence they are actually the same record. Instead, this is documented as a **limitation** (lower effective information content than the raw row count suggests). |
| Range filter (`HeartRate < 30`) | Removing implausible physiological values | A resting heart rate of 7 bpm is not compatible with a monitored, ambulatory patient — this is treated as a sensor/entry error, not a real measurement, and removed rather than imputed (we cannot know the true value, so guessing one would fabricate data). Only 2 of 1,014 rows were affected. |
| No imputation, no outlier-capping beyond the above | Keeping the pipeline minimal and transparent | The project brief explicitly asks to "keep preprocessing minimal and explain every step" — every other unusual-looking value (e.g. very young/old ages) is *reported*, not silently altered, since we cannot verify whether it is an error or a genuine record without more context. |

## 3. Exploratory Data Analysis (EDA)

| Method | Used for | Why chosen |
|---|---|---|
| Grouped descriptive statistics (`groupby("RiskLevel")`, mean/median/std) | Comparing Age, BP, BS, BodyTemp, HeartRate across risk groups (Q2) | Simple, interpretable, and appropriate for a small (~1,000-row) dataset — no need for more complex statistical machinery. |
| **Boxplots** (Age, BS, SystolicBP, DiastolicBP, HeartRate by group) | Visualizing distribution *and* spread per risk group, robust to the dataset's many repeated/rounded values | Boxplots show median, IQR, and outliers simultaneously and handle the heavy value-repetition in this dataset better than, e.g., overlapping histograms would. |
| **Scatter plot** (BS vs SystolicBP, colored by risk) | Visualizing how two of the strongest variables jointly relate to risk (Q3) | A 2-D view reveals whether combining two variables separates High Risk more cleanly than either alone — directly motivates the "association, not causation" discussion. |
| Bar chart (RiskLevel counts) | Showing class balance (Q1) | Needed up front because class imbalance directly shapes which evaluation metrics matter later (§5). |

## 4. Machine Learning — Model Choice

| Method | Used for | Why chosen |
|---|---|---|
| **Stratified train/test split** (80/20, `stratify=y`, `random_state=42`) | Creating a held-out evaluation set | Stratification keeps the same Low/Mid/High proportions in both splits — important with an imbalanced target so the small test set for High Risk (55 patients) is not accidentally under- or over-represented by chance. The fixed seed makes every run reproducible. |
| **StandardScaler, fit on the training set only** | Feature scaling for Logistic Regression | Logistic Regression's coefficients are only comparable across features when inputs are on the same scale; the scaler is fit on `X_train` alone and then applied to `X_test` to **prevent data leakage** — test-set statistics never influence any transformation used at prediction time. |
| **Logistic Regression** (multinomial, `max_iter=2000`) | Baseline linear classifier | Simple, fast, and directly interpretable via per-class coefficients — a natural first baseline the project brief explicitly allows. |
| **Decision Tree** (`max_depth=4`, `min_samples_leaf=15`) | Baseline non-linear classifier | Also explicitly allowed by the brief; deliberately kept **shallow** (depth 4, minimum 15 samples per leaf) so it stays interpretable and resists overfitting the small dataset, rather than jumping to an ensemble (which the brief asks to avoid for this project). |
| Model selection by **High-Risk Recall first, macro F1 as tie-break** | Choosing the final baseline between the two candidates | Reflects the clinical priority stated in the brief: missing a High Risk patient (false negative) is more costly than confusing Low/Mid Risk, so the model that catches more true High Risk cases is preferred even if it is not the single highest-accuracy model. |

## 5. Evaluation

| Method | Used for | Why chosen |
|---|---|---|
| **Confusion matrix** (both models) | Seeing exactly which classes get confused with which | Accuracy alone hides *where* a model fails; the confusion matrix is the only view that shows the actual Low/Mid/High error pattern (Q6). |
| **Class-wise Precision / Recall / F1** (`classification_report`) | Judging performance per class, not just overall | With an imbalanced target, a model could reach seemingly-reasonable overall accuracy while performing poorly on the smaller High Risk class — class-wise metrics make that visible, per the brief's explicit requirement (§5, "Do not report Accuracy alone"). |
| **High-Risk Recall as the headline metric** | Prioritizing the clinically costliest error type | Recall = (High Risk patients correctly caught) / (all true High Risk patients); this is the metric that answers "how many at-risk patients would this model miss?", which matters more here than overall accuracy. |
| Manual error-slicing (`results[results.true != results.pred]`, grouped by `(true, pred)`) | Root-causing *which* pairs of classes are hardest, and listing exactly which High-Risk patients were missed | Goes one level deeper than the confusion matrix — inspecting the actual feature values of missed High-Risk cases (Q6) to look for a pattern (none of the missed cases had extreme values on every variable at once, consistent with the Low/Mid/High overlap seen in EDA). |

## 6. Feature Interpretation

| Method | Used for | Why chosen |
|---|---|---|
| **Gini-based `feature_importances_`** (Decision Tree) | Ranking which variables the chosen model relies on most (Q7) | Native, unit-free output of the tree itself — directly reflects which features actually drove its splits, unlike a raw mean-difference (see the SQL caveat in §7 below). |
| `plot_tree` visualization | Showing the literal decision rules learned | Makes the model fully transparent — every split threshold is visible and explainable to a non-technical reader, supporting the "interpretable" goal of the project. |
| Explicit "importance ≠ causation" framing throughout | Preventing over-claiming | The `RiskLevel` label itself was likely assigned partly using BP/BS thresholds by the original clinicians, which can inflate the apparent association between those variables and the label — stated explicitly rather than left implicit. |

## 7. SQL Analysis (`sql/`)

Added as a complementary, dependency-light way to reproduce the core EDA findings using pure
SQL instead of pandas — useful for stakeholders who query the data directly rather than run
Python.

| Component | Purpose |
|---|---|
| `data/maternal_health.db` (SQLite) | The **cleaned** dataset (post implausible-value removal, same 1,012 rows used by the Python pipeline) loaded into a single `maternal_health` table, so SQL and Python analyses always start from the same cleaned data. |
| `sql/queries.sql` | Nine documented queries, each with a `-- Purpose:` comment, covering: class distribution (Q1a), overall summary stats (Q1b), a data-quality re-check (Q1c), per-group mean/median-proxy/spread (Q2, Q2b), threshold-based separation checks for High Risk (Q3a, Q3b), a cross-variable "separating power" ranking (Q3c, with an explicit caveat that it is scale-dependent — see below), and an age-band cross-tab sanity check. |
| `sql/run_queries.py` | Executes every query in `queries.sql` against the SQLite database and writes human-readable results to `sql/sql_results.md`, so the SQL layer is reproducible with one command and reviewable without a SQL client. |
| Window function (`ROW_NUMBER() OVER (PARTITION BY ...)`) | Used in Q2 as the basis for an ordered, per-group ranking (SQLite has no native `MEDIAN()`). |
| Manual standard-deviation formula (`SQRT(AVG(x*x) - AVG(x)*AVG(x))`) | Q2b — SQLite has no built-in `STDDEV()`, so population standard deviation is computed directly from its definition. |
| `CASE WHEN` age-banding | Q_extra — buckets continuous `Age` into clinically meaningful bands directly in SQL, without needing pandas. |
| **Caveat documented in `queries.sql` itself (Q3c)** | The raw "High Risk avg − Low Risk avg" SQL ranking puts SystolicBP first only because mmHg values are numerically larger than mmol/L values — it is **not** unit-comparable across variables. The unit-free, model-based ranking in `figures/08_feature_importance.png` (Python) is the authoritative feature ranking; the SQL version is included for transparency and to show *direction* of each shift, with this limitation stated explicitly rather than presented as a contradiction. |

**Why SQLite specifically:** zero setup (single file, no server), ships with Python's standard
library (`sqlite3`), and is enough to demonstrate real `GROUP BY`, `CASE WHEN`, `WITH` (CTEs),
`UNION ALL`, and window-function SQL on this dataset without adding infrastructure the project
doesn't need.

## 8. Reproducibility

| Method | Used for | Why chosen |
|---|---|---|
| Fixed `random_state=42` everywhere a random process occurs (train/test split, Decision Tree) | Making every run produce identical numbers | Required by the project brief ("include a fixed random seed when appropriate") and essential for anyone re-running the notebook/script to get the same figures and metrics reported here. |
| Two equivalent entry points (`src/model.py` script and `notebooks/analysis.ipynb`) | Letting the analysis be run either way | The notebook is for narrative, cell-by-cell exploration; the script is for automation/CI or quick re-runs — both implement the identical pipeline so their outputs match exactly. |
| `requirements.txt` with minimum versions | Pinning the environment | Ensures the same library versions (pandas, scikit-learn, matplotlib) are available to anyone re-running the project, avoiding "works on my machine" issues. |

---

## Summary Table — Method → Purpose (Quick Reference)

| Category | Method | One-line purpose |
|---|---|---|
| Loading | `utf-8-sig` CSV read | Strip hidden BOM character |
| Cleaning | Duplicate audit (kept) | Document sensor-rounding, avoid distorting class balance |
| Cleaning | Implausible-value filter | Remove 2 sensor-error rows (HeartRate=7) |
| EDA | Grouped stats + boxplots | Compare clinical variables across risk groups |
| EDA | Scatter plot | Show joint separation of two key variables |
| SQL | SQLite + `queries.sql` | Reproduce core EDA in pure SQL for non-Python stakeholders |
| ML | Stratified split + scaler-on-train-only | Fair evaluation, zero data leakage |
| ML | Logistic Regression | Simple, interpretable linear baseline |
| ML | Shallow Decision Tree | Simple, interpretable non-linear baseline |
| Evaluation | Confusion matrix + class-wise F1/Recall | Judge performance per class, not just overall |
| Evaluation | High-Risk-Recall-first model selection | Reflect real clinical cost of a missed High-Risk case |
| Interpretation | Gini feature importance + tree plot | Show and explain exactly what drives predictions |
| Interpretation | Explicit causation caveats | Prevent over-claiming from an observational dataset |
| Reproducibility | Fixed seed, pinned requirements, dual entry points | Anyone can re-run this and get the same results |
