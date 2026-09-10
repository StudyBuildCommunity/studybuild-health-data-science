# Maternal Health Risk Stratification — Interpretable Machine Learning with Python

Educational baseline classifier that stratifies maternal risk (Low / Mid / High) from vital-sign measurements. **This is not a diagnostic tool** — it's a careful, interpretable baseline meant to surface patterns and limitations, not to guide real clinical decisions.

## Clinical problem

A maternal-health analytics team wants to understand which routinely-collected vitals (age, blood pressure, blood sugar, body temperature, heart rate) are associated with maternal risk level, and whether a simple model can flag High Risk patients early enough to be clinically useful. The priority is **not missing High Risk cases** (false negatives are the costliest error here), not maximizing raw accuracy.

## Dataset source and license

- **Source:** UCI Machine Learning Repository — [Maternal Health Risk Data Set](https://archive.ics.uci.edu/dataset/863/pregnant+health+risk)
- **License:** CC BY 4.0
- **Citation:** Ahmed, M., Kashem, M. A., Rahman, M., Khatun, S. (2020). *Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT)*. Lecture Notes in Electrical Engineering, vol 632.
- Collected via an IoT-based monitoring system from hospitals, community clinics, and maternal health care centers in rural Bangladesh. 1,014 records, 6 features, 1 categorical target. UCI reports no missing values.

## Variable description

| Variable    | Description                             | Units  |
| ----------- | --------------------------------------- | ------ |
| Age         | Maternal age                            | years  |
| SystolicBP  | Systolic blood pressure                 | mmHg   |
| DiastolicBP | Diastolic blood pressure                | mmHg   |
| BS          | Blood sugar                             | mmol/L |
| BodyTemp    | Body temperature                        | °F     |
| HeartRate   | Heart rate                              | bpm    |
| RiskLevel   | Target: low risk / mid risk / high risk | —      |

## Data-quality checks

**No missing values** — confirmed with `df.isnull().sum()` (matches the UCI documentation).

**One genuine sensor error, two duplicate data-quality issues:**

1. **Implausible value:** two rows had `HeartRate = 7`, which is not compatible with a live reading — a clear sensor/entry error. All other extreme-looking values (e.g. `Age` up to 70, `BodyTemp` up to 103°F) were **kept, not clipped**: this dataset was deliberately collected to include both teenage pregnancies and advanced maternal age, so a wide age range is real clinical signal, not noise. Only `HeartRate` was corrected (invalid values → NaN → median-imputed); an earlier version of this pipeline that also "corrected" wide-but-real `Age` values toward the population median measurably destroyed the model's ability to detect High Risk cases, since older age is one of the clearest markers of High Risk in this population.

2. **Duplicate records — investigated, not assumed:** `df.duplicated()` showed 866 of 1,014 rows (85%) belong to duplicate groups. Before dropping anything, we tested whether this was a coincidence of coarse-grained features (BP rounded to steps of 10, BodyTemp almost always 98.0°F, few distinct BS values) rather than true duplication:
   
   | RiskLevel | Total rows | In duplicate groups | Unique (singleton) rows      |
   | --------- | ---------- | ------------------- | ---------------------------- |
   | low risk  | 406        | 292                 | 114 (77% of all unique rows) |
   | mid risk  | 336        | 336                 | 0 (0%)                       |
   | high risk | 272        | 238                 | 34 (23%)                     |
   
   If duplication were coincidental, the **majority** class (low risk, 40% of the data) should show the *most* duplication — instead it shows the *least*, while mid risk is 100% duplicated with zero unique records. That's a systematic artifact, not chance collision — most likely repeated/merged monitoring records, concentrated in mid/high-risk patients (plausibly from more frequent IoT monitoring of higher-risk cases). We also found **21 feature combinations mapped to more than one RiskLevel label** — i.e. identical clinical profiles were given different risk labels in the source data, indicating inherent label noise that caps achievable performance regardless of model choice.
   
   **Decision:** deduplicate (`drop_duplicates()`) **before** the train/test split, to avoid near-identical rows leaking across the split and to stop the artificial over-representation of mid/high risk from biasing the model. This is a deliberate data-quality call, not a routine cleaning step — see **Limitations** below for its cost.
   
   Effect: 1,014 → 562 duplicate rows removed → **452 rows remain**, and the class balance shifts from low 40% / mid 33% / high 27% to **low 52% / high 25% / mid 23%**. Mid risk lost the most (336 → ~106 unique cases, a ~68% reduction), which is the direct cause of the weak Mid Risk recall reported below — not a modeling failure, but a consequence of how little independent mid-risk data survives deduplication. We accepted this trade-off deliberately: keeping duplicates gave *apparently* better Mid/High metrics in a side experiment, but that improvement is attributable to leakage (near-identical rows on both sides of the split) and class-count inflation, not genuine predictive signal, so we rejected it despite the metric drop.

## EDA findings

Group means by risk level show a consistent pattern — High Risk patients skew older with higher blood pressure, blood sugar and temperature than Low Risk:

| RiskLevel | Age  | SystolicBP | DiastolicBP | BS   | BodyTemp | HeartRate |
| --------- | ---- | ---------- | ----------- | ---- | -------- | --------- |
| high risk | 33.7 | 119.5      | 81.5        | 11.2 | 99.2     | 76.5      |
| mid risk  | 28.5 | 112.4      | 74.9        | 7.9  | 98.9     | 73.9      |
| low risk  | 27.3 | 105.4      | 72.7        | 7.2  | 98.4     | 73.1      |

Boxplots confirm **BS, SystolicBP and DiastolicBP show the clearest separation** for High Risk (boxes shifted noticeably higher with less overlap), while Age, BodyTemp and HeartRate overlap heavily across groups and contribute less on their own. A BS-vs-Age scatter (colored by risk) shows this concretely: BS above roughly 11–12 mmol/L is almost exclusively High Risk regardless of age, while Age alone does not cleanly separate the classes (there are elderly Low Risk patients and young High Risk ones).

## Variables most associated with High Risk

Pearson correlation with a High-Risk-vs-rest indicator:

| Variable    | Correlation |
| ----------- | ----------- |
| BS          | 0.573       |
| SystolicBP  | 0.287       |
| DiastolicBP | 0.256       |
| BodyTemp    | 0.218       |
| Age         | 0.189       |
| HeartRate   | 0.182       |

**Blood sugar is by far the strongest single-variable association with High Risk.** This is an observed association in this sample, not a confirmed causal risk factor — it doesn't rule out confounding with unmeasured factors, and the dataset's small size and single-region origin limit how far this generalizes.

## Model choice and justification

Two simple, interpretable baselines were built and compared, per the project brief (no ensembles):

- **Logistic Regression**, grid-searched over solver (`saga`/`lbfgs`), regularization (`C`, L1/L2/ElasticNet), and `class_weight`, 5-fold CV. A degree-2 polynomial-features version was also tried to let the linear model capture simple interactions (e.g. BP × BodyTemp) — it outperformed the plain linear model, especially on High Risk (Recall 0.78 vs 0.67), so it was kept as the reported "Logistic Regression" model below.
- **Decision Tree** (entropy/gini), grid-searched over `max_depth`, `min_samples_leaf`, `min_samples_split`, 5-fold CV. Best params: `max_depth=5, min_samples_leaf=5, min_samples_split=16` (best CV accuracy 0.74).

Both use a fixed random seed (42) for the split and the tree, and preprocessing (scaling / polynomial expansion) is fit only inside each CV fold via `Pipeline` + `GridSearchCV`, avoiding leakage from test data into preprocessing.

## Evaluation metrics

Accuracy alone is not reported as the headline metric — class-wise Recall/F1 matter more here, especially for High Risk:

| Model               | Class     | Precision | Recall | F1   |
| ------------------- | --------- | --------- | ------ | ---- |
| Logistic Regression | low risk  | 0.66      | 0.91   | 0.77 |
| Logistic Regression | mid risk  | 0.50      | 0.19   | 0.28 |
| Logistic Regression | high risk | 0.88      | 0.78   | 0.82 |
| Decision Tree       | low risk  | 0.69      | 0.98   | 0.81 |
| Decision Tree       | mid risk  | 0.42      | 0.19   | 0.26 |
| Decision Tree       | high risk | 0.83      | 0.56   | 0.67 |

Overall accuracy: LR 0.68, DT 0.67 — nearly identical. But **High Risk Recall diverges sharply: LR misses 4/18 true High Risk test cases (22%), the Decision Tree misses 8/18 (44%)**. Since a missed High Risk case is the clinically costliest error in this project, **Logistic Regression is the preferred model**, despite the two models being close on accuracy. Both models share the same weakness: **Mid Risk Recall is poor (0.19 for both)** — the direct consequence of Mid Risk having the fewest independent training examples after deduplication (see Data-quality checks).



**For comparison — the same two models trained on the raw, non-deduplicated data (1,014 rows, 203-row test set):**

| Model                          | Class     | Precision | Recall | F1   |
| ------------------------------ | --------- | --------- | ------ | ---- |
| Logistic Regression (no dedup) | low risk  | 0.63      | 0.82   | 0.71 |
| Logistic Regression (no dedup) | mid risk  | 0.71      | 0.39   | 0.51 |
| Logistic Regression (no dedup) | high risk | 0.75      | 0.89   | 0.82 |
| Decision Tree (no dedup)       | low risk  | 0.82      | 0.57   | 0.68 |
| Decision Tree (no dedup)       | mid risk  | 0.61      | 0.83   | 0.70 |
| Decision Tree (no dedup)       | high risk | 0.86      | 0.81   | 0.84 |

Overall accuracy without dedup: LR 0.68, DT 0.72.

**Why High Risk looks dramatically better without deduplication — and why that comparison isn't trustworthy.** On the surface this looks like a large regression: Decision Tree High Risk recall drops from 0.81 (no dedup) to 0.56 (dedup), F1 from 0.84 to 0.67; Logistic Regression's High Risk recall drops from 0.89 to 0.78. Two things are happening, and only one of them is real:

1. **A real, honest effect:** deduplication removed ~59% of High Risk rows (272 → ~112 unique) and shrank the test set from 203 to 91 rows (18 High Risk cases instead of 47), so with dedup each misclassified case swings recall by ~5-6 points instead of ~2 — part of the gap is simply a smaller, higher-variance test set, not a systematic model failure.
2. **A leakage effect, and it's the larger one.** Mid Risk is the tell: its F1 jumps from 0.26-0.28 (dedup) to 0.51-0.70 (no dedup) — a *bigger* jump than High Risk gets — and Mid Risk is exactly the class that was 100% duplicated (see Data-quality checks). Without deduplication, near-identical readings for the same duplicated record land on *both* sides of the train/test split, so the model is partly evaluated on rows it already memorized. High Risk (87.5% duplicated) benefits from the same leakage, just less severely than Mid Risk. Low Risk — the *least*-duplicated class (72%) — shows the smallest gain, and for the Decision Tree it actually gets *worse* without dedup (recall 0.98 → 0.57). That pattern (inflation size tracking each class's duplication rate: Mid 100% → biggest jump, High 87.5% → sizeable jump, Low 72% → smallest/negative jump) is the signature of leakage, not of a genuinely better model.

**Conclusion:** the no-dedup numbers are not a fair baseline to compare against — they're inflated by the same duplicate-record leakage already documented in Data-quality checks. The deduplicated numbers above are the ones we trust: Logistic Regression's High Risk recall of 0.78 is the honest figure, and Mid Risk's poor recall (0.19) is a real, unresolved data-scarcity problem — not something deduplication caused, but something deduplication *revealed* by removing the artificial boost duplicate records were giving it.



## Error analysis

Misclassification pairs on the test set:

| Confused pair | Logistic Regression | Decision Tree |
| ------------- | ------------------- | ------------- |
| low / mid     | 23                  | 20            |
| high / mid    | 3                   | 8             |
| high / low    | 3                   | 2             |

Errors concentrate heavily on the **Low/Mid boundary** for both models — the clinically safer mistake, since it doesn't involve missing a High Risk patient. The Decision Tree confuses High/Mid substantially more than Logistic Regression (8 vs 3), which is the direct source of its worse High Risk recall. Looking at the Mid/High confusion cases specifically: the 3 cases LR confuses have notably elevated mean feature values (Age 54.7, SystolicBP 123, BS 14.3) — genuinely borderline, severe-looking cases — while the Decision Tree's 8 confused cases look more moderate (Age 36.6, SystolicBP 112.9, BS 11.6), suggesting the tree is more prone to misjudging cases that aren't clearly extreme in either direction.

## Feature interpretation

Both models agree on the top drivers, which cross-validates the EDA/correlation findings above:

- **Decision Tree feature importances:** BS (~0.55) and SystolicBP (~0.31) dominate; BodyTemp (~0.12) and Age (~0.02) contribute a little; DiastolicBP and HeartRate contribute almost nothing.
- **Logistic Regression (degree-2 polynomial) coefficients:** the largest terms are `SystolicBP²`, `BS²`, `BS`, and interaction terms like `SystolicBP × DiastolicBP` and `SystolicBP × BodyTemp` — for High Risk specifically, the top terms are `SystolicBP²`, `SystolicBP × BodyTemp`, `BS`, and `Age`.

**Both models converge on blood sugar and systolic blood pressure as the dominant signals**, with the linear model additionally surfacing nonlinear/interaction effects between blood pressure and temperature. These are contributions to *these models'* predictions on *this sample* — not confirmed clinical risk factors, and not causal claims.

## Clinical / business implications

In this sample, elevated blood sugar and systolic blood pressure are the measurements most worth flagging for follow-up, and a simple Logistic Regression baseline catches roughly 78% of High Risk cases in the test set while keeping the false-negative rate for High Risk lower than a Decision Tree. That's a reasonable starting point for a **screening aid that prioritizes cases for a clinician to review**, not a diagnostic system. The Mid Risk class is currently poorly served by both models (Recall 0.19) — before any real-world use, Mid Risk needs either more independent (non-duplicated) data or a revised label scheme, since right now the model cannot reliably tell Mid Risk apart from its neighbors.

## Limitations

- **Small, single-region sample:** 452 unique records after deduplication, all from rural/community clinics in one country (Bangladesh) — may not generalize elsewhere.
- **Deduplication trade-off:** removing 562 duplicate rows was necessary to avoid leakage and inflated mid/high representation, but it shrank the Mid Risk class the most (336 → ~106 unique cases), which is very likely the direct cause of the weak Mid Risk Recall in both models.
- **Label noise:** 21 identical feature combinations were assigned different RiskLevel labels in the source data, implying a ceiling on achievable accuracy independent of model choice.
- **Six features only:** no history of prior pregnancies, BMI, gestational age, or lab-confirmed diagnoses — all clinically relevant factors absent from this dataset.
- **No external validation:** evaluated on a held-out split of the same dataset, not on a different clinic, time period, or population.
- **Association, not causation:** all correlation/coefficient/importance results describe patterns in this specific sample.
- **Not a diagnostic tool** — an educational baseline only.

## Repository structure

```
maternal-health-risk/
├── README.md
├── requirements.txt
├── notebook.ipynb
├── figures/              # exported chart PNGs (generated by the notebook)
└── data/
    └── Maternal Health Risk Data Set.csv
```

## Running the code

```bash
# clone the repo, then from its root:
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

jupyter notebook notebooks.ipynb
# Kernel -> Restart & Run All
```

Random seed is fixed (`random_seed = 42`) for the train/test split and the Decision Tree, so results are reproducible. Figures are written to `figures/` automatically as each chart is generated.
