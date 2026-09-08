-- ============================================================================
-- Maternal Health Risk Stratification - SQL Analysis
-- Database: data/maternal_health.db (SQLite)
-- Table:    maternal_health (1,012 rows, cleaned: implausible HeartRate removed,
--           duplicates intentionally KEPT - see data/README.md / METHODOLOGY.md)
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Q1a. Population overview - RiskLevel distribution + percentage of sample
-- Purpose: same figure as figures/01_risklevel_distribution.png, produced in SQL.
-- ----------------------------------------------------------------------------
SELECT
    RiskLevel,
    COUNT(*)                                                  AS n_patients,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM maternal_health), 1) AS pct_of_sample
FROM maternal_health
GROUP BY RiskLevel
ORDER BY
    CASE RiskLevel WHEN 'low risk' THEN 1 WHEN 'mid risk' THEN 2 WHEN 'high risk' THEN 3 END;


-- ----------------------------------------------------------------------------
-- Q1b. Summary statistics for every clinical variable (overall, not by group)
-- Purpose: SQL equivalent of df.describe() - min/max/avg to spot implausible
--          ranges before deeper analysis.
-- ----------------------------------------------------------------------------
SELECT
    MIN(Age)          AS age_min,        MAX(Age)          AS age_max,        ROUND(AVG(Age), 1)        AS age_avg,
    MIN(SystolicBP)   AS sbp_min,        MAX(SystolicBP)   AS sbp_max,        ROUND(AVG(SystolicBP), 1) AS sbp_avg,
    MIN(DiastolicBP)  AS dbp_min,        MAX(DiastolicBP)  AS dbp_max,        ROUND(AVG(DiastolicBP),1) AS dbp_avg,
    MIN(BS)           AS bs_min,         MAX(BS)           AS bs_max,         ROUND(AVG(BS), 2)         AS bs_avg,
    MIN(BodyTemp)     AS temp_min,       MAX(BodyTemp)     AS temp_max,       ROUND(AVG(BodyTemp), 1)   AS temp_avg,
    MIN(HeartRate)    AS hr_min,         MAX(HeartRate)    AS hr_max,         ROUND(AVG(HeartRate), 1)  AS hr_avg
FROM maternal_health;


-- ----------------------------------------------------------------------------
-- Q1c. Data-quality check: rows outside a clinically plausible range
-- Purpose: flags implausible values with plain SQL filters (sanity check that
--          the earlier Python cleaning step actually removed them).
-- ----------------------------------------------------------------------------
SELECT *
FROM maternal_health
WHERE HeartRate < 30 OR HeartRate > 200
   OR Age < 12 OR Age > 60
   OR BodyTemp > 104;


-- ----------------------------------------------------------------------------
-- Q2. Group summaries - mean/median/spread of every variable BY RiskLevel
-- Purpose: numeric backbone behind figures 02-04 and 06 (boxplots by group).
--          SQLite has no native MEDIAN(), so median is approximated with
--          AVG of the two middle ordered rows per group via a window function.
-- ----------------------------------------------------------------------------
WITH ranked AS (
    SELECT
        RiskLevel, Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate,
        ROW_NUMBER() OVER (PARTITION BY RiskLevel ORDER BY BS)        AS bs_rank,
        COUNT(*)     OVER (PARTITION BY RiskLevel)                    AS grp_n
    FROM maternal_health
)
SELECT
    RiskLevel,
    COUNT(*)                              AS n,
    ROUND(AVG(Age), 1)                    AS avg_age,
    ROUND(AVG(SystolicBP), 1)             AS avg_systolic_bp,
    ROUND(AVG(DiastolicBP), 1)            AS avg_diastolic_bp,
    ROUND(AVG(BS), 2)                     AS avg_blood_sugar,
    ROUND(AVG(BodyTemp), 1)               AS avg_body_temp,
    ROUND(AVG(HeartRate), 1)              AS avg_heart_rate,
    ROUND(MIN(BS), 2)                     AS min_blood_sugar,
    ROUND(MAX(BS), 2)                     AS max_blood_sugar
FROM maternal_health
GROUP BY RiskLevel
ORDER BY
    CASE RiskLevel WHEN 'low risk' THEN 1 WHEN 'mid risk' THEN 2 WHEN 'high risk' THEN 3 END;


-- ----------------------------------------------------------------------------
-- Q2b. Spread (population standard deviation) per group per variable
-- Purpose: complements the mean/median above - shows which variable's spread
--          widens most in the High Risk group (BS is expected to lead).
--          SQLite has no built-in STDDEV, so it is computed manually:
--          sqrt( E[x^2] - (E[x])^2 ).
-- ----------------------------------------------------------------------------
SELECT
    RiskLevel,
    ROUND(SQRT(AVG(BS*BS) - AVG(BS)*AVG(BS)), 2)                 AS stddev_blood_sugar,
    ROUND(SQRT(AVG(SystolicBP*SystolicBP) - AVG(SystolicBP)*AVG(SystolicBP)), 2)   AS stddev_systolic_bp,
    ROUND(SQRT(AVG(DiastolicBP*DiastolicBP) - AVG(DiastolicBP)*AVG(DiastolicBP)), 2) AS stddev_diastolic_bp,
    ROUND(SQRT(AVG(HeartRate*HeartRate) - AVG(HeartRate)*AVG(HeartRate)), 2) AS stddev_heart_rate
FROM maternal_health
GROUP BY RiskLevel
ORDER BY
    CASE RiskLevel WHEN 'low risk' THEN 1 WHEN 'mid risk' THEN 2 WHEN 'high risk' THEN 3 END;


-- ----------------------------------------------------------------------------
-- Q3a. Which variables separate High Risk most clearly? - threshold framing
-- Purpose: turns the EDA finding ("BS separates High Risk best") into a
--          concrete, checkable rule: what % of each group falls above a
--          clinically meaningful blood-sugar cutoff (BS >= 11 mmol/L, roughly
--          the High-Risk median found in Q2).
-- ----------------------------------------------------------------------------
SELECT
    RiskLevel,
    COUNT(*)                                                            AS n,
    SUM(CASE WHEN BS >= 11 THEN 1 ELSE 0 END)                           AS n_high_bs,
    ROUND(100.0 * SUM(CASE WHEN BS >= 11 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_high_bs
FROM maternal_health
GROUP BY RiskLevel
ORDER BY
    CASE RiskLevel WHEN 'low risk' THEN 1 WHEN 'mid risk' THEN 2 WHEN 'high risk' THEN 3 END;


-- ----------------------------------------------------------------------------
-- Q3b. Combined high-BS AND high-BP patients - how concentrated in High Risk?
-- Purpose: mirrors figures/05_scatter_bs_vs_systolicbp.png - shows that patients
--          elevated on BOTH axes are disproportionately High Risk, supporting
--          the "association, not proof of causation" discussion in the report.
-- ----------------------------------------------------------------------------
SELECT
    RiskLevel,
    COUNT(*) AS n_high_bs_and_high_sbp
FROM maternal_health
WHERE BS >= 11 AND SystolicBP >= 130
GROUP BY RiskLevel
ORDER BY n_high_bs_and_high_sbp DESC;


-- ----------------------------------------------------------------------------
-- Q3c. Rank each variable's separating power via a simple SQL proxy
-- Purpose: for each variable, computes (High Risk avg - Low Risk avg) as a
--          quick, dependency-free proxy for "how much does this variable move
--          between the least and most severe group" - a SQL-only complement
--          to the Python feature-importance analysis in figures/08.
-- Caveat: this raw difference is in each variable's own unit (mmHg vs
--          mmol/L vs bpm), so it is NOT directly comparable across variables
--          the way the Python model's standardized coefficients / Gini
--          importances are - SystolicBP tops this SQL ranking simply because
--          mmHg values are numerically larger than mmol/L values, not
--          because BP is a stronger predictor than BS. Use this query to see
--          the direction and rough scale of each shift, and rely on
--          figures/08_feature_importance.png (unit-free) for actual ranking.
-- ----------------------------------------------------------------------------
WITH grp AS (
    SELECT RiskLevel, AVG(Age) AS avg_age, AVG(SystolicBP) AS avg_sbp,
           AVG(DiastolicBP) AS avg_dbp, AVG(BS) AS avg_bs,
           AVG(BodyTemp) AS avg_temp, AVG(HeartRate) AS avg_hr
    FROM maternal_health
    GROUP BY RiskLevel
),
diffs AS (
    SELECT
        'BS'         AS variable, ROUND(h.avg_bs  - l.avg_bs, 2)  AS high_minus_low_avg FROM grp h, grp l WHERE h.RiskLevel='high risk' AND l.RiskLevel='low risk'
    UNION ALL
    SELECT 'SystolicBP', ROUND(h.avg_sbp - l.avg_sbp, 2) FROM grp h, grp l WHERE h.RiskLevel='high risk' AND l.RiskLevel='low risk'
    UNION ALL
    SELECT 'DiastolicBP', ROUND(h.avg_dbp - l.avg_dbp, 2) FROM grp h, grp l WHERE h.RiskLevel='high risk' AND l.RiskLevel='low risk'
    UNION ALL
    SELECT 'BodyTemp',    ROUND(h.avg_temp - l.avg_temp, 2) FROM grp h, grp l WHERE h.RiskLevel='high risk' AND l.RiskLevel='low risk'
    UNION ALL
    SELECT 'HeartRate',   ROUND(h.avg_hr - l.avg_hr, 2) FROM grp h, grp l WHERE h.RiskLevel='high risk' AND l.RiskLevel='low risk'
    UNION ALL
    SELECT 'Age',         ROUND(h.avg_age - l.avg_age, 2) FROM grp h, grp l WHERE h.RiskLevel='high risk' AND l.RiskLevel='low risk'
)
SELECT variable, high_minus_low_avg
FROM diffs
ORDER BY ABS(high_minus_low_avg) DESC;


-- ----------------------------------------------------------------------------
-- Q_extra. Age-band breakdown of risk (clinical sanity check)
-- Purpose: buckets patients into age bands with CASE/WHEN and cross-tabulates
--          against RiskLevel - checks whether risk skews toward the very
--          young/older ages flagged as unusual in the EDA.
-- ----------------------------------------------------------------------------
SELECT
    CASE
        WHEN Age < 18 THEN '<18'
        WHEN Age BETWEEN 18 AND 34 THEN '18-34'
        WHEN Age BETWEEN 35 AND 45 THEN '35-45'
        ELSE '46+'
    END                     AS age_band,
    RiskLevel,
    COUNT(*)                AS n
FROM maternal_health
GROUP BY age_band, RiskLevel
ORDER BY age_band,
    CASE RiskLevel WHEN 'low risk' THEN 1 WHEN 'mid risk' THEN 2 WHEN 'high risk' THEN 3 END;
