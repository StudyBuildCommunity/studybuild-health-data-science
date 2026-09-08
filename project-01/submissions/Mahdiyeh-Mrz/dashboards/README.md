# Dashboards

Added on top of the original `maternal-health-risk/` project (unchanged).

## Files
- `Maternal_Health_Risk_Dashboard.twbx` — Tableau packaged workbook (data embedded).
  Opens in Tableau Desktop / Tableau Public. Contains KPI cards, risk distribution,
  vitals-by-risk bar charts, and a Blood Sugar vs Systolic BP scatter.
- `overview_dashboard.png` — rendered PNG of the overview dashboard (KPIs + core charts).
- `advanced_analytics_dashboard.png` — extended analytics dashboard with:
  Lollipop (risk volume), Violin (age by risk), ECDF (age cumulative distribution),
  Area chart (high-risk trend by age band), Bubble chart (BP band vs risk intensity),
  Radar chart (clinical profile index), Horizontal bar (top ML drivers / feature importance),
  Bullet chart (accuracy & high-risk recall vs target), Bullet/range charts (mean vs median
  for Systolic BP, Diastolic BP, Blood Sugar, Heart Rate), and decision-summary cards.
- `build_overview_dashboard.py` / `build_advanced_dashboard.py` — the Python (matplotlib/pandas)
  scripts that generate the two PNGs above from `Maternal_Health_Risk_Data_Set.csv`. Re-run with
  `python3 build_advanced_dashboard.py` after installing `pandas`, `numpy`, `matplotlib`.
- `Maternal_Health_Risk_Data_Set.csv` — copy of the dataset used by the scripts/workbook.

## Sources for the ML metrics shown on the advanced dashboard
Pulled directly from this project's own `report/summary.md` and
`notebooks/analysis.ipynb` (Decision Tree, max_depth=4):
Accuracy 0.680, Macro F1 0.670, High-Risk Recall 0.836, High-Risk Precision 0.920,
feature importances: BS 0.502, SystolicBP 0.330, BodyTemp 0.115, Age 0.045,
DiastolicBP 0.008, HeartRate 0.000.
The "target" lines on the bullet scorecard (75% accuracy / 90% high-risk recall) are
illustrative screening-aid benchmarks, not part of the original project — adjust freely
in `build_advanced_dashboard.py` (the `TARGETS` dict) if you have real targets in mind.
