# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "DejaVu Sans"

df_raw = pd.read_csv("Maternal_Health_Risk_Data_Set.csv")
df_raw.columns = [c.strip() for c in df_raw.columns]
df_raw["RiskLevel"] = df_raw["RiskLevel"].str.strip().str.lower()
df = df_raw[df_raw["HeartRate"] >= 30].copy()
df_clean = df.copy()  # canonical analytical dataset: 1,012 rows

order = ["low risk", "mid risk", "high risk"]
labels_disp = {"low risk": "Low Risk", "mid risk": "Mid Risk", "high risk": "High Risk"}

C_LOW, C_MID, C_HIGH = "#2E9E6B", "#E8A33D", "#D64545"
C_TEAL_DARK, C_TEAL = "#0E3B36", "#14746B"
C_BLUE = "#3B6FA0"
C_BG, C_CARD = "#F5F7F8", "#FFFFFF"
C_TEXT, C_SUBTEXT, C_GRID = "#1F2937", "#6B7280", "#E5E9EC"
palette = {"low risk": C_LOW, "mid risk": C_MID, "high risk": C_HIGH}

# reported model metrics (from report/summary.md, notebooks/analysis.ipynb)
METRICS = {"Accuracy": 0.680, "High-Risk Recall": 0.836}
TARGETS = {"Accuracy": 0.75, "High-Risk Recall": 0.90}  # illustrative screening-aid targets
IMPORTANCE = [("Blood Sugar", 0.502), ("Systolic BP", 0.330), ("Body Temp", 0.115),
              ("Age", 0.045), ("Diastolic BP", 0.008), ("Heart Rate", 0.000)]

def clean_ax(ax, title, grid_axis="y"):
    ax.set_facecolor(C_CARD)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D1D7DC")
    ax.tick_params(colors=C_SUBTEXT, labelsize=9, length=0)
    ax.set_title(title, fontsize=12, fontweight="bold", color=C_TEXT, loc="left", pad=10)
    if grid_axis:
        ax.grid(axis=grid_axis, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

fig = plt.figure(figsize=(21, 27), dpi=170, facecolor=C_BG)
gs = GridSpec(6, 1, height_ratios=[0.6, 1.1, 2.6, 2.6, 2.0, 1.9], hspace=0.62, figure=fig,
              left=0.04, right=0.975, top=0.975, bottom=0.025)

# ---------------- Title ----------------
ax_t = fig.add_subplot(gs[0]); ax_t.axis("off")
ax_t.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, transform=ax_t.transAxes,
                boxstyle="round,pad=0,rounding_size=0.05", linewidth=0, facecolor=C_TEAL_DARK))
ax_t.text(0.018, 0.62, "Advanced Analytics Dashboard", color="white", fontsize=23,
          fontweight="bold", va="center", transform=ax_t.transAxes)
ax_t.text(0.018, 0.2, "Clinical patterns, risk drivers & model performance — Maternal Health Risk dataset",
          color="#B9D8D3", fontsize=11, va="center", transform=ax_t.transAxes)

# ---------------- Decision Summary Cards ----------------
import textwrap
cards = [
    ("PRIMARY DRIVERS", "Blood Sugar (50%) & Systolic BP (33%) dominate risk prediction", C_TEAL_DARK),
    ("HARDEST CALL", "Low vs Mid Risk overlap heavily on every vital sign", C_MID),
    ("SAFETY GAP", "9 of 55 true High-Risk cases would be missed by the model", C_HIGH),
    ("MODEL STATUS", "Decision Tree: 68% accuracy, 84% High-Risk recall \u2014 screening aid only", C_TEAL),
]
gs_c = gs[1].subgridspec(1, 4, wspace=0.10)
for i, (tag, txt, color) in enumerate(cards):
    ax = fig.add_subplot(gs_c[i]); ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.05), 0.96, 0.9, transform=ax.transAxes,
                 boxstyle="round,pad=0.02,rounding_size=0.09", linewidth=1,
                 edgecolor="#E7EBEE", facecolor=C_CARD))
    ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.87), 0.96, 0.08, transform=ax.transAxes,
                 boxstyle="round,pad=0,rounding_size=0.03", linewidth=0, facecolor=color))
    ax.text(0.5, 0.7, tag, ha="center", fontsize=9.5, fontweight="bold", color=color, transform=ax.transAxes)
    wrapped = textwrap.fill(txt, width=33)
    ax.text(0.5, 0.42, wrapped, ha="center", va="center", fontsize=9.8, color=C_TEXT,
            transform=ax.transAxes, linespacing=1.6)

# ============ ROW 2: Lollipop | Violin | ECDF ============
gs_r2 = gs[2].subgridspec(1, 3, wspace=0.26)

# --- Lollipop: Risk Volume Ranking ---
ax = fig.add_subplot(gs_r2[0])
counts = df["RiskLevel"].value_counts().reindex(order)
ranked = counts.sort_values(ascending=True)
y = np.arange(len(ranked))
colors_r = [palette[o] for o in ranked.index]
ax.hlines(y, 0, ranked.values, color=colors_r, linewidth=3, zorder=2)
ax.scatter(ranked.values, y, s=260, color=colors_r, zorder=3, edgecolor="white", linewidth=1.5)
for yi, v in zip(y, ranked.values):
    ax.text(v + 12, yi, f"{v:.0f}", va="center", fontsize=10.5, fontweight="bold", color=C_TEXT)
ax.set_yticks(y); ax.set_yticklabels([labels_disp[o] for o in ranked.index], fontsize=10.5)
ax.set_xlim(0, ranked.values.max() * 1.2)
clean_ax(ax, "Lollipop — Risk Volume Ranking (n=1,012)", grid_axis="x")

# --- Violin: Age Distribution by Risk ---
ax = fig.add_subplot(gs_r2[1])
data_v = [df[df.RiskLevel == o]["Age"].values for o in order]
parts = ax.violinplot(data_v, showmeans=False, showmedians=True, widths=0.8)
for pc, o in zip(parts['bodies'], order):
    pc.set_facecolor(palette[o]); pc.set_edgecolor(palette[o]); pc.set_alpha(0.55)
for key in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    parts[key].set_color("#374151"); parts[key].set_linewidth(1.1)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([labels_disp[o] for o in order], fontsize=10.5)
clean_ax(ax, "Violin — Age Distribution by Risk", grid_axis="y")
ax.set_ylabel("Age (years)", fontsize=9.5, color=C_SUBTEXT)

# --- ECDF: Age Cumulative Distribution ---
ax = fig.add_subplot(gs_r2[2])
for o in order:
    vals = np.sort(df[df.RiskLevel == o]["Age"].values)
    yv = np.arange(1, len(vals) + 1) / len(vals)
    ax.step(vals, yv, where="post", color=palette[o], linewidth=2.2, label=labels_disp[o])
ax.set_xlabel("Age (years)", fontsize=9.5, color=C_SUBTEXT)
ax.set_ylabel("Cumulative proportion", fontsize=9.5, color=C_SUBTEXT)
clean_ax(ax, "ECDF — Age Cumulative Distribution", grid_axis="both")
ax.legend(frameon=False, fontsize=9.5, loc="lower right")

# ============ ROW 3: Area | Bubble | Radar ============
gs_r3 = gs[2 + 1].subgridspec(1, 3, wspace=0.30)

# --- Area: High-Risk Trend Across Age Bands ---
ax = fig.add_subplot(gs_r3[0])
age_bins = [10, 20, 30, 40, 50, 60, 71]
age_labels = ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69"]
df["AgeBand"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)
trend = df.groupby("AgeBand", observed=True)["RiskLevel"].apply(lambda s: (s == "high risk").mean() * 100)
trend = trend.reindex(age_labels)
xv = np.arange(len(age_labels))
ax.fill_between(xv, trend.values, color=C_HIGH, alpha=0.28, zorder=2)
ax.plot(xv, trend.values, color=C_HIGH, linewidth=2.4, marker="o", markersize=5.5, zorder=3)
for xi, v in zip(xv, trend.values):
    if not np.isnan(v):
        ax.text(xi, v + 3, f"{v:.0f}%", ha="center", fontsize=9.5, fontweight="bold", color=C_TEXT)
ax.set_xticks(xv); ax.set_xticklabels(age_labels, fontsize=9.5)
ax.set_ylim(0, max(trend.max() * 1.3, 20))
ax.set_ylabel("High-Risk share (%)", fontsize=9.5, color=C_SUBTEXT)
clean_ax(ax, "Area — High-Risk Trend Across Age Bands", grid_axis="y")

# --- Bubble: BP Band vs Risk Intensity ---
ax = fig.add_subplot(gs_r3[1])
bp_bins = [70, 90, 110, 130, 150, 161]
bp_labels = ["70-89", "90-109", "110-129", "130-149", "150+"]
df["BPBand"] = pd.cut(df["SystolicBP"], bins=bp_bins, labels=bp_labels, right=False)
bp_g = df.groupby("BPBand", observed=True).agg(
    pct_high=("RiskLevel", lambda s: (s == "high risk").mean() * 100),
    n=("RiskLevel", "size")).reindex(bp_labels)
xv = np.arange(len(bp_labels))
sizes = bp_g["n"] / bp_g["n"].max() * 2200 + 120
ax.scatter(xv, bp_g["pct_high"], s=sizes, color=C_BLUE, alpha=0.55, edgecolor=C_TEAL_DARK, linewidth=1.2, zorder=3)
for xi, (v, n) in zip(xv, zip(bp_g["pct_high"], bp_g["n"])):
    ax.text(xi, v, f"{v:.0f}%\n(n={n:.0f})", ha="center", va="center", fontsize=8.6, fontweight="bold", color=C_TEXT)
ax.set_xticks(xv); ax.set_xticklabels(bp_labels, fontsize=9.5)
ax.set_ylim(-5, max(bp_g["pct_high"].max() * 1.3, 30))
ax.set_xlabel("Systolic BP band (mmHg)", fontsize=9.5, color=C_SUBTEXT)
ax.set_ylabel("High-Risk share (%)", fontsize=9.5, color=C_SUBTEXT)
clean_ax(ax, "Bubble — BP Band \u00b7 Risk Intensity  (size = patients)", grid_axis="y")

# --- Radar: Clinical Profile Index ---
ax = fig.add_subplot(gs_r3[2], polar=True)
vitals = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
vlabels = ["Age", "Systolic\nBP", "Diastolic\nBP", "Blood\nSugar", "Body\nTemp", "Heart\nRate"]
means = df_clean.groupby("RiskLevel")[vitals].mean().reindex(order)
norm = (means - means.min()) / (means.max() - means.min())
norm = norm.fillna(0.5) * 0.85 + 0.15
angles = np.linspace(0, 2 * np.pi, len(vitals), endpoint=False).tolist()
angles += angles[:1]
ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(vlabels, fontsize=9)
ax.set_yticklabels([]); ax.set_ylim(0, 1)
ax.spines['polar'].set_color(C_GRID)
ax.grid(color=C_GRID, linewidth=0.8)
for o in order:
    vals = norm.loc[o].tolist(); vals += vals[:1]
    ax.plot(angles, vals, color=palette[o], linewidth=2, label=labels_disp[o])
    ax.fill(angles, vals, color=palette[o], alpha=0.12)
ax.set_title("Radar — Clinical Profile Index (normalized mean)", fontsize=12, fontweight="bold",
             color=C_TEXT, pad=28, loc="left", x=-0.08)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=9)

# ============ ROW 4: ML Drivers | Bullet Scorecard ============
gs_r4 = gs[4].subgridspec(1, 2, wspace=0.22, width_ratios=[1.15, 1])

# --- Horizontal bar: Top ML Drivers ---
ax = fig.add_subplot(gs_r4[0])
names = [n for n, v in IMPORTANCE][::-1]
vals = [v * 100 for n, v in IMPORTANCE][::-1]
bar_colors = [C_TEAL_DARK if v == max(vals) else C_TEAL for v in vals]
yv = np.arange(len(names))
ax.barh(yv, vals, color=bar_colors, height=0.55, zorder=3)
for yi, v in zip(yv, vals):
    ax.text(v + 1.5, yi, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold", color=C_TEXT)
ax.set_yticks(yv); ax.set_yticklabels(names, fontsize=10.5)
ax.set_xlim(0, 60)
clean_ax(ax, "Top ML Drivers — Decision Tree Feature Importance", grid_axis="x")

# --- Bullet scorecard: Accuracy & High-Risk Recall vs target ---
ax = fig.add_subplot(gs_r4[1]); ax.axis("off")
ax.set_title("Bullet — ML Performance Scorecard", fontsize=12, fontweight="bold",
             color=C_TEXT, loc="left", pad=14, x=0.0)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
rows_b = list(METRICS.items())
band_h = 0.16
for i, (name, val) in enumerate(rows_b):
    y0 = 0.78 - i * 0.38
    target = TARGETS[name]
    ax.text(0.0, y0 + band_h + 0.05, f"{name}", fontsize=11, fontweight="bold", color=C_TEXT)
    ax.text(0.0, y0 - 0.03, f"{val*100:.1f}%  (target {target*100:.0f}%)", fontsize=9.5, color=C_SUBTEXT)
    # qualitative bands
    ax.add_patch(mpatches.Rectangle((0.32, y0), 0.55, band_h, facecolor="#E7EBEE"))
    ax.add_patch(mpatches.Rectangle((0.32, y0), 0.55 * 0.6, band_h, facecolor="#D6DDE2"))
    ax.add_patch(mpatches.Rectangle((0.32, y0), 0.55 * 0.35, band_h, facecolor="#C3CCD3"))
    # value bar
    barcolor = C_TEAL_DARK if val >= target else C_HIGH
    ax.add_patch(mpatches.Rectangle((0.32, y0 + band_h * 0.28), 0.55 * val, band_h * 0.44, facecolor=barcolor))
    # target marker
    ax.plot([0.32 + 0.55 * target] * 2, [y0 - 0.015, y0 + band_h + 0.015], color=C_TEXT, linewidth=2.2)
    ax.text(0.32 + 0.55 * val, y0 + band_h + 0.03, f"{val*100:.0f}%", fontsize=9, fontweight="bold",
            color=barcolor, ha="center")

# ============ ROW 5: Clinical Indicator Ranges (mean vs median) ============
gs_r5 = gs[5].subgridspec(1, 4, wspace=0.32)
indicators = [("Systolic BP", "SystolicBP", "mmHg"), ("Diastolic BP", "DiastolicBP", "mmHg"),
              ("Blood Sugar", "BS", "mmol/L"), ("Heart Rate", "HeartRate", "bpm")]
for i, (label, col, unit) in enumerate(indicators):
    ax = fig.add_subplot(gs_r5[i])
    vmin, vmax = df_clean[col].min(), df_clean[col].max()
    vmean, vmed = df_clean[col].mean(), df_clean[col].median()
    ax.add_patch(mpatches.FancyBboxPatch((0, 0.42), 1, 0.16, boxstyle="round,pad=0,rounding_size=0.08",
                 linewidth=0, facecolor="#E7EBEE", transform=ax.transData))
    def sc(v): return (v - vmin) / (vmax - vmin)
    ax.plot([sc(vmed)], [0.5], marker="D", markersize=11, color=C_TEAL_DARK, zorder=4, label="Median")
    ax.plot([sc(vmean)], [0.5], marker="o", markersize=13, color=C_HIGH, zorder=3, alpha=0.85, label="Mean")
    ax.text(sc(vmean), 0.72, f"Mean {vmean:.1f}", ha="center", fontsize=9, fontweight="bold", color=C_HIGH)
    ax.text(sc(vmed), 0.20, f"Median {vmed:.1f}", ha="center", fontsize=9, fontweight="bold", color=C_TEAL_DARK)
    ax.text(0, 0.0, f"{vmin:.0f}", fontsize=8.5, color=C_SUBTEXT, ha="left")
    ax.text(1, 0.0, f"{vmax:.0f}", fontsize=8.5, color=C_SUBTEXT, ha="right")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.1, 1)
    ax.axis("off")
    ax.set_title(f"{label} ({unit})", fontsize=11.5, fontweight="bold", color=C_TEXT, loc="left", pad=6)
fig.text(0.04, 0.008, "Bullet / Range — Clinical Indicator Ranges: min \u2013 max span with mean & median markers",
          fontsize=10, color=C_SUBTEXT, style="italic")

fig.savefig("advanced_analytics_dashboard.png", facecolor=C_BG, dpi=170)
print("saved advanced dashboard")
