# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

df_raw = pd.read_csv("Maternal_Health_Risk_Data_Set.csv")
df_raw.columns = [c.strip() for c in df_raw.columns]
df_raw["RiskLevel"] = df_raw["RiskLevel"].str.strip().str.lower()
# Canonical analytical dataset: remove the two implausible HR=7 records.
df = df_raw[df_raw["HeartRate"] >= 30].copy()

order = ["low risk", "mid risk", "high risk"]
labels_disp = {"low risk": "Low Risk", "mid risk": "Mid Risk", "high risk": "High Risk"}

C_LOW  = "#2E9E6B"
C_MID  = "#E8A33D"
C_HIGH = "#D64545"
C_TEAL_DARK = "#0E3B36"
C_TEAL = "#14746B"
C_BG = "#F5F7F8"
C_CARD = "#FFFFFF"
C_TEXT = "#1F2937"
C_SUBTEXT = "#6B7280"
C_GRID = "#E5E9EC"
palette = {"low risk": C_LOW, "mid risk": C_MID, "high risk": C_HIGH}
plt.rcParams["font.family"] = "DejaVu Sans"

# ---------- stats ----------
total = len(df)
pct_high = (df["RiskLevel"] == "high risk").mean() * 100
avg_bs = df["BS"].mean()
avg_age = df["Age"].mean()

counts = df["RiskLevel"].value_counts().reindex(order).fillna(0)
grp = df.groupby("RiskLevel")[["BS", "SystolicBP", "DiastolicBP", "Age", "HeartRate"]].mean().reindex(order)

# ---------- figure ----------
fig = plt.figure(figsize=(18, 11), dpi=200, facecolor=C_BG)
gs = GridSpec(5, 1, height_ratios=[0.9, 1.5, 3.3, 3.3, 3.6], hspace=0.55, figure=fig,
              left=0.045, right=0.975, top=0.965, bottom=0.075)

# ---- Title banner ----
ax_title = fig.add_subplot(gs[0])
ax_title.set_facecolor(C_TEAL_DARK)
ax_title.axis("off")
rect = mpatches.FancyBboxPatch((0, 0), 1, 1, transform=ax_title.transAxes,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0, facecolor=C_TEAL_DARK, zorder=0)
ax_title.add_patch(rect)
ax_title.text(0.025, 0.68, "Maternal Health Risk Dashboard", color="white",
              fontsize=26, fontweight="bold", va="center", ha="left", transform=ax_title.transAxes)
ax_title.text(0.025, 0.22, "Vital-sign patterns across low / mid / high maternal risk groups  ·  UCI Maternal Health Risk dataset (rural Bangladesh, IoT-monitored)",
              color="#B7D8D3", fontsize=11.5, va="center", ha="left", transform=ax_title.transAxes)
ax_title.text(0.975, 0.5, "n = %d patients" % total, color="white", fontsize=12,
              va="center", ha="right", transform=ax_title.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", fc=C_TEAL, ec="none"))

# ---- KPI cards ----
gs_kpi = gs[1].subgridspec(1, 4, wspace=0.10)
kpis = [
    ("TOTAL PATIENTS", f"{total:,}", C_TEAL_DARK, None),
    ("HIGH RISK SHARE", f"{pct_high:.1f}%", C_HIGH, None),
    ("AVG BLOOD SUGAR", f"{avg_bs:.1f} mmol/L", C_TEAL_DARK, None),
    ("AVG MATERNAL AGE", f"{avg_age:.1f} yrs", C_TEAL_DARK, None),
]
for i, (label, value, color, _) in enumerate(kpis):
    ax = fig.add_subplot(gs_kpi[i])
    ax.axis("off")
    shadow = mpatches.FancyBboxPatch((0.025, 0.03), 0.96, 0.9, transform=ax.transAxes,
                                      boxstyle="round,pad=0.02,rounding_size=0.09",
                                      linewidth=0, facecolor="#000000", alpha=0.045, zorder=0)
    ax.add_patch(shadow)
    box = mpatches.FancyBboxPatch((0.02, 0.06), 0.96, 0.9, transform=ax.transAxes,
                                   boxstyle="round,pad=0.02,rounding_size=0.09",
                                   linewidth=1, edgecolor="#E7EBEE", facecolor=C_CARD, zorder=1)
    ax.add_patch(box)
    ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.06), 0.045, 0.9, transform=ax.transAxes,
                                          boxstyle="round,pad=0,rounding_size=0.045",
                                          linewidth=0, facecolor=color, zorder=2))
    ax.text(0.19, 0.63, value, fontsize=24, fontweight="bold", color=C_TEXT,
            va="center", ha="left", transform=ax.transAxes, zorder=3)
    ax.text(0.19, 0.30, label, fontsize=10, color=C_SUBTEXT, fontweight="medium",
            va="center", ha="left", transform=ax.transAxes, zorder=3, family="sans-serif")

def style_bar_axis(ax, title):
    ax.set_facecolor(C_CARD)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#D1D7DC")
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(colors=C_SUBTEXT, labelsize=9.5, length=0)
    ax.set_title(title, fontsize=12, fontweight="bold", color=C_TEXT, loc="left", pad=11)
    ax.grid(axis="y", color=C_GRID, linewidth=0.8, zorder=0, linestyle="-")
    ax.set_axisbelow(True)

def bar_by_risk(ax, values, title, fmt="{:.1f}", ylabel=""):
    x = range(len(order))
    colors = [palette[o] for o in order]
    bars = ax.bar(x, values, color=colors, width=0.55, zorder=3,
                   edgecolor="none")
    for b in bars:
        b.set_capstyle("round")
    ax.set_xticks(list(x))
    ax.set_xticklabels([labels_disp[o] for o in order], fontsize=9.8, color=C_TEXT)
    style_bar_axis(ax, title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.025, fmt.format(v),
                 ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=C_TEXT)
    ax.set_ylim(0, max(values) * 1.22)
    ax.tick_params(axis="y", labelsize=9)

# ---- Row: Risk distribution / Blood sugar / Systolic BP ----
gs_r1 = gs[2].subgridspec(1, 3, wspace=0.28)

ax1 = fig.add_subplot(gs_r1[0])
bar_by_risk(ax1, counts.values, "Patients by Risk Level", fmt="{:.0f}")

ax2 = fig.add_subplot(gs_r1[1])
bar_by_risk(ax2, grp["BS"].values, "Avg Blood Sugar (mmol/L) by Risk", fmt="{:.1f}")

ax3 = fig.add_subplot(gs_r1[2])
bar_by_risk(ax3, grp["SystolicBP"].values, "Avg Systolic BP (mmHg) by Risk", fmt="{:.0f}")

# ---- Row: Diastolic BP / Age / Heart Rate ----
gs_r2 = gs[3].subgridspec(1, 3, wspace=0.28)

ax4 = fig.add_subplot(gs_r2[0])
bar_by_risk(ax4, grp["DiastolicBP"].values, "Avg Diastolic BP (mmHg) by Risk", fmt="{:.0f}")

ax5 = fig.add_subplot(gs_r2[1])
bar_by_risk(ax5, grp["Age"].values, "Avg Maternal Age (yrs) by Risk", fmt="{:.1f}")

ax6 = fig.add_subplot(gs_r2[2])
bar_by_risk(ax6, grp["HeartRate"].values, "Avg Heart Rate (bpm) by Risk", fmt="{:.0f}")

# ---- Row: Scatter ----
gs_r3 = gs[4].subgridspec(1, 3, wspace=0.28, width_ratios=[2, 1, 0.02])
ax7 = fig.add_subplot(gs_r3[0])
for o in order:
    sub = df[df["RiskLevel"] == o]
    ax7.scatter(sub["BS"], sub["SystolicBP"], s=26, color=palette[o], alpha=0.65,
                edgecolor="white", linewidth=0.3, label=labels_disp[o], zorder=3)
ax7.set_xlabel("Blood Sugar (mmol/L)", fontsize=10.5, color=C_SUBTEXT)
ax7.set_ylabel("Systolic BP (mmHg)", fontsize=10.5, color=C_SUBTEXT)
style_bar_axis(ax7, "Blood Sugar vs Systolic BP (by Risk Level)")
ax7.grid(axis="both", color="#E2E8F0", linewidth=0.8, zorder=0)
leg = ax7.legend(loc="upper left", frameon=False, fontsize=9.5)

# ---- Donut: Risk share ----
ax8 = fig.add_subplot(gs_r3[1])
vals = counts.values
colors = [palette[o] for o in order]
wedges, _ = ax8.pie(vals, colors=colors, startangle=90, counterclock=False,
                     wedgeprops=dict(width=0.42, edgecolor=C_BG, linewidth=3))
ax8.set_title("Risk Share", fontsize=12.5, fontweight="bold", color=C_TEXT, loc="center", pad=10)
ax8.text(0, 0, f"{total}\npatients", ha="center", va="center", fontsize=12, fontweight="bold", color=C_TEXT)
handles = [mpatches.Patch(color=palette[o], label=f"{labels_disp[o]}  {counts[o]/total*100:.0f}%") for o in order]
ax8.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1,
           frameon=False, fontsize=9.5, handlelength=1.2, labelspacing=0.6)

fig.savefig("maternal_health_dashboard.png", facecolor=C_BG, dpi=200)
print("saved")
