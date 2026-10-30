"""Charts for the write-up. Run after model.py:  python figures.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model import d, a, g, PILOT, loss, CAUSES

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, CRIT = "#2a78d6", "#eb6834", "#d03b3b"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

def clean(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

# ---- Fig 1: where the money leaks
t = (d[loss & (d.primary_cause != "Unexplained")].groupby("primary_cause")
     .agg(n=("location_id", "count"), usd=("margin_mo", lambda x: -x.sum())).sort_values("usd"))
fig, ax = plt.subplots(figsize=(6.6, 2.5))
ax.barh(t.index, t.usd / 1000, color=BLUE, height=0.6)
for y, (usd, n) in enumerate(zip(t.usd, t.n)):
    ax.text(usd / 1000 + 0.8, y, f"${usd/1000:,.0f}k  ·  {n} sites", va="center", color=INK2, fontsize=8.5)
ax.set_xlabel("Monthly loss on loss-making sites, by primary cause ($k)")
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); clean(ax)
ax.set_xlim(0, t.usd.max() / 1000 * 1.35)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.text(0.01, 0.95, "Most losses come from geography and pricing, not vendor rates", fontsize=10.5,
         fontweight="bold", color=INK, va="top")
fig.savefig("figures/fig1_loss_by_cause.png", dpi=220); plt.close(fig)

# ---- Fig 2: two pilot metros, before vs after zoning + fixes
fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.6))
for r, m in enumerate(PILOT):
    s = a[a.metro == m]
    ax = axes[r, 0]
    ok = s.margin_mo >= 0
    ax.scatter(s.lon[ok], s.lat[ok], s=9, color="#b9b8b2", lw=0)
    ax.scatter(s.lon[~ok], s.lat[~ok], s=12, color=CRIT, lw=0)
    ax.set_title(f"{m} · before: {(~ok).sum()} of {len(s)} sites lose money\n"
                 f"{s.vendor_id.nunique()} vendors · GM {abs(s.margin_mo.sum()/s.customer_price_mo.sum())*100:.0f}%",
                 loc="left", fontsize=8.5, color=INK)
    ax = axes[r, 1]
    s2 = s[~s.churned]
    zones = sorted(s2.zone.unique())
    for i, z in enumerate(zones):
        zz = s2[s2.zone == z]
        ax.scatter(zz.lon, zz.lat, s=9, color=CAT[i % len(CAT)], lw=0, alpha=0.85)
    still = s2[s2.margin_after < 0]
    ax.scatter(still.lon, still.lat, s=26, facecolors="none", edgecolors=CRIT, lw=1)
    ax.set_title(f"{m} · after: {len(zones)} vendor zones, {len(still)} sites still losing\n"
                 f"GM {s.margin_after.sum()/s.customer_price_after.sum():.0%}"
                 f" ({int(s.churned.sum())} customers exited at repricing)",
                 loc="left", fontsize=8.5, color=INK)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(GRID)
fig.text(0.01, 0.005, "Left: red = losing money, grey = profitable.  Right: colour = vendor zone"
         " (one primary vendor each), red ring = still losing.", fontsize=7.5, color=INK2)
fig.suptitle("Pilot metros: vendor zones + targeted fixes remove most losses (simulated)",
             x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0.02, 1, 0.97)); fig.savefig("figures/fig2_pilot_maps.png", dpi=220); plt.close(fig)

# ---- Fig 3: gross margin over 24 months, pilot vs control
fig, ax = plt.subplots(figsize=(6.6, 2.5))
for grp, col in [("Pilot", BLUE), ("Control", ORANGE)]:
    s = g[g.group == grp]
    x = np.arange(len(s))
    ax.plot(x, s.gross_margin * 100, color=col, lw=2)
    ax.text(x[-1] + 0.4, s.gross_margin.iloc[-1] * 100,
            f"{'Pilot (Atlanta, Dallas)' if grp=='Pilot' else 'Control (Charlotte, Phoenix)'}",
            va="center", color=INK, fontsize=8.5)
months = g[g.group == "Pilot"].month.tolist()
ax.axvline(12, color=INK2, ls="--", lw=1)
ax.text(12.3, ax.get_ylim()[0] + 2, "fixes start Jan 2026,\nphased over 6 months", color=INK2, fontsize=7.5)
ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xticklabels([months[i] for i in [0, 6, 12, 18, 23]])
ax.set_ylabel("Gross margin (%)"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); clean(ax)
ax.set_xlim(0, 33)
p = g[g.group == "Pilot"]; c = g[g.group == "Control"]
yoy_p = (p[p.month.str.startswith("2026")].gross_margin.mean() - p[p.month.str.startswith("2025")].gross_margin.mean()) * 100
yoy_c = (c[c.month.str.startswith("2026")].gross_margin.mean() - c[c.month.str.startswith("2025")].gross_margin.mean()) * 100
ax.set_title(f"Pilot margin +{yoy_p:.0f} pts year over year; control drifts {yoy_c:.0f} pts (simulated)",
             loc="left", fontsize=10.5, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig("figures/fig3_margin_trend.png", dpi=220); plt.close(fig)
print("figures written")
