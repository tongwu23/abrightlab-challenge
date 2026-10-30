"""
abrightlab 2,000-Location Challenge - location profitability & root-cause model
Author: Tong Wu

All data here is SIMULATED. We plant six known loss causes into 2,000 synthetic
locations, then check whether a rules-based diagnostic built only on "observable"
fields (the kind Peazy + finance would hold) can find them again, attach a dollar
value to each cause, and prescribe a fix. Two pilot metros (Atlanta, Dallas) are
then "fixed" and compared with two control metros over 24 months.

Run:  python model.py          (writes data/*.csv and figures/*.png)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

rng = np.random.default_rng(7)
WEEKS_PER_MONTH = 4.33
BURDEN = 1.35        # payroll taxes, insurance on top of wage
VENDOR_MARKUP = 1.25 # vendor overhead + profit
SQFT_PER_HOUR = 3000 # standard cleaning productivity
TARGET_MARGIN = 0.25
COST_PER_MILE = 0.60 # vehicle cost (one-way leg charged per site; crews chain sites)
MPH = 30

# ---------------------------------------------------------------- 1. simulate
metros = {  # name: (lat, lon, wage $/hr, n sites, vendors in market)
    "Atlanta":   (33.75, -84.39, 17.0, 220, 9),
    "Dallas":    (32.78, -96.80, 16.5, 210, 8),
    "Charlotte": (35.23, -80.84, 16.0, 180, 4),
    "Phoenix":   (33.45, -112.07, 17.5, 190, 4),
    "Chicago":   (41.88, -87.63, 19.5, 200, 5),
    "Denver":    (39.74, -104.99, 19.0, 160, 4),
    "Houston":   (29.76, -95.37, 16.0, 190, 5),
    "Orlando":   (28.54, -81.38, 15.5, 170, 4),
    "Nashville": (36.16, -86.78, 16.0, 150, 4),
}
PILOT = ["Atlanta", "Dallas"]      # set up badly, chosen for the fix
CONTROL = ["Charlotte", "Phoenix"]  # comparable, left unchanged

rows = []
for m, (lat, lon, wage, n, _) in metros.items():
    for _ in range(n):
        rows.append(dict(metro=m, lat=lat + rng.normal(0, 0.16), lon=lon + rng.normal(0, 0.20),
                         local_wage=wage * rng.uniform(0.95, 1.05)))
n_rural = 2000 - len(rows)
for _ in range(n_rural):   # small-town sites scattered between metros
    rows.append(dict(metro="Non-metro", lat=rng.uniform(30, 44), lon=rng.uniform(-110, -78),
                     local_wage=rng.uniform(13.0, 15.0)))
df = pd.DataFrame(rows)
df.insert(0, "location_id", [f"L{i:04d}" for i in range(len(df))])
N = len(df)

types = np.array(["Retail", "Bank branch", "Office", "Restaurant", "Medical"])
df["site_type"] = rng.choice(types, N, p=[.35, .2, .2, .15, .1])
df["sqft"] = np.clip(rng.lognormal(np.log(4500), 0.45, N), 1500, 20000).round(-1)
base_freq = {"Retail": 5, "Bank branch": 3, "Office": 5, "Restaurant": 7, "Medical": 6}
df["contract_visits_wk"] = df.site_type.map(base_freq)

# distance to nearest other location (miles) - "density"
lat_r, lon_r = np.radians(df.lat.values), np.radians(df.lon.values)
def haversine(a1, o1, a2, o2):
    d = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(d))
D = haversine(lat_r[:, None], lon_r[:, None], lat_r[None, :], lon_r[None, :])
np.fill_diagonal(D, np.inf)
df["nearest_site_mi"] = D.min(axis=1).round(1)

# ---- planted causes (ground truth, hidden from the diagnostic)
is_metro = df.metro != "Non-metro"
pilot = df.metro.isin(PILOT)
def plant(p_base, p_pilot, mask=None):
    p = np.where(pilot, p_pilot, p_base)
    hit = rng.random(N) < p
    return hit & (mask if mask is not None else True)

gt = pd.DataFrame(index=df.index)
gt["underpriced"]   = plant(.05, .20)
gt["scope_creep"]   = plant(.04, .12)
gt["over_frequency"] = plant(.03, .07)
gt["vendor_cost"]   = plant(.04, .10)   # overpriced or under-delivering
gt["vendor_far"]    = plant(.03, .20, is_metro.values)
gt["remote_site"]   = (df.nearest_site_mi > 30).values

# ---- operations
std_hours = df.sqft / SQFT_PER_HOUR
df["contract_hours_visit"] = std_hours.round(2)
df["scheduled_visits_wk"] = df.contract_visits_wk + np.where(gt.over_frequency, rng.integers(2, 4, N), 0)
scope_mult = np.where(gt.scope_creep, rng.uniform(1.35, 1.7, N), rng.uniform(0.97, 1.05, N))
df["actual_scope_hours_visit"] = (std_hours * scope_mult).round(2)
df["extra_work_orders_mo"] = np.where(gt.scope_creep, rng.poisson(6, N), rng.poisson(0.6, N))

# vendor travel: normal metro 3-12 mi, mis-assigned 30-60 mi, remote sites = nearest vendor far
travel = rng.uniform(3, 12, N)
travel = np.where(gt.vendor_far, rng.uniform(25, 45, N), travel)
travel = np.where(gt.remote_site, rng.uniform(20, 40, N), travel)
df["vendor_travel_mi"] = travel.round(1)

rate_mult = np.ones(N) * rng.uniform(0.95, 1.05, N)
onsite_ratio = rng.uniform(0.88, 1.0, N)
vc = gt.vendor_cost.values
over = vc & (rng.random(N) < 0.5)
under = vc & ~over
rate_mult = np.where(over, rng.uniform(1.35, 1.6, N), rate_mult)
onsite_ratio = np.where(under, rng.uniform(0.40, 0.65, N), onsite_ratio)
df["vendor_hourly_rate"] = (df.local_wage * BURDEN * VENDOR_MARKUP * rate_mult).round(2)

paid_hours_visit = df.actual_scope_hours_visit
df["geofence_onsite_hours_visit"] = (paid_hours_visit * onsite_ratio).round(2)
travel_cost_visit = df.vendor_travel_mi * COST_PER_MILE + df.vendor_travel_mi / MPH * df.local_wage * BURDEN
visits_mo = df.scheduled_visits_wk * WEEKS_PER_MONTH
df["vendor_invoice_mo"] = ((paid_hours_visit * df.vendor_hourly_rate + travel_cost_visit) * visits_mo).round(0)

# fair should-cost & customer price
std_travel_visit = 7.5 * COST_PER_MILE + 7.5 / MPH * df.local_wage * BURDEN
std_rate = df.local_wage * BURDEN * VENDOR_MARKUP
should_cost = (std_hours * std_rate + std_travel_visit) * df.contract_visits_wk * WEEKS_PER_MONTH
price = should_cost / (1 - TARGET_MARGIN) * rng.uniform(0.95, 1.08, N)
price = np.where(gt.underpriced, price * rng.uniform(0.55, 0.75, N), price)
df["customer_price_mo"] = price.round(0)

# quality signals
df["inspection_score"] = np.clip(rng.normal(86, 5, N) + np.where(gt.over_frequency, 8, 0)
                                 - np.where(under, 12, 0), 50, 100).round(0)
df["complaints_90d"] = rng.poisson(np.where(under, 4, 0.8) * np.where(gt.over_frequency, 0.2, 1))

# vendors in each metro (for fragmentation view)
vend = []
for i, r in df.iterrows():
    k = metros.get(r.metro, (0, 0, 0, 0, 1))[4]
    vend.append(f"{r.metro[:3].upper()}-V{rng.integers(1, k + 1)}" if r.metro != "Non-metro" else f"NM-V{rng.integers(1, 60)}")
df["vendor_id"] = vend

df["margin_mo"] = df.customer_price_mo - df.vendor_invoice_mo
df["margin_pct"] = df.margin_mo / df.customer_price_mo

# ---------------------------------------------------------------- 2. diagnose
# Only observable fields are used below - never the gt table.
d = df.copy()
d["should_cost_mo"] = should_cost.round(0)
d["benchmark_price_mo"] = (should_cost / (1 - TARGET_MARGIN)).round(0)
metro_rate_median = d.groupby("metro").vendor_hourly_rate.transform("median")
v_mo = d.scheduled_visits_wk * WEEKS_PER_MONTH

flags = pd.DataFrame(index=d.index)
leak = pd.DataFrame(index=d.index)   # $/month attributable to each cause

flags["Underpriced"] = d.customer_price_mo < 0.88 * d.benchmark_price_mo
leak["Underpriced"] = np.where(flags.Underpriced, d.benchmark_price_mo - d.customer_price_mo, 0)

scope_ratio = d.actual_scope_hours_visit / d.contract_hours_visit
flags["Scope creep, not billed"] = (scope_ratio > 1.2) | (d.extra_work_orders_mo >= 4)
leak["Scope creep, not billed"] = np.where(flags["Scope creep, not billed"],
    (d.actual_scope_hours_visit - d.contract_hours_visit).clip(lower=0) * d.vendor_hourly_rate * v_mo, 0)

extra_visits = d.scheduled_visits_wk - d.contract_visits_wk
flags["Over-frequency"] = (extra_visits > 0) & (d.inspection_score >= 88)
leak["Over-frequency"] = np.where(flags["Over-frequency"],
    extra_visits * WEEKS_PER_MONTH * d.vendor_invoice_mo / v_mo, 0)

extra_travel = (travel_cost_visit - std_travel_visit).clip(lower=0) * v_mo
flags["Remote site"] = d.nearest_site_mi > 30
flags["Vendor too far"] = (d.vendor_travel_mi > 25) & ~flags["Remote site"]
leak["Remote site"] = np.where(flags["Remote site"], extra_travel, 0)
leak["Vendor too far"] = np.where(flags["Vendor too far"], extra_travel, 0)

onsite = d.geofence_onsite_hours_visit / d.actual_scope_hours_visit
flags["Vendor overpriced / under-delivering"] = (d.vendor_hourly_rate > 1.25 * metro_rate_median) | (onsite < 0.75)
leak["Vendor overpriced / under-delivering"] = np.where(flags["Vendor overpriced / under-delivering"],
    (d.vendor_hourly_rate - metro_rate_median).clip(lower=0) * d.actual_scope_hours_visit * v_mo
    + (1 - onsite).clip(lower=0) * d.actual_scope_hours_visit * d.vendor_hourly_rate * v_mo * (onsite < 0.75), 0)

CAUSES = list(flags.columns)
FIX = {
    "Underpriced": "Reprice at renewal to benchmark (wage x sqft x frequency)",
    "Scope creep, not billed": "Bill added scope as change orders; lock scope in Peazy",
    "Over-frequency": "Align schedule to contract / need (inspection-driven)",
    "Remote site": "Travel surcharge or bundle into a route; exit if still negative",
    "Vendor too far": "Reassign to nearest zone vendor (cluster)",
    "Vendor overpriced / under-delivering": "Pay geofence-verified hours + quality bonus; rebid",
}
loss = d.margin_mo < 0
d["primary_cause"] = np.where(loss, leak[CAUSES].idxmax(axis=1), "")
d.loc[loss & (leak[CAUSES].sum(axis=1) == 0), "primary_cause"] = "Unexplained"
d["all_causes"] = flags.apply(lambda r: "; ".join(c for c in CAUSES if r[c]), axis=1)
d["recommended_fix"] = d.primary_cause.map(FIX).fillna("")
d["leakage_mo"] = leak[CAUSES].sum(axis=1).round(0)

# validation vs planted truth
gt_map = {"Underpriced": "underpriced", "Scope creep, not billed": "scope_creep",
          "Over-frequency": "over_frequency", "Remote site": "remote_site",
          "Vendor too far": "vendor_far", "Vendor overpriced / under-delivering": "vendor_cost"}
val = []
for c, g in gt_map.items():
    tp = (flags[c] & gt[g]).sum(); fp = (flags[c] & ~gt[g]).sum(); fn = (~flags[c] & gt[g]).sum()
    val.append(dict(cause=c, planted=int(gt[g].sum()), flagged=int(flags[c].sum()),
                    precision=round(tp / max(tp + fp, 1), 2), recall=round(tp / max(tp + fn, 1), 2)))
val = pd.DataFrame(val)

# ---------------------------------------------------------------- 3. fix pilot metros
a = d.copy()
pm = a.metro.isin(PILOT)
lands = lambda: rng.random(len(a)) < 0.75   # not every fix lands inside the pilot window
new_price = a.customer_price_mo.astype(float).copy()
f = flags
# price-side fixes
new_price = np.where(pm & f.Underpriced & lands(), np.minimum(a.benchmark_price_mo, a.customer_price_mo * 1.30), new_price)
scope_bill = (a.actual_scope_hours_visit - a.contract_hours_visit).clip(lower=0) * a.vendor_hourly_rate \
             * a.contract_visits_wk * WEEKS_PER_MONTH / (1 - TARGET_MARGIN)
new_price = np.where(pm & f["Scope creep, not billed"], new_price + scope_bill, new_price)

# cost-side fixes
sched = np.where(pm & f["Over-frequency"] & lands(), a.contract_visits_wk, a.scheduled_visits_wk)
rate = np.where(pm & f["Vendor overpriced / under-delivering"] & lands(), np.minimum(a.vendor_hourly_rate, metro_rate_median), a.vendor_hourly_rate)
paid_h = np.where(pm & f["Vendor overpriced / under-delivering"] & (onsite < 0.75),
                  a.geofence_onsite_hours_visit, a.actual_scope_hours_visit)

# geographic zoning: one primary vendor per zone, travel = within-zone route hop
a["zone"] = ""
new_travel = a.vendor_travel_mi.values.astype(float).copy()
for m in PILOT:
    idx = a.index[a.metro == m]
    k = max(3, len(idx) // 35)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(a.loc[idx, ["lat", "lon"]])
    a.loc[idx, "zone"] = [f"{m[:3].upper()}-Z{z + 1}" for z in km.labels_]
    # route hop ~ distance to nearest site in same zone; first stop amortised over route
    for z in range(k):
        zi = idx[km.labels_ == z]
        sub = D[np.ix_(zi, zi)]
        hop = np.where(np.isinf(sub.min(axis=1)), 5, sub.min(axis=1))
        new_travel[zi] = np.clip(hop + 6 / max(len(zi), 1) * 4, 4, 12)
new_travel = np.where(pm & ~f["Remote site"], new_travel, new_travel)
t_cost = new_travel * COST_PER_MILE + new_travel / MPH * a.local_wage * BURDEN
accept_co = rng.random(len(a)) < 0.6
scope_cut = pm & f["Scope creep, not billed"] & ~accept_co
new_price = np.where(pm & f["Scope creep, not billed"] & ~accept_co, new_price - scope_bill, new_price)
paid_h = np.where(scope_cut, np.minimum(paid_h, a.contract_hours_visit), paid_h)
new_cost = (paid_h * rate + t_cost) * sched * WEEKS_PER_MONTH
churn = pm & f.Underpriced & (rng.random(len(a)) < 0.15)   # some customers walk at repricing
a["churned"] = churn
new_price = np.where(churn, 0, new_price); new_cost = np.where(churn, 0, new_cost)
a["customer_price_after"] = np.where(pm, new_price, a.customer_price_mo).round(0)
a["vendor_cost_after"] = np.where(pm, new_cost, a.vendor_invoice_mo).round(0)
a["margin_after"] = a.customer_price_after - a.vendor_cost_after
a["vendors_after"] = np.where(pm, a.zone, a.vendor_id)

def city_summary(frame, cities):
    out = []
    for c in cities:
        s = frame[frame.metro == c]
        out.append(dict(metro=c, sites=len(s), churned=int(s.churned.sum()) if "churned" in s else 0,
            vendors_before=s.vendor_id.nunique(), vendors_after=s.vendors_after.nunique(),
            loss_sites_before=int((s.margin_mo < 0).sum()), loss_sites_after=int((s.margin_after < 0).sum()),
            monthly_loss_before=int(-s.margin_mo[s.margin_mo < 0].sum()),
            monthly_loss_after=int(-s.margin_after[s.margin_after < 0].sum()),
            gm_before=round(s.margin_mo.sum() / s.customer_price_mo.sum(), 3),
            gm_after=round(s.margin_after.sum() / s.customer_price_after.sum(), 3),
            avg_travel_mi_before=round(s.vendor_travel_mi.mean(), 1)))
    return pd.DataFrame(out)
summary = city_summary(a, PILOT + CONTROL)

# ---------------------------------------------------------------- 4. 24-month view (pilot vs control)
months = pd.period_range("2025-01", periods=24, freq="M")
ts = []
for c in PILOT + CONTROL:
    s = a[a.metro == c]
    rev0, cost0 = s.customer_price_mo.sum(), s.vendor_invoice_mo.sum()
    rev1, cost1 = s.customer_price_after.sum(), s.vendor_cost_after.sum()
    for i, mth in enumerate(months):
        drift = 1 + 0.003 * i                     # unmanaged scope creep keeps eroding cost
        season = 1 + 0.015 * np.sin(2 * np.pi * i / 12)
        if c in PILOT and i >= 12:
            ramp = min(1, (i - 11) / 6)            # fixes phase in over 6 months (renewals, rebids)
            rev = rev0 + (rev1 - rev0) * ramp
            cost = (cost0 * drift) + (cost1 - cost0 * drift) * ramp
        else:
            rev, cost = rev0, cost0 * drift
        cost *= season * rng.normal(1, 0.006)
        ts.append(dict(month=str(mth), metro=c, group="Pilot" if c in PILOT else "Control",
                       revenue=rev, vendor_cost=cost))
ts = pd.DataFrame(ts)
g = ts.groupby(["month", "group"])[["revenue", "vendor_cost"]].sum().reset_index()
g["gross_margin"] = 1 - g.vendor_cost / g.revenue
yoy = g[g.month.str.startswith("2026")].groupby("group").gross_margin.mean() - \
      g[g.month.str.startswith("2025")].groupby("group").gross_margin.mean()

# ---------------------------------------------------------------- 5. outputs
cause_tbl = (d[loss].groupby("primary_cause")
             .agg(loss_sites=("location_id", "count"), monthly_loss=("margin_mo", lambda x: int(-x.sum())))
             .sort_values("monthly_loss", ascending=False))
df.to_csv("data/locations_simulated.csv", index=False)
d.loc[loss, ["location_id", "metro", "site_type", "sqft", "customer_price_mo", "vendor_invoice_mo",
             "margin_mo", "primary_cause", "all_causes", "recommended_fix", "leakage_mo"]] \
    .sort_values("margin_mo").to_csv("data/loss_locations_diagnosis.csv", index=False)
val.to_csv("data/diagnostic_validation.csv", index=False)
summary.to_csv("data/pilot_vs_control_summary.csv", index=False)
g.to_csv("data/monthly_margin_pilot_vs_control.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(f"Locations: {N}  | loss-making: {loss.sum()} ({loss.mean():.1%})  | "
          f"network GM: {d.margin_mo.sum() / d.customer_price_mo.sum():.1%}  | "
          f"monthly loss on losing sites: ${-d.margin_mo[loss].sum():,.0f}")
    print("\nLoss by primary cause:\n", cause_tbl)
    print("\nDiagnostic vs planted truth:\n", val)
    print("\nPilot vs control:\n", summary.T)
    print("\nYoY change in gross margin (2026 avg - 2025 avg):\n", yoy.round(3))
