# abrightlab 2,000-Location Challenge — location profitability & root-cause model

**Tong Wu** · Write-up: `TongWu_abrightlab_Challenge.pdf`

A loss-making location is not one problem but six. This repo simulates 2,000 sites, plants six known loss
causes, and shows that a rules-based diagnostic using only *observable* data (Peazy + finance) can find them,
put a dollar value on each, and prescribe a fix. Two pilot metros are then fixed and compared with two controls.

| Cause | Data signal | Fix |
|---|---|---|
| Underpriced | price < 88% of benchmark (wage × sqft × frequency) | reprice at renewal |
| Scope creep, not billed | actual hours > 1.2× contract, many extra work orders | change orders |
| Remote site | nearest other site > 30 mi | surcharge / route bundle / exit |
| Over-frequency | scheduled > contracted visits, high inspection score | align schedule |
| Vendor too far | vendor travel > 25 mi in a dense area | reassign to zone vendor |
| Vendor overpriced / under-delivering | rate > 1.25× metro median or geofence < 75% of paid hours | pay verified hours + bonus; rebid |

## Run
```bash
pip install numpy pandas scikit-learn matplotlib
python model.py      # prints results, writes data/*.csv
python figures.py    # writes figures/*.png
```

## Outputs
- `data/loss_locations_diagnosis.csv` — one row per losing site: causes, $ leakage, recommended fix
- `data/diagnostic_validation.csv` — precision / recall vs. planted truth
- `data/pilot_vs_control_summary.csv`, `data/monthly_margin_pilot_vs_control.csv`
- `figures/` — charts used in the write-up

**All data is simulated** to test the method; real results depend on actual contracts and markets.
AI use: Claude (Anthropic) helped brainstorm and write the simulation/charts; framework and choices are my own.
