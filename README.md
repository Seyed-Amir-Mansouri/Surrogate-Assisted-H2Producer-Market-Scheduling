# Price Formation

Per-zone **demand → price** models for the Central-European CORE-region electricity and
hydrogen markets, trained on Project 3's own LP economic-dispatch output (NT2030
scenario). Each bidding zone gets its own gradient-boosted demand → price curve.

| Commodity | Demand input | Price output | Zones modelled |
|-----------|--------------|---------------|----------------|
| Electricity | `Demand (-)` | `Marginal Price (EUR/MWh)` | 16 |
| Hydrogen | `Demand (-)` | `Marginal Price (EUR/MWhH2)` | 13 |

(Hub/offshore zones with no native demand have no trained model of their own.)

## Installation

```bash
pip install -r requirements.txt
```

## Python API

```python
from price_model import electricity_price, hydrogen_price, available_zones

electricity_price("DE00", 55000)              # price at 55 GW demand, median conditions
electricity_price("DE00", 55000, wind=20000)  # override a specific feature
electricity_price("DE00", [40000, 60000, 80000])  # vectorised over demand

hydrogen_price("AT00", 500)                   # H2 price at 500 MWH2 demand

available_zones("electricity")
available_zones("hydrogen")
```

Demand is the only required argument; every other feature defaults to that zone's median.

## Web user interface

A local Django dashboard (`webui/`) for exploring the models without writing code.

**Run it:**

```bash
app.bat
```

This creates a `.venv`, installs dependencies, applies migrations, and opens the
dashboard in your browser (`http://127.0.0.1:8000`).

**What it does:**

1. Pick one or more countries and a day-of-year range. Each country auto-selects its
   modelled zones (⚡ electricity, 🧪 hydrogen).
2. For each selected zone, it plots our predicted price against the real historical
   price over that range.
3. For each country's hydrogen zone, it re-solves a standalone Hydrogen Producer LP
   (`optimize_h2_producer.py`) priced off our model instead of the full network solve,
   and compares grid/pipeline exchange against the real historical schedule, plus the
   LP's own Green Certificate bought/sold figures (no historical GC data exists to
   compare against).
4. Every run is saved, so revisiting a country shows its last result instantly instead
   of re-solving.

## Training pipeline

```bash
python train_model.py               # train both commodities
python train_model.py --only hydrogen   # retrain just one
```

Training reads the committed sample parquets in `inputs/`. Regenerating those parquets
(`build_dataset.py`) is a maintainer-only step — see `CLAUDE.md`.

## Accuracy caveat

`outputs/*_metrics.csv`'s `cv_r2` is scored with every feature at its real value,
including each zone's most-correlated neighbour prices — usually the single strongest
feature. But `electricity_price(zone, demand)` defaults everything except demand to the
zone's median, so a real demand-only call can't use that feature. `demand_only_r2` in the
same CSV scores the model the way a real demand-only call actually queries it — treat
that number as the honest accuracy for this API, and `cv_r2` as an upper bound reachable
only by also passing real neighbour prices as keyword overrides.

## Project structure

```
price_model/          per-zone model training + prediction API
economic_dispatch/    Project 3's dispatch-engine code, vendored locally
optimize_h2_producer.py   standalone Hydrogen Producer LP, priced by our model
webui/                 Django dashboard
build_dataset.py       hourly_balance_*.csv -> sample parquets (maintainer-only)
train_model.py         sample parquets -> trained models + metrics
inputs/                sample parquets, balance CSVs, adjacency/zone/network data
outputs/               trained models + metrics (git-ignored, regenerate with train_model.py)
```
