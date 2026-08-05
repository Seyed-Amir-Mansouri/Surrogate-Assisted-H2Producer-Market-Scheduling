# Price Formation

Per-zone **demand → price** models for the Central-European CORE-region electricity and
hydrogen markets, trained on Project 3's own LP economic-dispatch output (NT2030
scenario). A separate gradient-boosted model is fit per bidding zone, so each zone gets
its own demand → price curve.

| Commodity | Demand input | Price output | Zones modelled |
|-----------|--------------|---------------|----------------|
| Electricity | `Demand (-)` | `Marginal Price (EUR/MWh)` | 16 |
| Hydrogen | `Demand (-)` | `Marginal Price (EUR/MWhH2)` | 13 |

(4 electricity / 7 hydrogen CORE-region zones are hub/offshore nodes with no native
demand of their own, so they have no trained model of their own — see
`price_model/extract.py`.)

## Features

- Python API — `electricity_price(zone, demand, **ctx)` / `hydrogen_price(zone, h2_demand, **ctx)`
- Demand is the only required argument; every other feature defaults to that zone's median

## Project structure

```
price_model/
  config.py        # commodities: target, demand, feature list, output filenames
  extract.py        # balance CSV -> tidy per-(zone, hour) feature table
  multivariate.py    # per-zone model training, CV scoring, predict()
  api.py             # electricity_price(), hydrogen_price(), available_zones()
  neighbors.py        # neighbour/system-total demand features

build_dataset.py    # hourly_balance_*.csv -> outputs/{elec,h2}_samples.parquet (maintainer-only)
train_model.py       # sample parquets -> outputs/*_model.joblib + *_metrics.csv

economic_dispatch/   # Project 3's dispatch-engine code, vendored locally (see CLAUDE.md)
optimize_h2_producer.py  # standalone H2 Producer LP, priced by this project's own
                      # trained proxy instead of Project 3's full network coupling
                      # (see Formulation.md §2)

inputs/              # sample parquets, hourly_balance_*.csv, adjacency JSONs, zones/
                      # networks parquet DBs (all committed, all local copies of Project 3
                      # exports — no dependency on a Project 3 checkout being present)
outputs/             # trained models + metrics (git-ignored)
```

## Accuracy caveat

The headline CV R² in `outputs/*_metrics.csv` (`cv_r2`) is scored with every feature at
its **real** historical value, including each zone's top-5 correlated-neighbour prices —
for most zones, a neighbour's own price is by far the strongest feature (permutation
importance 1–2, vs ~0.1 for everything else). But the documented API contract —
`electricity_price(zone, demand)` / `hydrogen_price(zone, h2_demand)`, demand as the only
required argument — defaults every feature *other* than demand, including those neighbour
prices, to that zone's training-set **median**. So a real demand-only call does not get to
use the feature that `cv_r2` is mostly measuring.

`outputs/*_metrics.csv` also reports `demand_only_r2`/`demand_only_rmse`: the same model,
evaluated the same way a real `electricity_price(zone, demand)` call would build its input
(see `price_model/multivariate.py::demand_only_row`, shared code with `api.py` so the two
never drift apart). Treat `demand_only_r2` as the accuracy a bare demand-only call actually
delivers; treat `cv_r2` as an upper bound reachable only if you also pass the real
neighbour prices (and other context) as keyword overrides, e.g.
`electricity_price("DE00", demand, price_LUG1=42.0)`.

## Installation

Dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### Python API

```python
from price_model import electricity_price, hydrogen_price, available_zones

electricity_price("DE00", 55000)              # price at 55 GW demand, median conditions
electricity_price("DE00", 55000, wind=20000)  # override a specific feature
electricity_price("DE00", [40000, 60000, 80000])  # vectorised over demand

hydrogen_price("AT00", 500)                   # H2 price at 500 MWH2 demand

available_zones("electricity")
available_zones("hydrogen")
```

### Training pipeline

```bash
# Train both models from the committed sample parquets
python train_model.py

# Retrain a single commodity
python train_model.py --only hydrogen
```

Regenerating the sample parquets (`build_dataset.py`) is a maintainer-only step: it reads
`inputs/hourly_balance_{elec,h2}.csv` and `inputs/networks_2030.parquet` — local copies of
Project 3's dispatch-model output. To refresh them, rerun Project 3's `run_dispatch.py`
for a fresh scenario and copy its `outputs/hourly_balance_*.csv` and
`inputs/networks_2030.parquet` here.
