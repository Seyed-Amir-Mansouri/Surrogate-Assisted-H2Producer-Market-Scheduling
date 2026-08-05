"""Configuration for the two commodity price models.

Each commodity maps a **demand** input to a **price** output, learned per bidding zone
from Project 3's own LP economic-dispatch output (``inputs/hourly_balance_{elec,h2}.csv``
-- the 20-zone Central-European CORE region, NT2030 scenario; see
``price_model/extract.py``):

* electricity -- ``hourly_balance_elec.csv``, price = Marginal Price (EUR/MWh)
* hydrogen    -- ``hourly_balance_h2.csv``,   price = Marginal Price (EUR/MWhH2)

``demand`` is the primary input; the remaining ``features`` are supporting context (they
default to each zone's median when a caller supplies only demand). ``target`` is the
price column the model predicts. ``adjacency`` names the JSON file (built by
``build_dataset.py`` from Project 3's own ``networks_2030.parquet`` line topology)
mapping each zone to its directly interconnected neighbours -- used to add per-zone
``neighbor_net_demand_<N>`` / ``neighbor_net_demand_total`` / ``net_demand_system_total``
features (see
``price_model/neighbors.py``). Because neighbour counts vary a lot per zone (median ~3,
up to 9 for a hub like DE00), each zone ends up with its own feature list, stored per
zone in the trained bundle rather than as one shared list.

``net_demand_col``, if set, both (a) is added to the zone's own base ``features`` as its
own net demand, and (b) is the column used for the neighbour/system-total *net* features
-- built *alongside* (not instead of) the raw-demand versions of each, since both are
wanted. "Net demand" means demand net of renewables (electricity's existing
``residual_load = demand - wind - solar``), not raw demand. Hydrogen has no wind/solar
equivalent tied to H2 zones, so it's left unset there -- only raw-demand neighbour/
system-total features exist for hydrogen, and there's no separate "own net demand" for
it either (h2_demand is the only demand quantity available).
"""
from __future__ import annotations

COMMODITIES = {
    "electricity": {
        "unit": "EUR/MWh",
        "demand": "demand",
        "target": "price_eur_mwh",
        "features": ["demand", "residual_load", "wind", "solar", "month", "season", "hour"],
        "samples": "elec_samples.parquet",
        "adjacency": "elec_adjacency.json",
        "net_demand_col": "residual_load",
        "max_price": 500,
        "model": "electricity_model.joblib",
        "metrics": "electricity_metrics.csv",
    },
    "hydrogen": {
        "unit": "EUR/MWhH2",
        "demand": "h2_demand",
        "target": "h2_price",
        "features": ["h2_demand", "smr", "month", "season", "hour"],
        "samples": "h2_samples.parquet",
        "adjacency": "h2_adjacency.json",
        "net_demand_col": None,
        "model": "hydrogen_model.joblib",
        "metrics": "hydrogen_metrics.csv",
    },
}
