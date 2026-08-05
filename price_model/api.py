"""The two demand -> price functions.

    electricity_price(zone, demand, **context)   -> electricity price [EUR/MWh]
    hydrogen_price(zone, h2_demand, **context)   -> hydrogen price   [EUR/MWhH2]

Both take the zone and its **demand** as the only required inputs. Any supporting driver
(wind, solar, ... for electricity; electrolyser_gen, smr, ... for hydrogen) may be passed
as a keyword to override its default, which is that zone's median. Vectorised: pass array
-like ``demand`` to get an array of prices back.

Example::

    from price_model import electricity_price, hydrogen_price
    electricity_price("DE00", 55000)                 # price at 55 GW demand, median weather
    electricity_price("DE00", 55000, wind=2000)      # ... with low wind
    hydrogen_price("DE00", 1200)                      # H2 price at 1200 MWH2 demand
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from .config import COMMODITIES
from .multivariate import demand_only_row, predict as _predict

_OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


@lru_cache(maxsize=None)
def _bundle(commodity: str) -> dict:
    path = _OUTPUTS / COMMODITIES[commodity]["model"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found — run `train_model.py` first ({path})."
        )
    return joblib.load(path)


def available_zones(commodity: str) -> list[str]:
    """Zones with a trained model for this commodity."""
    return sorted(_bundle(commodity)["zones"])


def _price(commodity: str, zone: str, demand, context: dict):
    bundle = _bundle(commodity)
    if zone not in bundle["zones"]:
        raise KeyError(
            f"No {commodity} model for zone {zone!r}. "
            f"Available: {', '.join(available_zones(commodity))}"
        )
    entry = bundle["zones"][zone]
    features = entry.get("features", bundle["features"])
    demand_col = bundle["demand"]
    medians = entry["medians"]

    unknown = set(context) - set(features)
    if unknown:
        raise TypeError(f"Unknown feature(s) {sorted(unknown)}; valid: {features}")

    demand = np.atleast_1d(np.asarray(demand, dtype=float))
    n = len(demand)
    X = demand_only_row(features, demand_col, medians, demand, context)
    out = _predict(bundle, zone, X)
    return float(out[0]) if n == 1 else out


def electricity_price(zone: str, demand, **context):
    """Electricity price [EUR/MWh] for ``zone`` at the given electricity ``demand`` [MW]."""
    return _price("electricity", zone, demand, context)


def hydrogen_price(zone: str, h2_demand, **context):
    """Hydrogen price [EUR/MWhH2] for ``zone`` at the given hydrogen demand [MWH2]."""
    return _price("hydrogen", zone, h2_demand, context)
