"""Extract per-zone, per-hour feature tables from Project 3's dispatch-model output.

Source: ``inputs/hourly_balance_elec.csv`` / ``inputs/hourly_balance_h2.csv`` -- the two
"hourly per-technology balance" CSVs written by the sibling **Project 3** LP economic-
dispatch model (``run_dispatch.py``), *not* the raw ENTSO-E TYNDP PLEXOS workbook (that
xlsx-based pipeline was retired in favour of this one). Each file is a wide PLEXOS-style
table with a two-level ``(zone, category)`` column header and one row per hour:

* row 0 -> ``zone``     : the bidding-zone code, repeated once per category column
* row 1 -> ``category``  : technology / metric, e.g. ``"Gas (ccgt_new) (MW)"``,
  ``"Demand (-)"``, ``"Marginal Price (EUR/MWh)"``
* row 2                 : unused placeholder ("hour", blank cells) -- skipped
* rows 3..              : hourly values, col 0 = hour index (0..8735)

Project 3 signs each category so **supply is `+` and consumption is `-`** (see its
README, "Hourly per-technology balance"), and marks consumption categories with a
``(-)`` suffix in the name -- charge/discharge pairs (battery, hydro, H2 storage) are
already net-summable as-is, but single-purpose "magnitude" columns (``Demand (-)``,
``Dumped/curtailed (-)``) are stored *negative*, so those are negated back to a positive
magnitude to match this project's existing feature semantics (a positive ``demand``,
a positive ``dumped``).

Two sheets, two tables, columns grouped the same way as the (now-retired) PLEXOS
extraction (see ``price_model/config.py`` for which of these are actually used as model
features -- most of the rest, e.g. ``thermal``/``hydro``/``battery``, are kept only as
context/for parity):

**electricity** (``hourly_balance_elec.csv``)::

    price_eur_mwh   Marginal Price (EUR/MWh)                   <- target
    demand          -1 * Demand (-)                            <- primary input
    wind/solar      wind & solar generation                    } exogenous weather drivers
    vre             wind + solar                                (derived)
    residual_load   demand - vre                                (derived; key price driver)
    dsr/hydro/battery/balance/thermal/ens/dumped                dispatch outcomes (context)

**hydrogen** (``hourly_balance_h2.csv``)::

    h2_price          Marginal Price (EUR/MWh)                 <- target
    h2_demand         -1 * Demand (-)                           <- primary input
    electrolyser_gen  Electrolyser production                  } hydrogen supply mix
    smr               SMR production                           }
    storage           H2 storage discharge - H2 storage charge  }
    h2_net_trade       Terminal import + Net pipeline import + External exchange
    h2_plant_consumption / dumped / hns                        } net position / spill / unserved
    elec_price         same zone's own electricity price        } cross-commodity context

``Electrolyser load (-)`` (electricity) and every ``H2 Producer *`` column (the optional
``--h2-producer`` scenario add-on in Project 3) are intentionally dropped -- consistent
with this project's existing choice to keep electrolyser load out of the electricity
model's features. ``h2_net_trade`` no longer needs a separate crossborder-exchange sheet
merge: Project 3's own balance file already reports net trade per zone directly.
``elec_price`` no longer needs country-level demand-weighted aggregation either: Project
3's electricity and hydrogen zone codes are the *same* 20 CORE-region codes (a true 1:1
zone mapping), unlike the old PLEXOS workbook where several electricity bidding zones
shared one H2 country zone.

Both tables also carry ``month`` (1-12), ``season`` (0=DJF winter .. 3=SON autumn), and
the absolute ``hour`` (0-8735, unique per row) -- all derived from ``datetime``/row
position in ``_assemble``, same as before.

Zone adjacency (for ``price_model/neighbors.py``) is no longer derivable from these
balance CSVs -- they carry no directed link-flow sheet. Instead it's built from Project
3's own network topology, ``inputs/networks_2030.parquet`` (a local copy of Project 3's
own export -- see ``DEFAULT_NETWORKS_PARQUET``), which lists every ``(carrier, frm, to,
cap_from_to_mw, cap_to_from_mw)`` line; see ``extract_adjacency``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

INPUTS = Path(__file__).resolve().parent.parent / "inputs"
DEFAULT_ELEC_CSV = INPUTS / "hourly_balance_elec.csv"
DEFAULT_H2_CSV = INPUTS / "hourly_balance_h2.csv"
DEFAULT_NETWORKS_PARQUET = INPUTS / "networks_2030.parquet"

HEADER_ROWS = 3


def _classify_elec(cat: str):
    c = cat.strip()
    if c.startswith("Marginal Price"):        return ("price_eur_mwh", 1)
    if c.startswith("Demand"):                return ("demand", -1)
    if c.startswith("DSR"):                   return ("dsr", 1)
    if c.startswith("Wind"):                  return ("wind", 1)
    if c.startswith("Solar"):                 return ("solar", 1)
    if c.startswith("Hydro "):                return ("hydro", 1)
    if c.startswith("Battery"):               return ("battery", 1)
    if c.startswith(("Net line import", "External exchange")): return ("balance", 1)
    if c.startswith("Load shedding"):         return ("ens", 1)
    if c.startswith("Dumped"):                return ("dumped", -1)
    if c.startswith(("Nuclear", "Lignite", "Hard Coal", "Gas ", "Light Oil", "Heavy oil",
                     "Hydrogen (ccgt)", "Hydrogen (fc)", "Other Non-RES")):
        return ("thermal", 1)
    return (None, 0)


def _classify_h2(cat: str):
    c = cat.strip()
    if c.startswith("Marginal Price"):        return ("h2_price", 1)
    if c.startswith("Demand"):                return ("h2_demand", -1)
    if c.startswith("Electrolyser production"): return ("electrolyser_gen", 1)
    if c.startswith("SMR production"):        return ("smr", 1)
    if c.startswith("H2 storage"):            return ("storage", 1)
    if c.startswith(("Terminal import", "Net pipeline import", "External exchange")):
        return ("h2_net_trade", 1)
    if c.startswith("Load shedding"):         return ("hns", 1)
    if c.startswith("Dumped"):                return ("dumped", -1)
    if c.startswith("H2 plant consumption"):  return ("h2_plant_consumption", -1)
    return (None, 0)


ELEC_GROUPS = ["price_eur_mwh", "demand", "dsr", "wind", "solar", "hydro",
               "battery", "balance", "ens", "dumped", "thermal"]
H2_GROUPS = ["h2_price", "h2_demand", "electrolyser_gen", "smr", "storage",
             "h2_net_trade", "h2_plant_consumption", "hns", "dumped"]

PRICE_GROUPS = {"price_eur_mwh", "h2_price"}


def _read_balance_csv(csv_path: str | Path):
    """Return (zones, categories, values) for a ``hourly_balance_*.csv`` file.

    ``zones``/``categories`` are the per-column labels (length = ncols - 1, the leading
    ``zone``/``category`` label column dropped); ``values`` is the numeric hourly data,
    shape (nhours, ncols - 1), NaN preserved for blank cells (undefined/degenerate
    marginal prices).
    """
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        zones = next(r)[1:]
        cats = next(r)[1:]
    raw = pd.read_csv(csv_path, skiprows=HEADER_ROWS, header=None, dtype=float)
    values = raw.iloc[:, 1:].to_numpy(dtype=float)
    return zones, cats, values


def _aggregate(zones, cats, values, classify, groups):
    """Sum each zone's matching columns into the named ``groups`` (dict-of-dict form,
    matching what ``_assemble`` expects)."""
    zone_list = sorted(set(zones))
    nrows = values.shape[0]
    accum: dict[str, dict[str, np.ndarray]] = {
        z: {g: np.zeros(nrows) for g in groups} for z in zone_list
    }
    for j, (z, c) in enumerate(zip(zones, cats)):
        g, sign = classify(c)
        if g is None:
            continue
        col = values[:, j]
        if g in PRICE_GROUPS:
            accum[z][g] = col
        else:
            accum[z][g] += sign * np.nan_to_num(col, nan=0.0)
    return zone_list, nrows, accum


def _assemble(zones, nrows, accum, groups, year):
    dt = pd.date_range(f"{year}-01-01", periods=nrows, freq="h")
    month = dt.month.to_numpy()
    season = (dt.month.to_numpy() % 12) // 3
    frames = []
    for z in zones:
        d = {g: accum[z][g][:nrows] for g in groups}
        df = pd.DataFrame(d)
        df.insert(0, "zone", z)
        df.insert(1, "hour", np.arange(nrows, dtype="int32"))
        df.insert(2, "datetime", dt)
        df.insert(3, "month", month)
        df.insert(4, "season", season)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def extract_electricity(csv_path: str | Path = DEFAULT_ELEC_CSV, year: int = 2030) -> pd.DataFrame:
    """Return the per-(zone, hour) electricity feature table."""
    zones, cats, values = _read_balance_csv(csv_path)
    zone_list, nrows, accum = _aggregate(zones, cats, values, _classify_elec, ELEC_GROUPS)
    feat = _assemble(zone_list, nrows, accum, ELEC_GROUPS, year)
    feat["vre"] = feat["wind"] + feat["solar"]
    feat["residual_load"] = feat["demand"] - feat["vre"]
    return feat


def extract_hydrogen(csv_path: str | Path = DEFAULT_H2_CSV, year: int = 2030,
                      elec_df: pd.DataFrame | None = None,
                      elec_csv: str | Path = DEFAULT_ELEC_CSV) -> pd.DataFrame:
    """Return the per-(zone, hour) hydrogen feature table.

    Adds ``elec_price`` from ``elec_df`` (or a fresh ``extract_electricity`` pass) --
    a direct zone/hour merge, since Project 3's electricity and hydrogen zone codes are
    the same 20 CORE-region codes (unlike the old PLEXOS workbook's country-prefix
    aggregation).
    """
    zones, cats, values = _read_balance_csv(csv_path)
    zone_list, nrows, accum = _aggregate(zones, cats, values, _classify_h2, H2_GROUPS)
    feat = _assemble(zone_list, nrows, accum, H2_GROUPS, year)

    if elec_df is None:
        elec_df = extract_electricity(elec_csv, year)
    price_map = elec_df[["zone", "hour", "price_eur_mwh"]].rename(
        columns={"price_eur_mwh": "elec_price"})
    feat = feat.merge(price_map, on=["zone", "hour"], how="left")
    return feat


def extract_adjacency(parquet_path: str | Path, carrier: str,
                       zones: set[str] | None = None) -> dict[str, list[str]]:
    """Undirected zone adjacency for ``carrier`` ("electricity"/"hydrogen") from Project
    3's own network topology (``networks_2030.parquet``: ``carrier, frm, to,
    cap_from_to_mw, cap_to_from_mw, loss_fraction``).

    Only links with real capacity in at least one direction are kept; if ``zones`` is
    given, both endpoints must be in that set (drops links to zones outside the current
    dataset, e.g. neighbouring non-CORE countries the dispatch model prices but doesn't
    export a balance table for).
    """
    net = pd.read_parquet(parquet_path)
    net = net[net["carrier"] == carrier]
    net = net[(net["cap_from_to_mw"] > 0) | (net["cap_to_from_mw"] > 0)]
    if zones is not None:
        net = net[net["frm"].isin(zones) & net["to"].isin(zones)]
    adj: dict[str, set[str]] = {}
    for a, b in zip(net["frm"], net["to"]):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return {z: sorted(n) for z, n in adj.items()}


if __name__ == "__main__":
    e = extract_electricity()
    h = extract_hydrogen()
    print("electricity:", e.shape, e["zone"].nunique(), "zones")
    print("hydrogen:   ", h.shape, h["zone"].nunique(), "zones")
