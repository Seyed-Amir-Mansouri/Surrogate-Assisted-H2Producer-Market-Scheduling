"""Standalone Hydrogen Producer optimization for one zone/day-range, priced by our
trained proxy.

Reuses Project 3's own Hydrogen Producer physics -- sizing and internal
constraints, ``economic_dispatch/model.py::_build_h2_producer`` (vendored locally,
copied from Project 3) -- but does
NOT run Project 3's full joint network LP. Instead of coupling
``prod_grid_net``/``prod_h2_net`` into a real multi-zone balance, this treats the
host zone as an external price-taker market: the cost of every MWh imported
(or revenue of every MWh exported) is priced using OUR trained price_model
proxy (``electricity_price``/``hydrogen_price``), evaluated at that zone's own
REAL historical hourly conditions (demand, weather, and -- since this is a
backtest against already-realized history, not a forward what-if -- real
historical neighbour prices too). See Formulation.md SS2 for the full math.

Solves any (start_day, end_day) range for one zone -- a single day, or the full
364-day/8736-hour year (``--start-day 1 --end-day 364``), which scopes storage
cyclic-closure, downstream-demand conservation, and the RED III quota to the
same horizon Project 3's own full-year joint solve uses, removing the
single-day horizon-scope artifact documented in Formulation.md SS2.5. Training
always stays on the full year / all CORE zones regardless (Formulation.md SS1)
-- only solving is ever scoped down.

Usage:
    python optimize_h2_producer.py --zone DE00 --day 5                    # one day
    python optimize_h2_producer.py --zone DE00 --start-day 1 --end-day 364  # full year
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import linopy
from economic_dispatch.config import RunConfig, discover_zones
from economic_dispatch import model as p3model
from economic_dispatch import data_loader as p3dl

from price_model.multivariate import predict as model_predict
from price_model.neighbors import add_neighbor_features, add_candidate_neighbor_prices, load_adjacency
from price_model.extract import _read_balance_csv, DEFAULT_ELEC_CSV, DEFAULT_H2_CSV
from price_model import api as price_api

HOURS_PER_DAY = 24

_ELEC_ZONE_OVERRIDE = {"BEOF": "BE00", "NLLL": "NL00"}

_ELEC_PRODUCER_COLS = ["H2 Producer wind (MW)", "H2 Producer pv (MW)",
                      "H2 Producer battery discharge (MW)", "H2 Producer battery charge (-) (MW)",
                      "H2 Producer electrolyser load (-) (MW)", "H2 Producer grid exchange (MW)"]
_H2_PRODUCER_COLS = ["H2 Producer electrolyser production (MW)", "H2 Producer tank discharge (MW)",
                    "H2 Producer tank charge (-) (MW)", "H2 Producer downstream demand (-) (MW)",
                    "H2 Producer pipeline exchange (MW)"]


def load_actual_schedule(zone: str, start_day: int, end_day: int) -> pd.DataFrame:
    """Actual H2 Producer schedule for this zone/day-range straight from Project 3's
    real full-year joint solve (``inputs/hourly_balance_{elec,h2}.csv``'s
    "H2 Producer *" columns)."""
    hours = list(range((start_day - 1) * HOURS_PER_DAY, end_day * HOURS_PER_DAY))
    ezones, ecats, evals = _read_balance_csv(DEFAULT_ELEC_CSV)
    hzones, hcats, hvals = _read_balance_csv(DEFAULT_H2_CSV)

    def col(zones, cats, vals, cat):
        idx = [i for i, (z, c) in enumerate(zip(zones, cats)) if z == zone and c == cat]
        if not idx:
            return np.full(len(hours), np.nan)
        return vals[hours, idx[0]]

    data = {"hour": list(range(len(hours)))}
    for c in _ELEC_PRODUCER_COLS:
        data[c] = col(ezones, ecats, evals, c)
    for c in _H2_PRODUCER_COLS:
        data[c] = col(hzones, hcats, hvals, c)
    return pd.DataFrame(data)


def _run_config(zone: str, start_day: int, end_day: int) -> RunConfig:
    """``data_dir``/``output_dir``/``exports_dir`` default to
    ``economic_dispatch.config``'s own ``PROJECT_ROOT``-relative paths -- unused anyway
    by the functions this module calls, which only ever touch ``zones_db``/``networks_db``
    (see module docstring: no XLSXs needed)."""
    return RunConfig(
        zones=[zone], start_day=start_day, end_day=end_day,
        zones_db=ROOT / "inputs" / "zones_2030.parquet",
        networks_db=ROOT / "inputs" / "networks_2030.parquet",
        enable_h2_producer=True,
    )


def _donor_zone(country: str, host_zone: str, resource_idx: int, cap_key: str,
                profile_info: dict, zdata: dict, all_zones: list[str]) -> str:
    """Mirrors model.py::_build_h2_producer's nested ``_donor_zone`` (not importable
    directly since it's a closure). Falls back to a same-country sibling zone with
    real profile data and the largest installed capacity for that tech, if the host
    zone's own profile is all-zero (e.g. BEOF/LUB1/NLLL H2-hub nodes)."""
    def has_data(z):
        return profile_info.get(z, (False, 0.0, False, 0.0))[resource_idx * 2]
    if has_data(host_zone):
        return host_zone
    candidates = [z for z in all_zones if z[:2] == country and z != host_zone and has_data(z)]
    if not candidates:
        return host_zone
    return max(candidates, key=lambda z: zdata[z].capacities.get(cap_key, 0.0))


def sizing_and_profiles(zone: str, start_day: int, end_day: int):
    """Resolve this country's Producer sizing + this horizon's wind/PV availability
    upper bounds, exactly as Project 3's own model would for the same day range."""
    cfg = _run_config(zone, start_day, end_day)
    country = zone[:2]
    sizing = p3model._h2_producer_sizing(cfg)[country]
    host_zone = p3model._h2_main_zones(cfg)[country]
    if host_zone != zone:
        raise ValueError(f"{zone} is not {country}'s main H2 zone (that's {host_zone}); "
                         f"pass the main H2 zone instead.")

    profile_info = p3model._h2_producer_renewable_profile_info(str(cfg.zones_db))
    all_zones = discover_zones(cfg.zones_db)

    wind_donor = _donor_zone(country, host_zone, 0, "Wind (onshore) (MW)", profile_info, {}, all_zones)
    pv_donor = _donor_zone(country, host_zone, 1, "Solar (MW)", profile_info, {}, all_zones)
    needed_zones = sorted({host_zone, wind_donor, pv_donor})

    h0, h1 = cfg.hour_slice()
    zdata = p3dl.load_zones_from_db(needed_zones, cfg.zones_db, h0, h1)
    wind_donor = _donor_zone(country, host_zone, 0, "Wind (onshore) (MW)", profile_info, zdata, all_zones)
    pv_donor = _donor_zone(country, host_zone, 1, "Solar (MW)", profile_info, zdata, all_zones)
    if {wind_donor, pv_donor} - set(zdata):
        zdata = p3dl.load_zones_from_db(sorted({host_zone, wind_donor, pv_donor}), cfg.zones_db, h0, h1)

    wind_max = max(profile_info.get(wind_donor, (False, 0.0, False, 0.0))[1], 1e-9)
    pv_max = max(profile_info.get(pv_donor, (False, 0.0, False, 0.0))[3], 1e-9)
    wind_cf = np.clip(zdata[wind_donor].profiles["Wind_Onshore Profile"].to_numpy(dtype=float), 0.0, None)
    pv_cf = np.clip(zdata[pv_donor].profiles["Solar Profile"].to_numpy(dtype=float), 0.0, None)
    wind_cf_norm = np.clip(wind_cf / wind_max, 0.0, 1.0)
    pv_cf_norm = np.clip(pv_cf / pv_max, 0.0, 1.0)

    wind_upper = wind_cf_norm * sizing["wind_mw"]
    pv_upper = pv_cf_norm * sizing["pv_mw"]
    return cfg, sizing, host_zone, wind_donor, pv_donor, wind_upper, pv_upper


def enriched_elec_df() -> pd.DataFrame:
    """Full elec_samples.parquet enriched with neighbour/candidate-price columns --
    the expensive step in proxy_price_series, cached here so a multi-zone caller
    (e.g. exporting every country) only pays it once instead of once per zone."""
    ROOT_IN = ROOT / "inputs"
    edf = pd.read_parquet(ROOT_IN / "elec_samples.parquet")
    eadj = load_adjacency(ROOT_IN / "elec_adjacency.json")
    edf, _ = add_neighbor_features(edf, "demand", eadj, "residual_load")
    edf, _ = add_candidate_neighbor_prices(edf, "price_eur_mwh", eadj)
    return edf


def enriched_h2_df() -> pd.DataFrame:
    """Full h2_samples.parquet enriched with neighbour/candidate-price columns --
    see enriched_elec_df."""
    ROOT_IN = ROOT / "inputs"
    hdf = pd.read_parquet(ROOT_IN / "h2_samples.parquet")
    hadj = load_adjacency(ROOT_IN / "h2_adjacency.json")
    hdf, _ = add_neighbor_features(hdf, "h2_demand", hadj, None)
    hdf, _ = add_candidate_neighbor_prices(hdf, "h2_price", hadj)
    return hdf


def proxy_price_series(elec_zone: str, start_day: int, end_day: int, h2_zone: str | None = None,
                       edf: pd.DataFrame | None = None, hdf: pd.DataFrame | None = None,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """This horizon's hourly (elec_price, h2_price) as predicted by OUR trained proxy,
    fed real historical feature values (including real neighbour prices -- legitimate
    here since this is a backtest over already-realized history, not a forward
    what-if scenario -- see the price-taker/circularity discussion in Formulation.md
    SS2). ``h2_zone`` defaults to ``elec_zone`` (true for 11 of 13 countries); pass it
    explicitly for the 2 whose H2-hub zone has no electricity demand of its own and so
    no trained electricity model (Belgium: BE00 elec / BEOF H2; Netherlands: NL00 elec
    / NLLL H2). Pass ``edf``/``hdf`` (from enriched_elec_df/enriched_h2_df) to reuse
    already-enriched frames across many zones instead of re-enriching per call."""
    if h2_zone is None:
        h2_zone = elec_zone
    H = (end_day - start_day + 1) * HOURS_PER_DAY
    hours = list(range((start_day - 1) * HOURS_PER_DAY, end_day * HOURS_PER_DAY))

    elec_bundle = price_api._bundle("electricity")
    h2_bundle = price_api._bundle("hydrogen")

    if edf is None:
        edf = enriched_elec_df()
    erow = edf[(edf["zone"] == elec_zone) & (edf["hour"].isin(hours))].sort_values("hour")
    efeats = elec_bundle["zones"][elec_zone]["features"]
    p_elec = model_predict(elec_bundle, elec_zone, erow[efeats])

    if hdf is None:
        hdf = enriched_h2_df()
    hrow = hdf[(hdf["zone"] == h2_zone) & (hdf["hour"].isin(hours))].sort_values("hour")
    hfeats = h2_bundle["zones"][h2_zone]["features"]
    p_h2 = model_predict(h2_bundle, h2_zone, hrow[hfeats])

    assert len(p_elec) == H and len(p_h2) == H, f"expected {H} hours, got {len(p_elec)}/{len(p_h2)}"
    return np.asarray(p_elec, dtype=float), np.asarray(p_h2, dtype=float)


def solve(zone: str, start_day: int, end_day: int | None = None, fix_storage: bool = False,
         elec_zone: str | None = None) -> pd.DataFrame:
    """``fix_storage=True`` fixes battery and H2-tank charge/discharge to their ACTUAL
    historical values (Project 3's real full-year joint solve) instead of leaving them
    as free decision variables -- everything else (electrolyser, grid/pipeline exchange,
    downstream demand) still re-optimizes against the proxy price signal. Diagnostic:
    isolates whether storage-arbitrage error (the weakest-fitting variables, see
    Formulation.md SS3.4) is bleeding into grid/pipeline exchange and electrolyser
    dispatch through the balance equations, versus those being wrong on their own.

    ``elec_zone`` overrides which zone's electricity price proxy prices the grid
    connection -- defaults to ``_ELEC_ZONE_OVERRIDE.get(zone, zone)`` (only BEOF/NLLL
    differ from their own zone; see ``proxy_price_series``'s docstring).

    Under ``fix_storage``, the battery/tank SoC bounds get a small slack (``soc_slack``/
    ``tank_slack``): the fixed dis/ch replay real values rounded to 3 decimals in the
    source CSV, and drift accumulated over 8736 hours can push the recomputed SoC a hair
    outside [0, cap] (observed: ~0.03 MWh battery, ~0 tank) -- widening the bound by that
    same slack keeps an exact-bound LP infeasibility from blocking an otherwise-sound
    diagnostic. For the same reason, the cyclic-storage closure constraint is skipped
    under ``fix_storage``: dis/ch are already the real (already-cyclic, up to rounding)
    historical trajectory, so re-imposing our own closure on top of that fixed sequence
    is redundant and can be marginally infeasible for reasons unrelated to what's being
    tested."""
    if end_day is None:
        end_day = start_day
    t0 = time.time()
    cfg, s, host_zone, wind_donor, pv_donor, wind_upper, pv_upper = sizing_and_profiles(zone, start_day, end_day)
    p_elec, p_h2 = proxy_price_series(elec_zone or _ELEC_ZONE_OVERRIDE.get(zone, zone), start_day, end_day,
                                      h2_zone=zone)
    H = (end_day - start_day + 1) * HOURS_PER_DAY
    hours = pd.RangeIndex(H, name="hour")

    ely_mw, batt_mw, batt_mwh = s["electrolyser_mw"], s["battery_mw"], s["battery_mwh"]
    tank_mw, tank_mwh = s["tank_mw"], s["tank_mwh"]
    ely_eff = cfg.h2_producer_electrolyser_efficiency
    capacity_h2 = ely_mw * ely_eff
    baseline = cfg.h2_producer_downstream_demand_pct_of_electrolyser_capacity * capacity_h2
    flex = cfg.h2_producer_demand_flex_pct * capacity_h2
    grid_cap, h2_cap = cfg.h2_producer_grid_connection_mw, cfg.h2_producer_h2_connection_mw
    batt_eff, tank_eff = cfg.h2_producer_battery_efficiency, cfg.h2_producer_tank_efficiency
    quota = max(cfg.h2_producer_renewable_h2_quota, 0.0)
    gc_price = cfg.h2_producer_gc_price_eur_per_mwh
    sto_cost = cfg.storage_op_cost_eur_per_mwh
    batt_soc0 = cfg.initial_soc_fraction * batt_mwh
    tank_soc0 = cfg.initial_soc_fraction * tank_mwh

    if fix_storage:
        act = load_actual_schedule(zone, start_day, end_day)
        fixed_batt_dis = np.nan_to_num(act["H2 Producer battery discharge (MW)"].to_numpy(), nan=0.0)
        fixed_batt_ch = np.nan_to_num(-act["H2 Producer battery charge (-) (MW)"].to_numpy(), nan=0.0)
        fixed_tank_dis = np.nan_to_num(act["H2 Producer tank discharge (MW)"].to_numpy(), nan=0.0)
        fixed_tank_ch = np.nan_to_num(-act["H2 Producer tank charge (-) (MW)"].to_numpy(), nan=0.0)

    m = linopy.Model()
    wind_p = m.add_variables(lower=0.0, upper=wind_upper, coords=[hours], name="wind_p")
    pv_p = m.add_variables(lower=0.0, upper=pv_upper, coords=[hours], name="pv_p")
    if fix_storage:
        batt_dis = m.add_variables(lower=fixed_batt_dis, upper=fixed_batt_dis, coords=[hours], name="batt_dis")
        batt_ch = m.add_variables(lower=fixed_batt_ch, upper=fixed_batt_ch, coords=[hours], name="batt_ch")
        tank_dis = m.add_variables(lower=fixed_tank_dis, upper=fixed_tank_dis, coords=[hours], name="tank_dis")
        tank_ch = m.add_variables(lower=fixed_tank_ch, upper=fixed_tank_ch, coords=[hours], name="tank_ch")
    else:
        batt_dis = m.add_variables(lower=0.0, upper=batt_mw, coords=[hours], name="batt_dis")
        batt_ch = m.add_variables(lower=0.0, upper=batt_mw, coords=[hours], name="batt_ch")
        tank_dis = m.add_variables(lower=0.0, upper=tank_mw, coords=[hours], name="tank_dis")
        tank_ch = m.add_variables(lower=0.0, upper=tank_mw, coords=[hours], name="tank_ch")
    soc_slack = max(1.0, 0.02 * batt_mwh) if fix_storage else 0.0
    tank_slack = max(1.0, 0.02 * tank_mwh) if fix_storage else 0.0
    batt_soc = m.add_variables(lower=-soc_slack, upper=batt_mwh + soc_slack, coords=[hours], name="batt_soc")
    ely_p = m.add_variables(lower=0.0, upper=ely_mw, coords=[hours], name="ely_p")
    tank_soc = m.add_variables(lower=-tank_slack, upper=tank_mwh + tank_slack, coords=[hours], name="tank_soc")
    x_grid = m.add_variables(lower=-grid_cap, upper=grid_cap, coords=[hours], name="x_grid")
    x_h2 = m.add_variables(lower=-h2_cap, upper=h2_cap, coords=[hours], name="x_h2")
    demand = m.add_variables(lower=baseline - flex, upper=baseline + flex, coords=[hours], name="demand")
    ely_ren = m.add_variables(lower=0.0, coords=[hours], name="ely_ren")
    gc_buy = m.add_variables(lower=0.0, name="gc_buy")
    gc_sell = m.add_variables(lower=0.0, name="gc_sell")

    m.add_constraints(wind_p + pv_p + batt_dis - batt_ch - ely_p - x_grid == 0, name="elec_balance")
    m.add_constraints(ely_eff * ely_p + tank_dis - tank_ch - demand - x_h2 == 0, name="h2_balance")
    m.add_constraints(ely_ren <= wind_p + pv_p, name="ely_ren_cap_avail")
    m.add_constraints(ely_ren <= ely_p, name="ely_ren_cap_ely")
    m.add_constraints(gc_buy <= ely_p.sum() - ely_ren.sum(), name="gc_buy_cap")
    m.add_constraints(gc_sell <= (wind_p + pv_p).sum() - ely_ren.sum(), name="gc_sell_cap")

    m.add_constraints(batt_soc.isel(hour=0) - batt_soc0 - batt_eff * batt_ch.isel(hour=0) + batt_dis.isel(hour=0) == 0,
                      name="batt_balance_0")
    m.add_constraints(batt_soc.isel(hour=slice(1, None)) - batt_soc.isel(hour=slice(None, -1))
                      - batt_eff * batt_ch.isel(hour=slice(1, None)) + batt_dis.isel(hour=slice(1, None)) == 0,
                      name="batt_balance")
    m.add_constraints(tank_soc.isel(hour=0) - tank_soc0 - tank_eff * tank_ch.isel(hour=0) + tank_dis.isel(hour=0) == 0,
                      name="tank_balance_0")
    m.add_constraints(tank_soc.isel(hour=slice(1, None)) - tank_soc.isel(hour=slice(None, -1))
                      - tank_eff * tank_ch.isel(hour=slice(1, None)) + tank_dis.isel(hour=slice(1, None)) == 0,
                      name="tank_balance")
    if cfg.cyclic_storage and not fix_storage:
        m.add_constraints(batt_soc.isel(hour=-1) >= batt_soc0, name="batt_cyclic")
        m.add_constraints(tank_soc.isel(hour=-1) >= tank_soc0, name="tank_cyclic")

    m.add_constraints(demand.sum() == baseline * H, name="demand_conservation")
    if quota > 0.0:
        m.add_constraints(ely_eff * (ely_ren.sum() + gc_buy) >= quota * demand.sum(), name="red3_quota")

    cost = (-(p_elec * x_grid).sum() - (p_h2 * x_h2).sum()
           + sto_cost * (batt_ch.sum() + batt_dis.sum() + tank_ch.sum() + tank_dis.sum())
           + gc_price * gc_buy - gc_price * gc_sell)
    m.add_objective(cost)

    build_s = time.time() - t0
    t1 = time.time()
    status, condition = m.solve(solver_name="highs")
    solve_s = time.time() - t1
    if status != "ok":
        raise RuntimeError(f"solve failed: {status}/{condition}")

    sol = m.solution
    out = pd.DataFrame({
        "hour": hours,
        "H2 Producer wind (MW)": sol["wind_p"].values,
        "H2 Producer pv (MW)": sol["pv_p"].values,
        "H2 Producer battery discharge (MW)": sol["batt_dis"].values,
        "H2 Producer battery charge (-) (MW)": -sol["batt_ch"].values,
        "H2 Producer electrolyser load (-) (MW)": -sol["ely_p"].values,
        "H2 Producer grid exchange (MW)": sol["x_grid"].values,
        "H2 Producer electrolyser production (MW)": ely_eff * sol["ely_p"].values,
        "H2 Producer tank discharge (MW)": sol["tank_dis"].values,
        "H2 Producer tank charge (-) (MW)": -sol["tank_ch"].values,
        "H2 Producer downstream demand (-) (MW)": -sol["demand"].values,
        "H2 Producer pipeline exchange (MW)": sol["x_h2"].values,
    })
    out.attrs["objective"] = float(m.objective.value)
    out.attrs["gc_buy_mwh"] = float(sol["gc_buy"].values)
    out.attrs["gc_sell_mwh"] = float(sol["gc_sell"].values)
    out.attrs["p_elec"] = p_elec
    out.attrs["p_h2"] = p_h2
    out.attrs["host_zone"] = host_zone
    out.attrs["sizing"] = s
    out.attrs["build_seconds"] = build_s
    out.attrs["solve_seconds"] = solve_s
    out.attrs["n_hours"] = H
    out.attrs["fix_storage"] = fix_storage
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DE00")
    ap.add_argument("--day", type=int, default=None, help="single day (shorthand for --start-day/--end-day)")
    ap.add_argument("--start-day", type=int, default=5)
    ap.add_argument("--end-day", type=int, default=None)
    ap.add_argument("--fix-storage", action="store_true",
                    help="fix battery/tank charge-discharge to actual historical values")
    args = ap.parse_args()
    if args.day is not None:
        start_day, end_day = args.day, args.day
    else:
        start_day, end_day = args.start_day, (args.end_day or args.start_day)
    df = solve(args.zone, start_day, end_day, fix_storage=args.fix_storage)
    if df.attrs["n_hours"] <= 48:
        print(df.to_string(index=False))
    else:
        print(df.describe().to_string())
    print(f"\nhours solved: {df.attrs['n_hours']} | build: {df.attrs['build_seconds']:.1f}s | "
         f"solve: {df.attrs['solve_seconds']:.1f}s")
    print("objective (EUR, negative = net revenue):", round(df.attrs["objective"], 2))
