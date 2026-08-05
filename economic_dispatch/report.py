"""Extract, validate, and export the solved dispatch.

``validate`` independently recomputes both nodal balances from the solution
(pure numpy) and asserts the residuals are ~0 — the primary correctness check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import BuildResult
from . import data_loader as dl


def _sol(build: BuildResult, name: str) -> pd.DataFrame:
    """Return a variable's solution as a (dim0 x hour) DataFrame, or empty."""
    if name not in build.model.variables:
        return pd.DataFrame()
    da = build.model.solution[name]
    return da.to_pandas()


def _to_pandas(da) -> pd.DataFrame:
    """DataArray -> pandas, evaluating an unresolved linopy ``LinearExpression``
    first (``build.external_e``/``external_h2`` are one whenever the zone
    selection has real priced cross-border legs, rather than falling back to a
    plain zero DataArray -- ``LinearExpression`` has no ``to_pandas()`` of its
    own, only ``.solution``)."""
    if hasattr(da, "solution"):
        da = da.solution
    return da.to_pandas()


def extract(build: BuildResult) -> dict[str, pd.DataFrame]:
    return {
        "gen_p": _sol(build, "gen_p"),
        "dis": _sol(build, "dis"),
        "ch": _sol(build, "ch"),
        "soc": _sol(build, "soc"),
        "spill": _sol(build, "spill"),
        "ely_p": _sol(build, "ely_p"),
        "term_h2": _sol(build, "term_h2"),
        "shed_e": _sol(build, "shed_e"),
        "shed_h": _sol(build, "shed_h"),
        "dump_e": _sol(build, "dump_e"),
        "dump_h": _sol(build, "dump_h"),
        "smr_gen": _sol(build, "smr_gen"),
        "prod_wind_p": _sol(build, "prod_wind_p"),
        "prod_pv_p": _sol(build, "prod_pv_p"),
        "prod_batt_dis": _sol(build, "prod_batt_dis"),
        "prod_batt_ch": _sol(build, "prod_batt_ch"),
        "prod_ely_p": _sol(build, "prod_ely_p"),
        "prod_tank_dis": _sol(build, "prod_tank_dis"),
        "prod_tank_ch": _sol(build, "prod_tank_ch"),
        "prod_grid_net": _sol(build, "prod_grid_net"),
        "prod_h2_net": _sol(build, "prod_h2_net"),
        "prod_demand": _sol(build, "prod_demand"),
        "prod_ely_ren": _sol(build, "prod_ely_ren"),
        "prod_gc_buy": _sol(build, "prod_gc_buy"),
        "prod_gc_sell": _sol(build, "prod_gc_sell"),
    }


def _ids_on_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Orient so that 'zone|...' resource ids are the row index (linopy's
    to_pandas() dim order is not guaranteed)."""
    if any(isinstance(i, str) and "|" in i for i in df.index):
        return df
    if any(isinstance(c, str) and "|" in c for c in df.columns):
        return df.T
    return df


def _zones_on_rows(df: pd.DataFrame, zones: list[str]) -> pd.DataFrame:
    """Orient a (zone x hour) frame so zones are the row index."""
    if set(zones) & set(df.index):
        return df
    if set(zones) & set(df.columns):
        return df.T
    return df


def _zone_sum(df: pd.DataFrame, zones: list[str], H: int) -> pd.DataFrame:
    """Sum a (gen|sto id x hour) frame into (zone x hour), id = 'zone|...'.

    When the frame is empty (e.g. no storage) return a zero (zone x H) grid so
    it stays column-aligned with the other balance terms.
    """
    if df.empty:
        return pd.DataFrame(0.0, index=zones, columns=range(H))
    df = _ids_on_rows(df)
    grp = df.groupby(lambda gid: str(gid).split("|", 1)[0]).sum()
    return grp.reindex(zones).fillna(0.0)


def _sto_ids(build: BuildResult, carrier: str):
    """Storage device ids for a carrier ('electricity' / 'hydrogen')."""
    st = build.storage
    if st.empty or "carrier" not in st.columns:
        return None
    return list(st.index[st["carrier"] == carrier])


def _zone_sum_carrier(df: pd.DataFrame, zones: list[str], H: int, ids) -> pd.DataFrame:
    """Like _zone_sum but summing only the storage ids in ``ids``."""
    if df.empty:
        return pd.DataFrame(0.0, index=zones, columns=range(H))
    df = _ids_on_rows(df)
    if ids is not None:
        df = df.loc[df.index.intersection(ids)]
    if df.empty:
        return pd.DataFrame(0.0, index=zones, columns=range(H))
    grp = df.groupby(lambda gid: str(gid).split("|", 1)[0]).sum()
    return grp.reindex(zones).fillna(0.0)


def _prod_on_rows(df: pd.DataFrame, prod_idx) -> pd.DataFrame:
    """Orient a (country x hour) Hydrogen-Producer solution frame so country
    ids are the row index."""
    if set(prod_idx) & set(df.index):
        return df
    return df.T


def _prod_zone_sum(df: pd.DataFrame, build: BuildResult, zones: list[str], H: int) -> pd.DataFrame:
    """Sum a (country x hour) Hydrogen-Producer solution frame into (zone x
    hour) using ``build.h2_producer``'s country -> attachment-zone map. Zero
    (zone x H) if disabled/empty, same convention as ``_zone_sum``."""
    prod = build.h2_producer
    if df.empty or prod.empty:
        return pd.DataFrame(0.0, index=zones, columns=range(H))
    df = _prod_on_rows(df, prod.index)
    grp = df.groupby(lambda c: prod.loc[c, "zone"]).sum()
    return grp.reindex(zones).fillna(0.0)


def validate(build: BuildResult, tol: float = 1e-3) -> dict[str, float]:
    """Recompute elec & H2 balances from the solution; return max residuals."""
    z = build.zones
    sol = extract(build)
    H = len(build.hours)

    def zrows(name):
        df = sol[name]
        if df.empty:
            return pd.DataFrame(0.0, index=z, columns=range(H))
        return _zones_on_rows(df, z).reindex(z).fillna(0.0)

    eids, hids = _sto_ids(build, "electricity"), _sto_ids(build, "hydrogen")
    gen_z = _zone_sum(sol["gen_p"], z, H)
    dis_z = _zone_sum_carrier(sol["dis"], z, H, eids)
    ch_z = _zone_sum_carrier(sol["ch"], z, H, eids)
    dis_h2_z = _zone_sum_carrier(sol["dis"], z, H, hids)
    ch_h2_z = _zone_sum_carrier(sol["ch"], z, H, hids)
    ely = zrows("ely_p")
    term = zrows("term_h2")
    shed_e = zrows("shed_e")
    shed_h = zrows("shed_h")
    dump_e = zrows("dump_e")
    dump_h = zrows("dump_h")
    smr = zrows("smr_gen")

    demand_e = _zones_on_rows(_to_pandas(build.demand_e), z).reindex(z)
    demand_h = _zones_on_rows(_to_pandas(build.demand_h), z).reindex(z)
    external_e = _zones_on_rows(_to_pandas(build.external_e), z).reindex(z)
    external_h2 = _zones_on_rows(_to_pandas(build.external_h2), z).reindex(z)

    h2 = build.gens[build.gens["h2_fuel"]]
    h2_cons = pd.DataFrame(0.0, index=z, columns=range(H))
    if not h2.empty and not sol["gen_p"].empty:
        for gid, row in h2.iterrows():
            h2_cons.loc[row["zone"]] += sol["gen_p"].loc[gid] / row["eff"]

    net_e = _net_import_from_solution(build, "e")
    net_h = _net_import_from_solution(build, "h")
    prod_grid = _prod_zone_sum(sol["prod_grid_net"], build, z, H)
    prod_h2 = _prod_zone_sum(sol["prod_h2_net"], build, z, H)

    res_e = (gen_z + dis_z - ch_z - ely + net_e + external_e + shed_e - dump_e
             - demand_e + prod_grid)
    max_e = float(np.abs(res_e.to_numpy()).max()) if res_e.size else 0.0

    ely_prod = _ely_production(build, sol)
    res_h = (ely_prod + term + net_h + shed_h + external_h2 + dis_h2_z - ch_h2_z
             - dump_h - demand_h - h2_cons + prod_h2 + smr)
    max_h = float(np.abs(res_h.to_numpy()).max()) if res_h.size else 0.0

    return {"max_elec_residual": max_e, "max_h2_residual": max_h, "tol": tol}


def _ely_production(build: BuildResult, sol) -> pd.DataFrame:
    """H2 produced per zone = eff * elec consumed (electrolyser efficiency)."""
    z = build.zones
    ely = sol["ely_p"]
    ely = _zones_on_rows(ely, z).reindex(z).fillna(0.0) if not ely.empty \
        else pd.DataFrame(0.0, index=z, columns=range(len(build.hours)))
    eff = getattr(build, "_ely_eff", pd.Series(0.68, index=z)).reindex(z).fillna(0.68)
    return ely.mul(eff, axis=0)


def _lines_on_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Orient a flow frame so line ids (containing '->') are the row index."""
    if any(isinstance(i, str) and "->" in i for i in df.index):
        return df
    return df.T


def _net_import_from_solution(build: BuildResult, tag: str) -> pd.DataFrame:
    z = build.zones
    lines = build.elines if tag == "e" else build.hlines
    H = len(build.hours)
    out = pd.DataFrame(0.0, index=z, columns=range(H))
    if not lines:
        return out
    fpos = _lines_on_rows(build.model.solution[f"f{tag}_pos"].to_pandas())
    fneg = _lines_on_rows(build.model.solution[f"f{tag}_neg"].to_pandas())
    for i, l in enumerate(lines):
        key = f"{tag}{i}:{l.frm}->{l.to}"
        p = fpos.loc[key].to_numpy(dtype=float)
        n = fneg.loc[key].to_numpy(dtype=float)
        out.loc[l.to] = out.loc[l.to].to_numpy() + p * (1 - l.loss) - n
        out.loc[l.frm] = out.loc[l.frm].to_numpy() + n * (1 - l.loss) - p
    return out


def summary(build: BuildResult) -> dict[str, float]:
    sol = extract(build)
    cfg = build.cfg
    shed_e = sol["shed_e"].to_numpy().sum() if not sol["shed_e"].empty else 0.0
    shed_h = sol["shed_h"].to_numpy().sum() if not sol["shed_h"].empty else 0.0
    dump_e = sol["dump_e"].to_numpy().sum() if not sol["dump_e"].empty else 0.0
    dump_h = sol["dump_h"].to_numpy().sum() if not sol["dump_h"].empty else 0.0
    gen_total = sol["gen_p"].to_numpy().sum() if not sol["gen_p"].empty else 0.0
    ely_total = sol["ely_p"].to_numpy().sum() if not sol["ely_p"].empty else 0.0
    term_total = sol["term_h2"].to_numpy().sum() if not sol["term_h2"].empty else 0.0
    smr_total = sol["smr_gen"].to_numpy().sum() if not sol["smr_gen"].empty else 0.0

    gen_cost = 0.0
    if not sol["gen_p"].empty:
        mc = build.gens["mc"].reindex(sol["gen_p"].index).fillna(0.0)
        gen_cost = float((sol["gen_p"].mul(mc, axis=0)).to_numpy().sum())
    startup_cost = float(getattr(build, "startup_cost_eur", 0.0) or 0.0)
    obj = (gen_cost + cfg.h2_terminal_price * float(term_total)
           + cfg.voll_eur_per_mwh * float(shed_e + shed_h)
           + startup_cost)
    return {
        "objective_eur": obj,
        "generation_cost_eur": gen_cost,
        "startup_cost_eur": startup_cost,
        "total_generation_mwh": float(gen_total),
        "electrolyser_load_mwh": float(ely_total),
        "h2_terminal_import_mwh": float(term_total),
        "smr_production_mwh": float(smr_total),
        "elec_shed_mwh": float(shed_e),
        "h2_shed_mwh": float(shed_h),
        "elec_dumped_mwh": float(dump_e),
        "h2_dumped_mwh": float(dump_h),
        "h2_producer_renewable_h2_share": _prod_renewable_share(build, sol, cfg),
        "h2_producer_gc_purchased_mwh": _prod_gc_total(sol, "prod_gc_buy"),
        "h2_producer_gc_sold_mwh": _prod_gc_total(sol, "prod_gc_sell"),
    }


def _prod_gc_total(sol, key: str) -> float | None:
    """Total Green Certificate MWh bought/sold (``key``) across every
    Producer this run (§18.7). ``None`` if the Producer is disabled/unbuilt."""
    if sol["prod_demand"].empty:
        return None
    return float(sol[key].to_numpy().sum())


def _prod_renewable_share(build: BuildResult, sol, cfg) -> float | None:
    """Fraction of Hydrogen Producer downstream demand covered by renewable-
    attributed electrolyser output -- physically onsite (wind/PV, hourly-
    matched) plus Green-Certificate-backed grid draw (horizon-total, §18.7) --
    blended across every producer in this run. ``None`` if the Producer is
    disabled/unbuilt (no ``prod_demand`` variable at all)."""
    if sol["prod_demand"].empty:
        return None
    demand_total = float(sol["prod_demand"].to_numpy().sum())
    if demand_total <= 0.0:
        return None
    ren_mwh = float(sol["prod_ely_ren"].to_numpy().sum()) + float(sol["prod_gc_buy"].to_numpy().sum())
    ren_total = ren_mwh * cfg.h2_producer_electrolyser_efficiency
    return ren_total / demand_total


def write_outputs(build: BuildResult, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        try:
            old.unlink()
        except OSError as e:
            print(f"  warning: could not remove {old.name} ({e.strerror}); "
                  f"close it if open in another program")
    write_hourly_balance(build, out_dir)


def write_hourly_balance(build: BuildResult, out_dir: Path) -> None:
    """Write the hourly per-technology balance tables (elec & H2) to CSV."""
    write_balance_tables(hourly_balance_tables(build), Path(out_dir))


def write_balance_tables(tables: dict, out_dir: Path) -> None:
    """Clean-slate write the two balance tables to ``out_dir`` as CSVs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        try:
            old.unlink()
        except OSError as e:
            print(f"  warning: could not remove {old.name} ({e.strerror})")
    tables["elec"].to_csv(out_dir / "hourly_balance_elec.csv")
    tables["h2"].to_csv(out_dir / "hourly_balance_h2.csv")


def hourly_balance_tables(build: BuildResult) -> dict:
    """PLEXOS-style hourly per-technology balances as two wide DataFrames.

    Two-level column header ``(zone, category)``, one row per hour. Signs are
    chosen so supply is + and consumption - (the energy categories sum to ~0);
    a final ``Marginal Price (EUR/MWh)`` column carries the zonal price if
    computed. Returns ``{"elec": df, "h2": df}``.
    """
    z = build.zones
    H = len(build.hours)
    sol = extract(build)

    def zrows(name):
        df = sol[name]
        if df.empty:
            return pd.DataFrame(0.0, index=z, columns=range(H))
        return _zones_on_rows(df, z).reindex(z).fillna(0.0)

    def da_rows(da):
        return _zones_on_rows(_to_pandas(da), z).reindex(z).fillna(0.0)

    gp = _ids_on_rows(sol["gen_p"]) if not sol["gen_p"].empty else pd.DataFrame()
    dis_ids = _ids_on_rows(sol["dis"]) if not sol["dis"].empty else pd.DataFrame()
    ch_ids = _ids_on_rows(sol["ch"]) if not sol["ch"].empty else pd.DataFrame()

    def storage_kind_cols(zone: str, carrier: str):
        """Per-device discharge/charge columns, e.g. 'Hydro reservoir discharge (MW)'."""
        st = build.storage
        if st.empty:
            return []
        out = []
        rows = st[(st["zone"] == zone) & (st["carrier"] == carrier)]
        for sid, row in rows.iterrows():
            kind = row["kind"]
            dis_s = dis_ids.loc[sid].to_numpy(dtype=float) if sid in dis_ids.index else np.zeros(H)
            ch_s = ch_ids.loc[sid].to_numpy(dtype=float) if sid in ch_ids.index else np.zeros(H)
            out.append((f"{kind} discharge (MW)", dis_s))
            out.append((f"{kind} charge (-) (MW)", -ch_s))
        return out

    ely, term = zrows("ely_p"), zrows("term_h2")
    shed_e, shed_h = zrows("shed_e"), zrows("shed_h")
    dmp_e, dmp_h = zrows("dump_e"), zrows("dump_h")
    smr = zrows("smr_gen")
    net_e, net_h = _net_import_from_solution(build, "e"), _net_import_from_solution(build, "h")
    dem_e, dem_h = da_rows(build.demand_e), da_rows(build.demand_h)
    ext_e, ext_h = da_rows(build.external_e), da_rows(build.external_h2)
    ely_prod = _ely_production(build, sol)
    voll = build.cfg.voll_eur_per_mwh
    price_e = price_h = None
    if getattr(build, "price_e", None) is not None:
        price_e = da_rows(build.price_e).mask(
            (da_rows(build.price_e).abs() >= 0.99 * voll) & (shed_e.abs() <= 1e-6))
    if getattr(build, "price_h", None) is not None:
        price_h = da_rows(build.price_h).mask(
            (da_rows(build.price_h).abs() >= 0.99 * voll) & (shed_h.abs() <= 1e-6))

    h2 = build.gens[build.gens["h2_fuel"]]
    h2_cons = pd.DataFrame(0.0, index=z, columns=range(H))
    if not h2.empty and not gp.empty:
        for gid, row in h2.iterrows():
            if gid in gp.index:
                h2_cons.loc[row["zone"]] += gp.loc[gid].to_numpy() / row["eff"]

    prod_wind = _prod_zone_sum(sol["prod_wind_p"], build, z, H)
    prod_pv = _prod_zone_sum(sol["prod_pv_p"], build, z, H)
    prod_batt_dis = _prod_zone_sum(sol["prod_batt_dis"], build, z, H)
    prod_batt_ch = _prod_zone_sum(sol["prod_batt_ch"], build, z, H)
    prod_ely = _prod_zone_sum(sol["prod_ely_p"], build, z, H)
    prod_tank_dis = _prod_zone_sum(sol["prod_tank_dis"], build, z, H)
    prod_tank_ch = _prod_zone_sum(sol["prod_tank_ch"], build, z, H)
    prod_grid_net = _prod_zone_sum(sol["prod_grid_net"], build, z, H)
    prod_h2_net = _prod_zone_sum(sol["prod_h2_net"], build, z, H)
    prod_demand = _prod_zone_sum(sol["prod_demand"], build, z, H)
    prod_ely_h2 = prod_ely * build.cfg.h2_producer_electrolyser_efficiency

    def build_table(per_zone_cols):
        data = {}
        for zone in z:
            for cat, series in per_zone_cols(zone):
                data[(zone, cat)] = np.asarray(series, dtype=float)
        df = pd.DataFrame(data, index=pd.Index(range(H), name="hour"))
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["zone", "category"])
        return df.round(3)

    def elec_cols(zone):
        out = []
        if not gp.empty:
            for gid in gp.index:
                if gid.split("|", 1)[0] == zone:
                    out.append((gid.split("|", 1)[1], gp.loc[gid].to_numpy()))
        out += storage_kind_cols(zone, "electricity")
        out += [
            ("Electrolyser load (-)", -ely.loc[zone]),
            ("Net line import", net_e.loc[zone]),
            ("External exchange", ext_e.loc[zone]),
            ("Load shedding", shed_e.loc[zone]),
            ("Dumped/curtailed (-)", -dmp_e.loc[zone]),
            ("Demand (-)", -dem_e.loc[zone]),
            ("H2 Producer wind (MW)", prod_wind.loc[zone]),
            ("H2 Producer pv (MW)", prod_pv.loc[zone]),
            ("H2 Producer battery discharge (MW)", prod_batt_dis.loc[zone]),
            ("H2 Producer battery charge (-) (MW)", -prod_batt_ch.loc[zone]),
            ("H2 Producer electrolyser load (-) (MW)", -prod_ely.loc[zone]),
            ("H2 Producer grid exchange (MW)", prod_grid_net.loc[zone]),
        ]
        if price_e is not None:
            out.append(("Marginal Price (EUR/MWh)", price_e.loc[zone]))
        return out

    def h2_cols(zone):
        out = [
            ("Electrolyser production", ely_prod.loc[zone]),
            ("Terminal import", term.loc[zone]),
            ("SMR production", smr.loc[zone]),
            ("Net pipeline import", net_h.loc[zone]),
            ("External exchange", ext_h.loc[zone]),
        ]
        out += storage_kind_cols(zone, "hydrogen")
        out += [
            ("Load shedding", shed_h.loc[zone]),
            ("Dumped/curtailed (-)", -dmp_h.loc[zone]),
            ("H2 plant consumption (-)", -h2_cons.loc[zone]),
            ("Demand (-)", -dem_h.loc[zone]),
            ("H2 Producer electrolyser production (MW)", prod_ely_h2.loc[zone]),
            ("H2 Producer tank discharge (MW)", prod_tank_dis.loc[zone]),
            ("H2 Producer tank charge (-) (MW)", -prod_tank_ch.loc[zone]),
            ("H2 Producer downstream demand (-) (MW)", -prod_demand.loc[zone]),
            ("H2 Producer pipeline exchange (MW)", prod_h2_net.loc[zone]),
        ]
        if price_h is not None:
            out.append(("Marginal Price (EUR/MWh)", price_h.loc[zone]))
        return out

    return {"elec": build_table(elec_cols), "h2": build_table(h2_cols)}


def write_inputs(build: BuildResult, out_dir: Path) -> None:
    """Export the per-node input data exactly as the model resolved and used it.

    Writes, into ``out_dir/inputs/``:
      * nodes_generators.csv - every generation resource per node with resolved
        parameters (capacity, units, min/max per-unit power, marginal cost,
        efficiency, ramp, must-run, category, H2-fuel flag)
      * nodes_storage.csv    - storage devices per node (power, energy, efficiency)
      * network_lines.csv    - the elec & H2 lines actually used (endpoints, caps, loss)
      * nodes_summary.csv    - one row per node: demands, exchange, capacities by
        type, electrolyser/terminal capacity, resource counts
      * nodes_h2_producer.csv - one row per country with a Hydrogen Producer:
        attachment zone, assigned electrolyser capacity (rank-based) and the
        wind/PV/battery/H2-tank capacity derived from it (§18.2), the zone
        each of wind/PV actually draws its weather profile from
        (`wind_donor_zone`/`pv_donor_zone` -- usually the attachment zone
        itself, but not for BE/LU/NL, §18.2), and the flat downstream demand
        baseline derived from electrolyser capacity (§18.6) --
        absent if disabled
    """
    out_dir = Path(out_dir) / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()

    z = build.zones
    H = len(build.hours)
    g = build.gens

    g.reset_index().to_csv(out_dir / "nodes_generators.csv", index=False)

    if not build.storage.empty:
        build.storage.reset_index().to_csv(out_dir / "nodes_storage.csv", index=False)

    if not build.h2_producer.empty:
        build.h2_producer.reset_index().to_csv(out_dir / "nodes_h2_producer.csv", index=False)

    line_rows = []
    for carrier, lines in [("electricity", build.elines), ("hydrogen", build.hlines)]:
        for l in lines:
            line_rows.append(dict(carrier=carrier, frm=l.frm, to=l.to,
                                  cap_from_to_mw=l.cap_ft, cap_to_from_mw=l.cap_tf,
                                  loss_fraction=l.loss))
    pd.DataFrame(line_rows).to_csv(out_dir / "network_lines.csv", index=False)

    def cap_by(cat):
        return g[g["category"] == cat].groupby("zone")["pmax"].sum().reindex(z).fillna(0.0)

    def demand_sum(da):
        return _zones_on_rows(_to_pandas(da), z).reindex(z).fillna(0.0).sum(axis=1)

    sto = build.storage
    sto_e = (sto.groupby("zone")["ecap"].sum().reindex(z).fillna(0.0)
             if not sto.empty else pd.Series(0.0, index=z))
    sto_p = (sto.groupby("zone")["pdis"].sum().reindex(z).fillna(0.0)
             if not sto.empty else pd.Series(0.0, index=z))

    summ = pd.DataFrame(index=pd.Index(z, name="zone"))
    summ["elec_demand_mwh"] = demand_sum(build.demand_e)
    summ["h2_demand_mwh"] = demand_sum(build.demand_h)
    summ["ext_exchange_mwh"] = demand_sum(build.external_e)
    summ["h2_ext_exchange_mwh"] = demand_sum(build.external_h2)
    summ["committable_cap_mw"] = cap_by(dl.CAT_COMMIT)
    summ["vres_cap_mw"] = cap_by(dl.CAT_VRES)
    summ["ror_cap_mw"] = cap_by(dl.CAT_ROR)
    summ["profile_cap_mw"] = cap_by(dl.CAT_PROFILE)
    summ["electrolyser_cap_mw"] = getattr(build, "_ely_cap", pd.Series(0.0, index=z)).reindex(z).fillna(0.0)
    summ["h2_terminal_cap_mw"] = getattr(build, "_term_cap", pd.Series(0.0, index=z)).reindex(z).fillna(0.0)
    summ["storage_energy_mwh"] = sto_e
    summ["storage_power_mw"] = sto_p
    summ["n_generators"] = g.groupby("zone").size().reindex(z).fillna(0).astype(int)
    summ["n_committable"] = (g[g["category"] == dl.CAT_COMMIT].groupby("zone").size()
                             .reindex(z).fillna(0).astype(int))
    summ["n_storage"] = (sto.groupby("zone").size().reindex(z).fillna(0).astype(int)
                         if not sto.empty else 0)
    summ.round(3).to_csv(out_dir / "nodes_summary.csv")
