"""Views for the price-proxy dashboard.

Lets the user pick zones (grouped by country), a day-of-year range, run our
trained electricity/hydrogen price proxies against real historical data for
that range, and -- for each country's H2-Producer host zone -- re-optimize
``optimize_h2_producer``'s Hydrogen Producer LP against the same proxy to get
grid/pipeline exchange and Green Certificate figures. Unlike Project 3's
``dispatcher`` app, every run is persisted (``RunResult``) so a country's most
recent results can be redisplayed instantly without recomputing anything --
see ``country_detail``.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from django.shortcuts import render

import optimize_h2_producer as h2p
from economic_dispatch.config import discover_zones
from price_model import api as price_api
from price_model.multivariate import predict as model_predict

from .models import RunResult

_COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CZ": "Czechia", "DE": "Germany",
    "FR": "France", "HR": "Croatia", "HU": "Hungary", "LU": "Luxembourg",
    "NL": "Netherlands", "PL": "Poland", "RO": "Romania", "SI": "Slovenia",
    "SK": "Slovakia",
}

DAY_MIN, DAY_MAX = 1, 364
DEFAULT_START_DAY, DEFAULT_END_DAY = 1, 7

_C_REAL = "#38bdf8"
_C_PRED = "#c084fc"
_C_OURS = "#8b5cf6"
_C_ACTUAL = "#22d3ee"
_C_GC_BUY = "#34d399"
_C_GC_SELL = "#fbbf24"

_PLOTLY_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family='"Segoe UI", system-ui, -apple-system, sans-serif', color="#cbd5e1", size=12.5),
    height=320, autosize=True, margin=dict(l=55, r=20, t=45, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _country_groups() -> list[dict]:
    """One entry per country, listing the zones that'll actually be used when
    that country is selected -- i.e. every zone of its with a trained
    electricity and/or hydrogen model (zones with neither, like DEKF/LUV1,
    are dropped here since selecting a country auto-includes only its usable
    zones -- see ``run``)."""
    elec_zones = set(price_api.available_zones("electricity"))
    h2_zones = set(price_api.available_zones("hydrogen"))
    groups: dict[str, list[str]] = {}
    for z in discover_zones():
        if z in elec_zones or z in h2_zones:
            groups.setdefault(z[:2], []).append(z)
    return [
        {
            "code": c, "name": _COUNTRY_NAMES.get(c, c),
            "zones": [
                {"code": z, "has_elec": z in elec_zones, "has_h2": z in h2_zones}
                for z in sorted(zs)
            ],
        }
        for c, zs in sorted(groups.items(), key=lambda kv: _COUNTRY_NAMES.get(kv[0], kv[0]))
    ]


def _index_context(**extra) -> dict:
    return {
        "country_groups": _country_groups(),
        "day_min": DAY_MIN, "day_max": DAY_MAX,
        "default_start_day": DEFAULT_START_DAY, "default_end_day": DEFAULT_END_DAY,
        **extra,
    }


def index(request):
    return render(request, "proxy_dashboard/index.html", _index_context())


def _metrics(real, pred) -> dict | None:
    """R^2 / RMSE / MAE / Pearson corr between real and predicted price."""
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(real) < 2:
        return None
    resid = real - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((real - real.mean()) ** 2))
    corr = float(np.corrcoef(real, pred)[0, 1]) if real.std() > 0 and pred.std() > 0 else None
    return {
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "corr": corr,
    }


def _price_series_for_zone(zone: str, start_day: int, end_day: int, edf, hdf) -> dict:
    """{'elec': {'hours', 'real', 'pred'}, 'h2': {...}} -- only for whichever
    commodity this zone actually has a trained model for. Real values come
    straight from the enriched samples (built from Project 3's own
    hourly_balance_*.csv), predicted from our trained proxy fed the same real
    historical context (neighbour prices/weather included -- see
    optimize_h2_producer.proxy_price_series's own docstring on why that's a
    legitimate backtest, not a forward what-if)."""
    hours = list(range((start_day - 1) * 24, end_day * 24))
    out: dict = {}

    if zone in price_api.available_zones("electricity"):
        bundle = price_api._bundle("electricity")
        feats = bundle["zones"][zone].get("features", bundle["features"])
        row = edf[(edf["zone"] == zone) & (edf["hour"].isin(hours))].sort_values("hour")
        pred = model_predict(bundle, zone, row[feats])
        out["elec"] = {
            "hours": row["hour"].tolist(),
            "real": row["price_eur_mwh"].astype(float).tolist(),
            "pred": np.asarray(pred, dtype=float).tolist(),
        }

    if zone in price_api.available_zones("hydrogen"):
        bundle = price_api._bundle("hydrogen")
        feats = bundle["zones"][zone].get("features", bundle["features"])
        row = hdf[(hdf["zone"] == zone) & (hdf["hour"].isin(hours))].sort_values("hour")
        pred = model_predict(bundle, zone, row[feats])
        out["h2"] = {
            "hours": row["hour"].tolist(),
            "real": row["h2_price"].astype(float).tolist(),
            "pred": np.asarray(pred, dtype=float).tolist(),
        }

    return out


def _exchange_summary(h2_zone: str, start_day: int, end_day: int) -> tuple[dict | None, str | None]:
    """Re-optimizes the Hydrogen Producer LP for this H2 host zone and compares
    it against the real historical schedule. Green Certificates have no
    historical ground truth in this project's data (Project 3's raw balance
    CSVs carry no GC columns) -- only the LP's own ('ours') bought/sold figures
    are reported for those two."""
    try:
        df = h2p.solve(h2_zone, start_day, end_day)
    except Exception as exc:
        return None, str(exc)

    actual = h2p.load_actual_schedule(h2_zone, start_day, end_day)
    grid_actual = float(np.nansum(actual["H2 Producer grid exchange (MW)"].to_numpy()))
    pipe_actual = float(np.nansum(actual["H2 Producer pipeline exchange (MW)"].to_numpy()))

    exchange = {
        "grid_mwh": {"ours": float(df["H2 Producer grid exchange (MW)"].sum()), "actual": grid_actual},
        "pipeline_mwh": {"ours": float(df["H2 Producer pipeline exchange (MW)"].sum()), "actual": pipe_actual},
        "gc_bought_mwh": {"ours": df.attrs["gc_buy_mwh"]},
        "gc_sold_mwh": {"ours": df.attrs["gc_sell_mwh"]},
        "objective_eur": df.attrs["objective"],
        "solve_seconds": round(df.attrs["solve_seconds"], 2),
    }
    return exchange, None


def run(request):
    """Selecting a country auto-includes every zone of its that carries a trained model --
    a country's electricity/hydrogen zones are a coupled pair (e.g. Belgium's BE00 elec /
    BEOF H2) that always get used together, so there's no reason to make the user pick
    zones individually."""
    if request.method != "POST":
        return index(request)

    valid_countries = {z[:2] for z in discover_zones()}
    countries = [c for c in request.POST.getlist("countries") if c in valid_countries]

    errors: list[str] = []
    if not countries:
        errors.append("Select at least one country.")
    try:
        start_day = int(request.POST.get("start_day", DEFAULT_START_DAY))
        end_day = int(request.POST.get("end_day", DEFAULT_END_DAY))
        if not (DAY_MIN <= start_day <= DAY_MAX and DAY_MIN <= end_day <= DAY_MAX):
            errors.append(f"Day range must be within {DAY_MIN}-{DAY_MAX}.")
        elif end_day < start_day:
            errors.append("End day must be on or after the start day.")
    except (TypeError, ValueError):
        errors.append("Enter valid start/end day numbers.")
        start_day, end_day = DEFAULT_START_DAY, DEFAULT_END_DAY

    if errors:
        return render(request, "proxy_dashboard/index.html", _index_context(
            errors=errors, selected_countries=countries, start_day=start_day, end_day=end_day,
        ))

    edf = h2p.enriched_elec_df()
    hdf = h2p.enriched_h2_df()

    elec_zones = set(price_api.available_zones("electricity"))
    h2_zones = set(price_api.available_zones("hydrogen"))
    all_zones = discover_zones()

    by_country = {
        c: sorted(z for z in all_zones if z[:2] == c and (z in elec_zones or z in h2_zones))
        for c in countries
    }

    country_results = []
    for country, country_zones in sorted(by_country.items()):
        elec_zone = h2_zone = None
        elec_metrics = h2_metrics = None
        price_series: dict = {}
        exchange = lp_error = None

        for zone in country_zones:
            series = _price_series_for_zone(zone, start_day, end_day, edf, hdf)
            if "elec" in series:
                elec_zone = zone
                elec_metrics = _metrics(series["elec"]["real"], series["elec"]["pred"])
                price_series["hours"] = series["elec"]["hours"]
                price_series["elec_real"] = series["elec"]["real"]
                price_series["elec_pred"] = series["elec"]["pred"]
            if "h2" in series:
                h2_zone = zone
                h2_metrics = _metrics(series["h2"]["real"], series["h2"]["pred"])
                price_series.setdefault("hours", series["h2"]["hours"])
                price_series["h2_real"] = series["h2"]["real"]
                price_series["h2_pred"] = series["h2"]["pred"]
                exchange, lp_error = _exchange_summary(zone, start_day, end_day)

        rr = RunResult.objects.create(
            country=country, elec_zone=elec_zone, h2_zone=h2_zone,
            start_day=start_day, end_day=end_day,
            elec_metrics=elec_metrics, h2_metrics=h2_metrics,
            price_series=price_series, exchange=exchange, lp_error=lp_error,
        )
        country_results.append(_country_view(rr))

    return render(request, "proxy_dashboard/results.html", {
        "country_results": country_results, "start_day": start_day, "end_day": end_day,
    })


def country_detail(request, country: str):
    country = country.upper()
    rr = RunResult.objects.filter(country=country).order_by("-created_at").first()
    if rr is None:
        return render(request, "proxy_dashboard/results.html", {
            "country_results": [], "not_found": country,
            "country_name": _COUNTRY_NAMES.get(country, country),
        })
    return render(request, "proxy_dashboard/results.html", {
        "country_results": [_country_view(rr)], "start_day": rr.start_day, "end_day": rr.end_day,
        "single_country": True,
    })


def _country_view(rr: RunResult) -> dict:
    """Chart HTML fragments + display data for one country's RunResult --
    shared by ``run`` (fresh data) and ``country_detail`` (stored data), so
    there's exactly one charting code path either way."""
    ps = rr.price_series or {}
    figs = {}
    if rr.elec_zone:
        figs["elec"] = _plot_price(
            ps.get("hours", []), ps.get("elec_real", []), ps.get("elec_pred", []),
            f"{rr.elec_zone} — Electricity price", "EUR/MWh",
        )
    if rr.h2_zone:
        figs["h2"] = _plot_price(
            ps.get("hours", []), ps.get("h2_real", []), ps.get("h2_pred", []),
            f"{rr.h2_zone} — Hydrogen price", "EUR/MWhH2",
        )
    if rr.exchange:
        figs["exchange"] = _plot_exchange_bars(rr.exchange)

    return {
        "run": rr,
        "country": rr.country,
        "country_name": _COUNTRY_NAMES.get(rr.country, rr.country),
        "figs": figs,
    }


def _plot_price(hours, real, pred, title, yaxis_title) -> str:
    fig = go.Figure()
    if real:
        fig.add_trace(go.Scatter(x=hours, y=real, mode="lines", name="Real",
                                 line=dict(color=_C_REAL, width=1.8)))
    if pred:
        fig.add_trace(go.Scatter(x=hours, y=pred, mode="lines", name="Our proxy",
                                 line=dict(color=_C_PRED, width=1.6, dash="dot")))
    fig.update_layout(title=title, xaxis_title="Hour of year", yaxis_title=yaxis_title, **_PLOTLY_LAYOUT)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def _plot_exchange_bars(exchange: dict) -> str:
    labels = ["Grid (power) exchange", "Pipeline (H2) exchange"]
    ours = [exchange["grid_mwh"]["ours"], exchange["pipeline_mwh"]["ours"]]
    actual = [exchange["grid_mwh"]["actual"], exchange["pipeline_mwh"]["actual"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=ours, name="Ours (LP-optimized)", marker_color=_C_OURS))
    fig.add_trace(go.Bar(x=labels, y=actual, name="Actual (historical)", marker_color=_C_ACTUAL))
    fig.add_trace(go.Bar(x=["Green Certificates bought"], y=[exchange["gc_bought_mwh"]["ours"]],
                         name="GC bought (ours)", marker_color=_C_GC_BUY, showlegend=True))
    fig.add_trace(go.Bar(x=["Green Certificates sold"], y=[exchange["gc_sold_mwh"]["ours"]],
                         name="GC sold (ours)", marker_color=_C_GC_SELL, showlegend=True))
    fig.update_layout(title="Exchange volumes over the horizon (MWh, net over the range)",
                      yaxis_title="MWh", barmode="group", **_PLOTLY_LAYOUT)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})
