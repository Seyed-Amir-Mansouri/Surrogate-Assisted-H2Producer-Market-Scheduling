"""Train the per-zone demand -> price models for both commodities and save them.

For each commodity in ``price_model/config.py`` reads its feature table and writes:

* ``outputs/<commodity>_model.joblib`` -- the trained per-zone model bundle
* ``outputs/<commodity>_metrics.csv``  -- per-zone CV R^2, RMSE, n, top features

Usage:
    ../projects-venv/Scripts/python.exe train_model.py                 # both commodities
    ../projects-venv/Scripts/python.exe train_model.py --only hydrogen
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from price_model.config import COMMODITIES
from price_model.multivariate import train_all
from price_model.neighbors import load_adjacency

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUT = ROOT / "outputs"


def _metrics_frame(bundle: dict) -> pd.DataFrame:
    rows = []
    for zone, e in bundle["zones"].items():
        top = sorted(e["importances"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        rows.append({
            "zone": zone,
            "top_features": ", ".join(f"{k} ({v:.1f})" for k, v in top),
        })
    return pd.DataFrame(rows).sort_values("zone")


def _summary(bundle: dict) -> str:
    r2 = np.array([e["cv_r2"] for e in bundle["zones"].values()])
    dor2 = np.array([e["demand_only_r2"] for e in bundle["zones"].values()])
    n = np.array([e["n"] for e in bundle["zones"].values()], float)
    return (f"{len(r2)} zones | mean CV R2 {r2.mean():.3f} | "
            f"sample-weighted R2 {np.average(r2, weights=n):.3f} | "
            f"median {np.median(r2):.3f} || "
            f"mean demand-only R2 {dor2.mean():.3f} "
            f"(median {np.median(dor2):.3f}) -- see README's 'Accuracy caveat'")


def train_commodity(name: str) -> None:
    cfg = COMMODITIES[name]
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUTS / cfg["samples"])
    adjacency = load_adjacency(INPUTS / cfg["adjacency"])
    bundle = train_all(df, name, cfg["target"], cfg["features"],
                       cfg["demand"], cfg["unit"], adjacency=adjacency,
                       net_demand_col=cfg.get("net_demand_col"),
                       max_price=cfg.get("max_price"))
    joblib.dump(bundle, OUT / cfg["model"])
    metrics = _metrics_frame(bundle)
    metrics.to_csv(OUT / cfg["metrics"], index=False)

    print(f"\n=== {name.upper()}  ({cfg['demand']} -> {cfg['target']}, {cfg['unit']}) ===")
    net = cfg.get("net_demand_col")
    extra = "+ per-zone neighbor_demand_*/neighbor_demand_total/demand_system_total"
    if net:
        extra += " AND neighbor_net_demand_*/neighbor_net_demand_total/net_demand_system_total"
    print("base features:", ", ".join(cfg["features"]), extra)
    print(_summary(bundle))
    print(f"wrote {cfg['model']}, {cfg['metrics']}")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(metrics.to_string(index=False))


def main(only: str | None = None) -> None:
    for name in ([only] if only else COMMODITIES):
        train_commodity(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(COMMODITIES), help="train just one commodity")
    args = ap.parse_args()
    main(args.only)
