"""Build the electricity and hydrogen feature tables from Project 3's dispatch output.

Writes one parquet per commodity under ``inputs/`` (one row per zone-hour):

* ``inputs/elec_samples.parquet`` -- electricity price + demand + system drivers
* ``inputs/h2_samples.parquet``   -- hydrogen price + demand + supply mix

Source is ``inputs/hourly_balance_elec.csv`` / ``inputs/hourly_balance_h2.csv`` (Project
3's own LP dispatch-model output, committed here) plus ``inputs/networks_2030.parquet``
for zone adjacency (not derivable from the balance CSVs themselves) -- both are local
copies of Project 3's own exports, so this step needs no access to the Project 3 folder
itself. These sample tables are committed to the repo, so training normally skips this
step; run it only to regenerate them after a fresh Project 3 run (copy its refreshed
``outputs/hourly_balance_*.csv`` and ``inputs/networks_2030.parquet`` here first). See
``price_model/extract.py``.

Usage:
    ../projects-venv/Scripts/python.exe build_dataset.py
"""
from __future__ import annotations

import json
from pathlib import Path

from price_model.extract import (
    extract_electricity, extract_hydrogen, extract_adjacency,
    DEFAULT_ELEC_CSV, DEFAULT_H2_CSV, DEFAULT_NETWORKS_PARQUET,
)
from price_model.config import COMMODITIES

OUT = Path(__file__).resolve().parent / "inputs"


def _report(name, df, demand, price):
    feat = [c for c in df.columns if c not in ("zone", "hour", "datetime")]
    active = df[df[demand].fillna(0) > 0]
    print(
        f"[{name}] {len(df):,} rows | {df['zone'].nunique()} zones | "
        f"{df['hour'].nunique()} hours\n"
        f"    columns: {', '.join(feat)}\n"
        f"    {len(active):,} rows with {demand} > 0 | "
        f"price {df[price].min():.2f} .. {df[price].max():.2f}"
    )


def main(elec_csv=DEFAULT_ELEC_CSV, h2_csv=DEFAULT_H2_CSV,
         networks_parquet=DEFAULT_NETWORKS_PARQUET) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Reading {Path(elec_csv).name} / {Path(h2_csv).name} ...")

    elec = extract_electricity(elec_csv)
    elec.to_parquet(OUT / COMMODITIES["electricity"]["samples"], index=False,
                    compression="zstd")
    _report("electricity", elec, "demand", "price_eur_mwh")

    h2 = extract_hydrogen(h2_csv, elec_df=elec)
    h2.to_parquet(OUT / COMMODITIES["hydrogen"]["samples"], index=False,
                  compression="zstd")
    _report("hydrogen", h2, "h2_demand", "h2_price")

    if not Path(networks_parquet).exists():
        print(f"WARNING: {networks_parquet} not found -- skipping adjacency "
              f"(no neighbour/system-total features will be trained).")
        return

    elec_zones = set(elec["zone"])
    elec_adj = extract_adjacency(networks_parquet, "electricity", elec_zones)
    (OUT / COMMODITIES["electricity"]["adjacency"]).write_text(json.dumps(elec_adj, indent=1))
    print(f"[electricity] adjacency: {len(elec_adj)} zones -> "
          f"{COMMODITIES['electricity']['adjacency']}")

    h2_zones = set(h2["zone"])
    h2_adj = extract_adjacency(networks_parquet, "hydrogen", h2_zones)
    (OUT / COMMODITIES["hydrogen"]["adjacency"]).write_text(json.dumps(h2_adj, indent=1))
    print(f"[hydrogen] adjacency: {len(h2_adj)} zones -> "
          f"{COMMODITIES['hydrogen']['adjacency']}")


if __name__ == "__main__":
    main()
