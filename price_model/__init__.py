"""Demand -> price models for electricity and hydrogen (Project 3 dispatch output, NT2030).

Public API::

    from price_model import electricity_price, hydrogen_price
    electricity_price("DE00", 55000)   # EUR/MWh   at 55 GW electricity demand
    hydrogen_price("DE00", 1200)        # EUR/MWhH2 at 1200 MWH2 hydrogen demand
"""
from .api import electricity_price, hydrogen_price, available_zones
from .config import COMMODITIES
from .extract import extract_electricity, extract_hydrogen
from .multivariate import train_all, predict

__all__ = ["electricity_price", "hydrogen_price", "available_zones", "COMMODITIES",
           "extract_electricity", "extract_hydrogen", "train_all", "predict"]
