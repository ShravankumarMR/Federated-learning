"""Centralised raw-CSV loaders for IEEE-CIS and PaySim transaction data.

All agents that need to read raw transaction CSVs import from here so
there is a single source of truth for how each dataset is loaded and
what schema transformations are applied.

Public API
----------
load_ieee_cis_transactions  — returns a DataFrame with ``source_account``,
                              ``target_account``, ``timestamp`` columns derived
                              from the original card/address fields.
load_paysim_transactions    — returns a DataFrame with the same canonical
                              columns mapped from PaySim's ``nameOrig`` /
                              ``nameDest`` / ``step`` fields.
preprocess_transactions     — handles missing values and normalises numeric
                              columns; used by graph-based agents before
                              building PyG Data objects.
"""
from __future__ import annotations

from app.data_engineering.features.transaction_graph import (
    load_ieee_cis_transactions,
    load_paysim_transactions,
    preprocess_transactions,
)

__all__ = [
    "load_ieee_cis_transactions",
    "load_paysim_transactions",
    "preprocess_transactions",
]
