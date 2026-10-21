"""Cached data and model access for the Streamlit dashboard.

On first use the app builds the DuckDB database from the raw CSV in the repo
(the .duckdb file is not in git), then trains the step 4 Random Forest on
2014-2016 with the tuned settings from notebook 04. Both are cached for the
life of the server process, so later visits are instant.

All analysis logic is reused from src/; nothing is recomputed differently here.
"""

import duckdb
import pandas as pd
import streamlit as st

import src.customer_comparison as cc
import src.loss_model as lm
from src.build_database import DB_PATH, build_database

DISCOUNT_BANDS = ["0%", "1-20%", "21-40%", ">40%"]


@st.cache_resource(show_spinner="Building the database from the raw data (first visit only)...")
def ensure_database() -> str:
    """Build the database once per server process if it does not exist yet."""
    if not DB_PATH.exists():
        build_database()
    return str(DB_PATH)


def connect() -> duckdb.DuckDBPyConnection:
    """A short-lived read-only connection; callers close it."""
    return duckdb.connect(ensure_database(), read_only=True)


@st.cache_data(show_spinner=False)
def load_lines() -> pd.DataFrame:
    """orders_clean: one row per order line."""
    with connect() as con:
        return con.execute("SELECT * FROM orders_clean").df()


@st.cache_data(show_spinner=False)
def load_customers() -> pd.DataFrame:
    """Customer features joined with RFM segment and clusters (step 3)."""
    with connect() as con:
        return cc.load_customers(con)


@st.cache_data(show_spinner=False)
def discount_mix_by_group() -> pd.DataFrame:
    """Gross-sales share by discount band for HV-loss and HV-profit customers."""
    with connect() as con:
        return cc.group_mix(con, "discount_band")


@st.cache_resource(show_spinner="Training the loss-risk model (first visit only)...")
def load_model() -> dict:
    """The step 4 Random Forest and Baseline B, trained on 2014-2016, plus lookups."""
    with connect() as con:
        lines = lm.load_lines(con)
    train, _ = lm.split_by_year(lines)
    forest = lm.make_forest(**lm.TUNED_FOREST_PARAMS).fit(train[lm.FEATURES], train[lm.TARGET])
    baseline_b = lm.BaselineB(lm.TUNED_B_THRESHOLD).fit(train)
    return {
        "forest": forest,
        "explainer": lm.make_explainer(forest),
        "baseline_b": baseline_b,
        "train": train,
        "base_rate": float(train[lm.TARGET].mean()),
        "category_of": train.groupby("sub_category")["category"].first().to_dict(),
        "region_of": train.groupby("state")["region"].first().to_dict(),
        "median_price": train.groupby("sub_category")["unit_price"].median().to_dict(),
    }
