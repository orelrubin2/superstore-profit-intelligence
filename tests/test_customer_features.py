"""Tests for the step 3 customer tables (customer_features, customer_mix).

The main test recomputes every feature independently in pandas from
orders_clean.parquet and compares it with the SQL-built table, customer by
customer.

Run from the project root (after `python -m src.build_database`):
    python -m pytest
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.build_database import DB_PATH, PARQUET_PATH
from src.customer_features import (
    HIGH_DISCOUNT,
    HIGH_VOLUME_QUANTILE,
    NEAR_ZERO_PROFIT,
    check_customer_tables,
)


@pytest.fixture(scope="module")
def con():
    """Read-only connection to the built database."""
    if not DB_PATH.exists():
        pytest.skip("database not built; run `python -m src.build_database` first")
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def features(con):
    """customer_features indexed by customer_id."""
    return con.sql("SELECT * FROM customer_features").df().set_index("customer_id")


@pytest.fixture(scope="module")
def expected():
    """The same features recomputed in pandas from the Parquet export."""
    lines = pd.read_parquet(PARQUET_PATH)
    reference_date = lines["order_date"].max() + pd.Timedelta(days=1)
    lines["is_high_discount"] = lines["discount"] > HIGH_DISCOUNT
    lines["high_gross"] = lines["gross_sales"].where(lines["is_high_discount"], 0)
    lines["loss"] = lines["profit"].where(lines["profit"] < 0)

    g = lines.groupby("customer_id")
    out = pd.DataFrame({
        "recency_days": (reference_date - g["order_date"].max()).dt.days,
        "tenure_days": (reference_date - g["order_date"].min()).dt.days,
        "frequency": g["order_id"].nunique(),
        "order_lines": g.size(),
        "quantity": g["quantity"].sum(),
        "sales": g["sales"].sum(),
        "gross_sales": g["gross_sales"].sum(),
        "profit": g["profit"].sum(),
        "avg_discount": g["discount"].mean(),
        "high_discount_line_share": g["is_high_discount"].mean(),
        "high_discount_gross_share": g["high_gross"].sum() / g["gross_sales"].sum(),
        "loss_lines": g["loss"].count(),
        "worst_line_profit": g["profit"].min(),
        "worst_line_share_of_losses": g["profit"].min() / g["loss"].sum(min_count=1),
    })
    out["profit_margin"] = out["profit"] / out["sales"]
    out.loc[out["loss_lines"] == 0, "worst_line_share_of_losses"] = np.nan

    # Thresholds use linear interpolation, like DuckDB's QUANTILE_CONT.
    gross_p75 = out["gross_sales"].quantile(HIGH_VOLUME_QUANTILE)
    out["is_high_volume"] = out["gross_sales"] >= gross_p75
    out["is_net_loss"] = out["profit"] < 0
    out["is_near_zero_profit"] = out["profit"].abs() < NEAR_ZERO_PROFIT
    out["volume_group"] = np.select(
        [out["is_high_volume"] & out["is_net_loss"], out["is_high_volume"]],
        ["HV-loss", "HV-profit"], default="Other",
    )
    out["hv_gross_p80"] = out["gross_sales"] >= out["gross_sales"].quantile(0.80)
    out["hv_net_sales_p75"] = out["sales"] >= out["sales"].quantile(HIGH_VOLUME_QUANTILE)
    out["hv_quantity_p75"] = out["quantity"] >= out["quantity"].quantile(HIGH_VOLUME_QUANTILE)
    return out


def test_tables_reconcile(con):
    """One row per customer; totals and mix shares reconcile with orders_clean."""
    check_customer_tables(con)


def test_features_match_independent_pandas(features, expected):
    """Every SQL feature equals the pandas recomputation for every customer."""
    assert set(features.index) == set(expected.index)
    sql = features.loc[expected.index]
    for column in expected.columns:
        if expected[column].dtype.kind in "fi":
            np.testing.assert_allclose(
                sql[column].astype(float), expected[column].astype(float),
                rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=column,
            )
        else:
            assert (sql[column].astype(str) == expected[column].astype(str)).all(), column


def test_high_volume_is_top_quarter(features):
    """About 25% of customers are high-volume, split into HV-loss and HV-profit."""
    share = features["is_high_volume"].mean()
    assert 0.24 <= share <= 0.26
    counts = features["volume_group"].value_counts()
    assert counts["HV-loss"] + counts["HV-profit"] == features["is_high_volume"].sum()


def test_value_ranges(features):
    """Shares are within [0, 1]; the stricter p80 group is inside the p75 group."""
    for column in ["avg_discount", "high_discount_line_share", "high_discount_gross_share"]:
        assert features[column].between(0, 1).all(), column
    shares = features["worst_line_share_of_losses"].dropna()
    assert shares.between(0, 1).all()
    assert not (features["hv_gross_p80"] & ~features["is_high_volume"]).any()


def test_mix_dimensions_complete(con):
    """Every customer has rows for all five mix dimensions."""
    per_customer = con.sql("""
        SELECT customer_id, COUNT(DISTINCT dimension) AS dims
        FROM customer_mix GROUP BY customer_id
    """).df()
    assert (per_customer["dims"] == 5).all()
