"""Integrity tests for the DuckDB database.

Run from the project root (after `python -m src.build_database`):
    python -m pytest
"""

import duckdb
import pytest

from src.build_database import (
    DB_PATH,
    DIM_KEYS,
    check_matches_raw,
    check_star_join_reconciles,
    check_unique_keys,
)


@pytest.fixture(scope="module")
def con():
    """Read-only connection to the built database."""
    if not DB_PATH.exists():
        pytest.skip("database not built; run `python -m src.build_database` first")
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    yield connection
    connection.close()


def test_dim_keys_unique(con):
    """Every dimension table has a unique, non-null primary key."""
    check_unique_keys(con)


@pytest.mark.parametrize("table", DIM_KEYS)
def test_every_fact_row_has_dim_match(con, table):
    """No fact row points to a key that is missing from a dimension."""
    if table == "dim_date":
        fact_col = "order_date"
    else:
        fact_col = DIM_KEYS[table]
    key = DIM_KEYS[table]
    missing = con.execute(f"""
        SELECT COUNT(*) FROM fact_order_lines f
        LEFT JOIN {table} d ON f.{fact_col} = d.{key}
        WHERE d.{key} IS NULL
    """).fetchone()[0]
    assert missing == 0


def test_star_join_matches_orders_clean(con):
    """Fact joined to all dims has the same row count and totals as orders_clean."""
    check_star_join_reconciles(con)


def test_orders_clean_matches_raw(con):
    """orders_clean equals the raw file minus the logged dropped rows."""
    dropped = [r[0] for r in con.execute("SELECT row_id FROM dropped_rows").fetchall()]
    assert dropped == [3407]
    check_matches_raw(con, dropped)


def test_dim_product_uses_most_frequent_name(con):
    """For each multi-name product, dim_product holds the chosen variant."""
    mismatches = con.execute("""
        SELECT COUNT(*) FROM product_name_variants v
        JOIN dim_product p USING (product_id)
        WHERE v.is_chosen AND v.product_name <> p.product_name
    """).fetchone()[0]
    n_products = con.execute(
        "SELECT COUNT(DISTINCT product_id) FROM product_name_variants"
    ).fetchone()[0]
    assert mismatches == 0
    assert n_products == 32


def test_orders_view_one_row_per_order(con):
    """The orders view has one row per order and the same totals as the fact table."""
    view_rows, view_sales = con.execute(
        "SELECT COUNT(*), ROUND(SUM(sales), 4) FROM orders"
    ).fetchone()
    fact_orders, fact_sales = con.execute(
        "SELECT COUNT(DISTINCT order_id), ROUND(SUM(sales), 4) FROM fact_order_lines"
    ).fetchone()
    assert (view_rows, view_sales) == (fact_orders, fact_sales)
