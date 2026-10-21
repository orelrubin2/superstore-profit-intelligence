"""Build the DuckDB database and Parquet export from the cleaned data.

Tables created in data/processed/superstore.duckdb:
    orders_clean           wide order-line table (one row per order line)
    fact_order_lines       star-schema fact table
    dim_customer           one row per customer_id
    dim_product            one row per product_id (most frequent name)
    dim_location           one row per location_id (postal_code|city|state)
    dim_date               calendar covering all order and ship dates
    product_name_variants  the product_ids that have more than one name
    data_quality_log       every cleaning check and the action taken
    orders (view)          one row per order, for RFM analysis
    customer_features      one row per customer: RFM, profit, discounts, groups
    customer_mix           customer x discount band / category / sub-category /
                           region / state, with gross-sales shares
    customer_rfm           one row per customer: R, F, M scores and R x F segment
    customer_clusters      one row per customer: K-Means cluster (k=3) and contrast (k=2)

Run from the project root:
    python -m src.build_database
"""

from pathlib import Path

import duckdb
import pandas as pd

from src.cleaning import RAW_PATH, clean_superstore, load_raw
from src.customer_features import check_customer_tables, create_customer_tables
from src.segmentation import (
    check_cluster_table,
    check_rfm_table,
    create_cluster_table,
    create_rfm_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "superstore.duckdb"
PARQUET_PATH = PROCESSED_DIR / "orders_clean.parquet"

# Primary key of every dimension table, checked for uniqueness.
DIM_KEYS = {
    "dim_customer": "customer_id",
    "dim_product": "product_id",
    "dim_location": "location_id",
    "dim_date": "date",
}


def add_location_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add location_id = 'postal_code|city|state'.

    postal_code alone is not unique (92024 is both San Diego and Encinitas),
    so the key combines all three fields.
    """
    df = df.copy()
    df["location_id"] = df["postal_code"] + "|" + df["city"] + "|" + df["state"]
    return df


def build_product_name_variants(df: pd.DataFrame) -> pd.DataFrame:
    """List every name used by product_ids that have more than one name.

    is_chosen marks the name used in dim_product: the most frequent one, with
    ties broken alphabetically so the result is the same on every run.
    """
    counts = (
        df.groupby(["product_id", "product_name"], as_index=False)
        .size()
        .rename(columns={"size": "line_count"})
    )
    counts["name_count"] = counts.groupby("product_id")["product_name"].transform("count")
    variants = counts[counts["name_count"] > 1].copy()
    variants = variants.sort_values(
        ["product_id", "line_count", "product_name"], ascending=[True, False, True]
    )
    variants["is_chosen"] = ~variants.duplicated("product_id", keep="first")
    max_count = variants.groupby("product_id")["line_count"].transform("max")
    variants["is_tie"] = (
        variants.groupby("product_id")["line_count"].transform(lambda s: (s == s.max()).sum()) > 1
    ) & (variants["line_count"] == max_count)
    return variants.drop(columns="name_count").reset_index(drop=True)


def build_dim_product(df: pd.DataFrame) -> pd.DataFrame:
    """One row per product_id with its most frequent name (ties: alphabetical)."""
    names = (
        df.groupby(["product_id", "product_name"], as_index=False)
        .size()
        .sort_values(["product_id", "size", "product_name"], ascending=[True, False, True])
        .drop_duplicates("product_id")
        [["product_id", "product_name"]]
    )
    name_counts = (
        df.groupby("product_id")["product_name"].nunique().rename("name_variant_count")
    )
    categories = df[["product_id", "category", "sub_category"]].drop_duplicates()
    return (
        names.merge(categories, on="product_id")
        .merge(name_counts, on="product_id")
        .sort_values("product_id")
        .reset_index(drop=True)
    )


def build_dim_customer(df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer_id with name and segment."""
    return (
        df[["customer_id", "customer_name", "segment"]]
        .drop_duplicates()
        .sort_values("customer_id")
        .reset_index(drop=True)
    )


def build_dim_location(df: pd.DataFrame) -> pd.DataFrame:
    """One row per location_id with postal code, city, state and region."""
    return (
        df[["location_id", "postal_code", "city", "state", "region"]]
        .drop_duplicates()
        .sort_values("location_id")
        .reset_index(drop=True)
    )


def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    """Daily calendar from the first order date to the last ship date."""
    dates = pd.date_range(df["order_date"].min(), df["ship_date"].max(), freq="D")
    return pd.DataFrame({
        "date": dates,
        "year": dates.year,
        "quarter": dates.quarter,
        "month": dates.month,
        "month_name": dates.month_name(),
        "year_month": dates.to_period("M").astype(str),
        "day_of_week": dates.dayofweek + 1,  # 1 = Monday
        "day_name": dates.day_name(),
        "is_weekend": dates.dayofweek >= 5,
    })


def build_fact_order_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Order-line facts with foreign keys to every dimension.

    Descriptive attributes (names, segment, category, geography) live in the
    dimensions. Outlier flags are excluded: they are for analysis only and
    remain in orders_clean.
    """
    columns = [
        "row_id", "order_id", "order_date", "ship_date", "ship_mode",
        "customer_id", "product_id", "location_id",
        "sales", "quantity", "discount", "profit",
        "profit_margin", "is_loss", "discount_band",
        "gross_sales", "discount_amount", "unit_price", "ship_days",
    ]
    return df[columns].copy()


ORDERS_VIEW_SQL = """
CREATE OR REPLACE VIEW orders AS
SELECT
    order_id,
    ANY_VALUE(order_date)   AS order_date,
    ANY_VALUE(ship_date)    AS ship_date,
    ANY_VALUE(ship_mode)    AS ship_mode,
    ANY_VALUE(customer_id)  AS customer_id,
    ANY_VALUE(location_id)  AS location_id,
    COUNT(*)                AS line_count,
    SUM(quantity)           AS quantity,
    SUM(sales)              AS sales,
    SUM(profit)             AS profit,
    SUM(discount_amount)    AS discount_amount,
    SUM(profit) < 0         AS is_loss
FROM fact_order_lines
GROUP BY order_id
"""


def write_tables(con: duckdb.DuckDBPyConnection, tables: dict[str, pd.DataFrame]) -> None:
    """Create or replace one DuckDB table per DataFrame."""
    for name, frame in tables.items():
        con.register("_frame", frame)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _frame")
        con.unregister("_frame")


def check_unique_keys(con: duckdb.DuckDBPyConnection) -> None:
    """Assert that every dimension's key is unique and never null."""
    for table, key in DIM_KEYS.items():
        rows, distinct, nulls = con.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT {key}), COUNT(*) - COUNT({key}) FROM {table}"
        ).fetchone()
        assert nulls == 0, f"{table}.{key} has {nulls} null values"
        assert rows == distinct, f"{table}.{key} is not unique ({rows} rows, {distinct} distinct)"


def check_star_join_reconciles(con: duckdb.DuckDBPyConnection) -> None:
    """Assert that fact joined to all dims matches orders_clean exactly.

    Compares row count and the totals of sales, profit, quantity and
    gross_sales. INNER JOINs are used, so a missing dimension row would show
    up as lost rows; a duplicated key would show up as extra rows.
    """
    totals_sql = """
        SELECT COUNT(*), ROUND(SUM(sales), 4), ROUND(SUM(profit), 4),
               SUM(quantity), ROUND(SUM(gross_sales), 4)
        FROM {source}
    """
    joined = """(
        SELECT f.*
        FROM fact_order_lines f
        JOIN dim_customer c ON f.customer_id = c.customer_id
        JOIN dim_product  p ON f.product_id  = p.product_id
        JOIN dim_location l ON f.location_id = l.location_id
        JOIN dim_date     od ON f.order_date = od.date
        JOIN dim_date     sd ON f.ship_date  = sd.date
    )"""
    expected = con.execute(totals_sql.format(source="orders_clean")).fetchone()
    actual = con.execute(totals_sql.format(source=joined)).fetchone()
    assert actual == expected, f"star join {actual} != orders_clean {expected}"


def check_matches_raw(con: duckdb.DuckDBPyConnection, dropped_row_ids: list[int]) -> None:
    """Assert orders_clean equals the raw file minus the dropped duplicate rows.

    The raw file is read with pandas as cp1252: DuckDB's CSV reader does not
    support cp1252, and its latin-1 mode rejects the smart-quote bytes
    (0x93/0x94) that the file contains.
    """
    raw = load_raw(RAW_PATH)
    raw = raw[~raw["Row ID"].isin(dropped_row_ids)]
    con.register("_raw", raw)
    raw_totals = con.execute("""
        SELECT COUNT(*), ROUND(SUM("Sales"), 4), ROUND(SUM("Profit"), 4), SUM("Quantity")
        FROM _raw
    """).fetchone()
    con.unregister("_raw")
    clean_totals = con.execute("""
        SELECT COUNT(*), ROUND(SUM(sales), 4), ROUND(SUM(profit), 4), SUM(quantity)
        FROM orders_clean
    """).fetchone()
    assert raw_totals == clean_totals, (
        f"orders_clean {clean_totals} != raw minus dropped rows {raw_totals}"
    )


def run_checks(con: duckdb.DuckDBPyConnection) -> None:
    """Run every integrity check against an open database."""
    dropped = con.execute(
        "SELECT row_id FROM dropped_rows ORDER BY row_id"
    ).fetchall()
    check_unique_keys(con)
    check_star_join_reconciles(con)
    check_matches_raw(con, [r[0] for r in dropped])


def build_database(db_path: Path = DB_PATH, parquet_path: Path = PARQUET_PATH) -> None:
    """Clean the raw data, write all tables and the Parquet file, then run checks."""
    orders_clean, quality_log = clean_superstore()
    orders_clean = add_location_id(orders_clean)

    raw_ids = set(range(1, 9_995))
    dropped_rows = pd.DataFrame(
        {"row_id": sorted(raw_ids - set(orders_clean["row_id"]))}
    )
    variants = build_product_name_variants(orders_clean)
    quality_log = pd.concat([quality_log, pd.DataFrame([{
        "check": "product_name_ties",
        "column": "product_name",
        "rows_affected": int(variants.loc[variants["is_tie"], "product_id"].nunique()),
        "action": "tie for most frequent name broken alphabetically in dim_product",
    }])], ignore_index=True)

    tables = {
        "orders_clean": orders_clean,
        "fact_order_lines": build_fact_order_lines(orders_clean),
        "dim_customer": build_dim_customer(orders_clean),
        "dim_product": build_dim_product(orders_clean),
        "dim_location": build_dim_location(orders_clean),
        "dim_date": build_dim_date(orders_clean),
        "product_name_variants": variants,
        "data_quality_log": quality_log,
        "dropped_rows": dropped_rows,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    orders_clean.to_parquet(parquet_path, index=False)

    with duckdb.connect(str(db_path)) as con:
        write_tables(con, tables)
        con.execute(ORDERS_VIEW_SQL)
        create_customer_tables(con)
        create_rfm_table(con)
        create_cluster_table(con)
        run_checks(con)
        check_customer_tables(con)
        check_rfm_table(con)
        check_cluster_table(con)
        summary = con.execute("""
            SELECT table_name, estimated_size AS rows
            FROM duckdb_tables() ORDER BY table_name
        """).df()

    print(summary.to_string(index=False))
    print(f"\nAll checks passed. Database: {db_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    build_database()
