"""Customer-level tables for step 3 (customer segmentation).

Two tables are built inside the DuckDB database from orders_clean:

    customer_features  one row per customer: RFM metrics, profit, discounts,
                       loss concentration and the high-volume group flags
    customer_mix       one row per customer x dimension value (discount band,
                       category, sub-category, region, state) with that slice's
                       gross sales, sales, profit and share of the customer's
                       gross sales

Definitions (agreed in the step 3 plan):
    reference date  the day after the last order date in the data (2017-12-31);
                    recency_days = reference date - last order date
    frequency       number of distinct orders (not order lines)
    high-volume     gross_sales at or above the 75th percentile of customers.
                    Gross sales (before discount) is used instead of net sales so
                    that heavily discounted customers are not pushed out of the
                    group. ASSUMPTION: gross_sales assumes `sales` is recorded
                    after discount (see notebooks/01_data_quality.ipynb).
    net loss        total profit over 2014-2017 below 0

WARNING (leakage): every feature here summarizes a customer's full history,
including orders after any given order line. Do not use these columns, or
segments derived from them, as features in the step 4 order-line model.
"""

import duckdb

HIGH_VOLUME_QUANTILE = 0.75   # primary definition: top 25% by gross sales
HIGH_DISCOUNT = 0.20          # "high discount" means strictly above 20%, as in step 2
NEAR_ZERO_PROFIT = 100        # |profit| below this is treated as near break-even

CUSTOMER_FEATURES_SQL = f"""
CREATE OR REPLACE TABLE customer_features AS
WITH reference AS (
    SELECT MAX(order_date) + INTERVAL 1 DAY AS reference_date FROM orders_clean
),
per_customer AS (
    SELECT
        customer_id,
        MIN(order_date)                                      AS first_order_date,
        MAX(order_date)                                      AS last_order_date,
        COUNT(DISTINCT order_id)                             AS frequency,
        COUNT(*)                                             AS order_lines,
        SUM(quantity)                                        AS quantity,
        SUM(sales)                                           AS sales,
        SUM(gross_sales)                                     AS gross_sales,
        SUM(profit)                                          AS profit,
        SUM(profit) / SUM(sales)                             AS profit_margin,
        AVG(discount)                                        AS avg_discount,
        AVG((discount > {HIGH_DISCOUNT})::INT)               AS high_discount_line_share,
        SUM(gross_sales) FILTER (WHERE discount > {HIGH_DISCOUNT})
            / SUM(gross_sales)                               AS high_discount_gross_share,
        COUNT(*) FILTER (WHERE profit < 0)                   AS loss_lines,
        MIN(profit)                                          AS worst_line_profit,
        SUM(profit) FILTER (WHERE profit < 0)                AS total_line_losses
    FROM orders_clean
    GROUP BY customer_id
),
thresholds AS (
    SELECT
        QUANTILE_CONT(gross_sales, {HIGH_VOLUME_QUANTILE}) AS gross_p75,
        QUANTILE_CONT(gross_sales, 0.80)                   AS gross_p80,
        QUANTILE_CONT(sales, {HIGH_VOLUME_QUANTILE})       AS sales_p75,
        QUANTILE_CONT(quantity, {HIGH_VOLUME_QUANTILE})    AS quantity_p75
    FROM per_customer
)
SELECT
    p.customer_id,
    c.customer_name,
    c.segment,
    p.first_order_date,
    p.last_order_date,
    DATE_DIFF('day', p.last_order_date, r.reference_date)  AS recency_days,
    DATE_DIFF('day', p.first_order_date, r.reference_date) AS tenure_days,
    p.frequency,
    p.order_lines,
    p.quantity,
    p.sales,
    p.gross_sales,
    p.profit,
    p.profit_margin,
    p.avg_discount,
    p.high_discount_line_share,
    COALESCE(p.high_discount_gross_share, 0)               AS high_discount_gross_share,
    p.loss_lines,
    p.worst_line_profit,
    -- Share of all the customer's line-level losses that comes from the single
    -- worst line (1 = one bad line explains every loss); NULL if no loss lines.
    CASE WHEN p.loss_lines > 0
         THEN p.worst_line_profit / p.total_line_losses END AS worst_line_share_of_losses,
    p.gross_sales >= t.gross_p75                           AS is_high_volume,
    p.profit < 0                                           AS is_net_loss,
    ABS(p.profit) < {NEAR_ZERO_PROFIT}                     AS is_near_zero_profit,
    CASE
        WHEN p.gross_sales >= t.gross_p75 AND p.profit < 0 THEN 'HV-loss'
        WHEN p.gross_sales >= t.gross_p75                  THEN 'HV-profit'
        ELSE 'Other'
    END                                                    AS volume_group,
    -- Alternative high-volume definitions for sensitivity checks.
    p.gross_sales >= t.gross_p80                           AS hv_gross_p80,
    p.sales >= t.sales_p75                                 AS hv_net_sales_p75,
    p.quantity >= t.quantity_p75                           AS hv_quantity_p75
FROM per_customer AS p
JOIN dim_customer AS c USING (customer_id)
CROSS JOIN reference AS r
CROSS JOIN thresholds AS t
ORDER BY p.customer_id
"""

MIX_DIMENSIONS = ["discount_band", "category", "sub_category", "region", "state"]

CUSTOMER_MIX_SQL = "CREATE OR REPLACE TABLE customer_mix AS\n" + "\nUNION ALL\n".join(
    f"""
    SELECT
        customer_id,
        '{dimension}'                                       AS dimension,
        {dimension}                                         AS value,
        COUNT(*)                                            AS order_lines,
        SUM(gross_sales)                                    AS gross_sales,
        SUM(sales)                                          AS sales,
        SUM(profit)                                         AS profit,
        SUM(gross_sales) / SUM(SUM(gross_sales)) OVER (PARTITION BY customer_id)
                                                            AS gross_share
    FROM orders_clean
    GROUP BY customer_id, {dimension}"""
    for dimension in MIX_DIMENSIONS
) + "\nORDER BY customer_id, dimension, value"


def create_customer_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create customer_features and customer_mix from orders_clean."""
    con.execute(CUSTOMER_FEATURES_SQL)
    con.execute(CUSTOMER_MIX_SQL)


def check_customer_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Assert the customer tables are complete and reconcile with orders_clean."""
    rows, distinct = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT customer_id) FROM customer_features"
    ).fetchone()
    customers = con.execute(
        "SELECT COUNT(DISTINCT customer_id) FROM orders_clean"
    ).fetchone()[0]
    assert rows == distinct == customers, (
        f"customer_features has {rows} rows / {distinct} ids, expected {customers}"
    )

    totals_sql = """
        SELECT ROUND(SUM(sales), 4), ROUND(SUM(gross_sales), 4),
               ROUND(SUM(profit), 4), SUM(quantity), SUM(order_lines)
        FROM {source}
    """
    lines_sql = totals_sql.replace("SUM(order_lines)", "COUNT(*)")
    expected = con.execute(lines_sql.format(source="orders_clean")).fetchone()
    actual = con.execute(totals_sql.format(source="customer_features")).fetchone()
    assert actual == expected, f"customer_features totals {actual} != orders_clean {expected}"

    orders_total = con.execute("SELECT COUNT(DISTINCT order_id) FROM orders_clean").fetchone()[0]
    frequency_total = con.execute("SELECT SUM(frequency) FROM customer_features").fetchone()[0]
    assert frequency_total == orders_total, "frequency does not add up to the number of orders"

    min_recency = con.execute("SELECT MIN(recency_days) FROM customer_features").fetchone()[0]
    assert min_recency >= 1, "recency must be at least 1 day (reference = last date + 1)"

    for dimension in MIX_DIMENSIONS:
        mix = con.execute(f"""
            SELECT ROUND(SUM(gross_sales), 4), ROUND(SUM(profit), 4),
                   SUM(order_lines), MAX(ABS(share_total - 1))
            FROM (
                SELECT customer_id, SUM(gross_sales) AS gross_sales, SUM(profit) AS profit,
                       SUM(order_lines) AS order_lines, SUM(gross_share) AS share_total
                FROM customer_mix WHERE dimension = '{dimension}'
                GROUP BY customer_id
            )
        """).fetchone()
        assert mix[:3] == (expected[1], expected[2], expected[4]), (
            f"customer_mix[{dimension}] totals {mix[:3]} do not match orders_clean"
        )
        assert mix[3] < 1e-9, f"customer_mix[{dimension}] shares do not sum to 1"

    groups = con.execute("""
        SELECT COUNT(*) FILTER (WHERE is_high_volume),
               COUNT(*) FILTER (WHERE volume_group IN ('HV-loss', 'HV-profit'))
        FROM customer_features
    """).fetchone()
    assert groups[0] == groups[1], "volume groups do not partition the high-volume customers"
