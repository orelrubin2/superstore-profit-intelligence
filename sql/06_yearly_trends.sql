-- Business question: How did sales, profit and discounting change from year to year?
-- Why we checked: Queries 01-05 looked at all four years together. We need to know
--                 whether the business is growing and whether the discount problem
--                 is getting better or worse over time.
--  Insight: The business is growing, but growth became less profitable in 2017.
--          - Sales grew from $483,966 (2014) to $733,215 (2017), about +51.5%,
--            and orders grew from 969 to 1,687.
--          - Through 2016, profit grew faster than sales (2016: sales +29.47%,
--            profit +32.74%). In 2017 this reversed: sales +20.36% but profit only
--            +14.24%, and the margin fell from 13.43% to 12.74%. This is one year,
--            so it is an early warning sign rather than a proven trend.
--          - The share of high-discount lines stayed almost the same every year
--            (13.74%-14.11%), so the discount pattern is a stable, long-running policy,
--            not a new one. Addressing it could improve profit in every future year.
-- Next: measure how much of the profit comes from the top customers (Pareto analysis).

SELECT
    T.order_year,
    COUNT(DISTINCT T.order_id)                                        AS orders,
    COUNT(DISTINCT T.customer_id)                                     AS active_customers,
    ROUND(SUM(T.sales), 2)                                            AS total_sales,
    ROUND(SUM(T.profit), 2)                                           AS total_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)                      AS profit_margin_pct,
    ROUND(
        (SUM(T.sales) - LAG(SUM(T.sales)) OVER (ORDER BY T.order_year))
        / LAG(SUM(T.sales)) OVER (ORDER BY T.order_year) * 100, 2)    AS sales_growth_pct,
    ROUND(
        (SUM(T.profit) - LAG(SUM(T.profit)) OVER (ORDER BY T.order_year))
        / LAG(SUM(T.profit)) OVER (ORDER BY T.order_year) * 100, 2)   AS profit_growth_pct,
    ROUND(AVG(CASE WHEN T.discount > 0.2 THEN 1 ELSE 0 END) * 100, 2) AS pct_high_discount_lines
FROM orders_clean AS T
GROUP BY T.order_year
ORDER BY T.order_year;


-- Part 2: text chart of sales and profit over time (DuckDB bar() function).
-- Each bar has its own scale: sales up to $800K, profit up to $100K.
SELECT
    T.order_year,
    ROUND(SUM(T.sales), 0)               AS total_sales,
    bar(SUM(T.sales), 0, 800000, 30)     AS sales_chart,
    ROUND(SUM(T.profit), 0)              AS total_profit,
    bar(SUM(T.profit), 0, 100000, 30)    AS profit_chart
FROM orders_clean AS T
GROUP BY T.order_year
ORDER BY T.order_year;