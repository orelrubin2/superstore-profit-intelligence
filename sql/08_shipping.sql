-- Business question: How do profit, profit margin and shipping time differ by ship mode?
-- Why we checked: Queries 01-07 covered products, discounts, regions, time and customers.
--                 Ship mode is the last order attribute chosen at order time, and the
--                 ML step (04) uses it as a feature, so we need to know if it matters.
-- Insight: Ship mode changes delivery time, but not profitability.
--          - Standard Class carries most of the business: 59.12% of sales, 5,967 of
--            9,993 lines, with the slowest shipping (5.01 days on average, 3-7 days).
--          - Same Day ships in 0.04 days on average (0-1 days) and First Class in
--            2.18 days, but margins are almost the same in every mode:
--            12.08% (Standard) to 13.93% (First Class), a spread under 2 points.
--          - Discounting is also similar across modes (13.89%-16.46% average
--            discount, 12.19%-14.50% of lines above 20%), so no mode is a hidden
--            discount channel.
--          - Part 2 shows why: within each discount band the modes have similar
--            margins (e.g. 0%: 25.24%-31.59%; 21-40%: -19.89% to -11.54%), while
--            moving from the 0% band to >40% drops the margin of every mode below
--            -58%. The discount, not the shipping choice, decides profit.
-- Caveats: The data has no shipping cost column, so we cannot tell whether faster
--          modes cost the company more; profit may or may not include shipping.
--          Same Day is small (543 lines), and the >40% band per mode has 45-589 lines.
-- Next: treat ship mode as a service decision, not a profit lever; keep the focus on
--       discount policy (queries 03-05).
SELECT
    T.ship_mode,
    COUNT(*)                                                           AS order_lines,
    COUNT(DISTINCT T.order_id)                                         AS orders,
    ROUND(SUM(T.sales), 2)                                             AS total_sales,
    ROUND(SUM(T.sales) / SUM(SUM(T.sales)) OVER () * 100, 2)           AS pct_of_sales,
    ROUND(SUM(T.profit), 2)                                            AS total_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)                       AS profit_margin_pct,
    ROUND(AVG(T.ship_days), 2)                                         AS avg_ship_days,
    MIN(T.ship_days)                                                   AS min_ship_days,
    MAX(T.ship_days)                                                   AS max_ship_days,
    ROUND(AVG(T.discount) * 100, 2)                                    AS avg_discount_pct,
    ROUND(AVG(CASE WHEN T.discount > 0.2 THEN 1 ELSE 0 END) * 100, 2)   AS pct_high_discount_lines,
    ROUND(AVG(CASE WHEN T.is_loss THEN 1 ELSE 0 END) * 100, 2)         AS pct_loss_lines
FROM orders_clean AS T
GROUP BY T.ship_mode
ORDER BY avg_ship_days ASC;


-- Part 2: profit margin by ship mode within each discount band
SELECT
    T.discount_band,
    T.ship_mode,
    COUNT(*)                                                           AS order_lines,
    ROUND(SUM(T.profit), 2)                                            AS total_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)                       AS profit_margin_pct
FROM orders_clean AS T
GROUP BY T.discount_band, T.ship_mode
ORDER BY MIN(T.discount), T.ship_mode;
