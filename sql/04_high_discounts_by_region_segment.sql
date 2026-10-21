-- Business question: Which regions and customer segments receive high discounts
--                    (above 20%), and how much profit do those discounts cost?
-- Why we checked: Query 03 showed that lines with discounts above 20% lose money in
--                 every category. The next question is where those discounts are given.
-- Insight: ___

-- Part 1: by region
SELECT
    T.region,
    COUNT(*)                                                         AS order_lines,
    ROUND(AVG(T.discount) * 100, 2)                                  AS avg_discount_pct,
    ROUND(AVG(CASE WHEN T.discount > 0.2 THEN 1 ELSE 0 END) * 100, 2) AS pct_high_discount_lines,
    ROUND(SUM(T.profit), 2)                                          AS total_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)                     AS profit_margin_pct,
    ROUND(SUM(CASE WHEN T.discount > 0.2 THEN T.profit ELSE 0 END), 2) AS profit_from_high_discounts
FROM orders_clean AS T
GROUP BY T.region
ORDER BY profit_margin_pct ASC;

-- Part 2: by customer segment
SELECT
    T.segment,
    COUNT(*)                                                         AS order_lines,
    ROUND(AVG(T.discount) * 100, 2)                                  AS avg_discount_pct,
    ROUND(AVG(CASE WHEN T.discount > 0.2 THEN 1 ELSE 0 END) * 100, 2) AS pct_high_discount_lines,
    ROUND(SUM(T.profit), 2)                                          AS total_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)                     AS profit_margin_pct,
    ROUND(SUM(CASE WHEN T.discount > 0.2 THEN T.profit ELSE 0 END), 2) AS profit_from_high_discounts
FROM orders_clean AS T
GROUP BY T.segment
ORDER BY profit_margin_pct ASC;