-- Business question: Do high discounts lead to losses in every category,
--                    or only in Furniture?
-- Insight: ___

SELECT
    T.category,
    T.discount_band,
    COUNT(*)                                                   AS order_lines,
    ROUND(SUM(T.sales), 2)                                     AS total_sales,
    ROUND(SUM(T.profit), 2)                                    AS total_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)               AS profit_margin_pct,
    ROUND(AVG(CASE WHEN T.is_loss THEN 1 ELSE 0 END) * 100, 2) AS pct_loss_lines
FROM orders_clean AS T
GROUP BY T.category, T.discount_band
ORDER BY T.category, MIN(T.discount);