-- Business question: What are the sales, profit and profit margin of each category?
-- Insight: Furniture generates about a third of total sales but only ~6% of total profit
--          (2.5% margin vs. ~17% in Technology and Office Supplies).
--          Next: drill down into Furniture sub-categories to find where the margin is lost.

SELECT
    T.category,
    ROUND(SUM(T.sales), 2)                                    AS total_sales,
    ROUND(SUM(T.profit), 2)                                   AS total_profit,
    ROUND(SUM(T.sales)  / SUM(SUM(T.sales))  OVER () * 100, 2) AS pct_of_sales,
    ROUND(SUM(T.profit) / SUM(SUM(T.profit)) OVER () * 100, 2) AS pct_of_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)              AS profit_margin_pct
FROM orders_clean AS T
GROUP BY T.category
ORDER BY total_profit DESC;