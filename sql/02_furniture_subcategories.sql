-- Business question: Within Furniture, which sub-categories lose money, and how do
--                    their discounts and loss rates compare?
-- Insight: Tables and Bookcases are the only Furniture sub-categories that lose money
--          They also have the highest average discounts (26.13% and 21.11%) and the
--          highest share of loss-making lines.
--          By contrast, Furnishings has the lowest average discount (13.83%) and the
--          highest margin (14.24%).
--          Note: Chairs is profitable (8.11% margin) despite a 17% average discount,
--          so discount is not the only driver.
SELECT
    T.sub_category,                                                   
    ROUND(SUM(T.sales), 2)                                AS total_sales,
    ROUND(SUM(T.profit), 2)                               AS total_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)          AS profit_margin_pct,
    ROUND(AVG(T.discount) * 100, 2)                       AS avg_discount_pct,   
    ROUND(AVG(CASE WHEN T.is_loss THEN 1 ELSE 0 END) * 100, 2) AS pct_loss_lines 
FROM orders_clean AS T
WHERE T.category = 'Furniture'
GROUP BY T.sub_category
ORDER BY total_profit ASC;                               