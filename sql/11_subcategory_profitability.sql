-- Business question: Which sub-categories, across all categories, make or lose money?
-- Why we checked: Query 02 looked only inside Furniture. To compare products on one
--                 scale we need all 17 sub-categories together.
-- Insight: Three sub-categories lose money, and size does not predict profit.
--          - Tables (-$17,725), Bookcases (-$3,473) and Supplies (-$1,189) lose
--            $22,387 together on 16.04% of sales.
--          - Tables and Bookcases have high average discounts (26.13% and 21.11%), but
--            Supplies does not (7.68%), so its loss has another cause (to check).
--          - Machines barely breaks even (1.79% margin on $189,239 in sales) with the
--            highest average discount of Technology (30.61%).
--          - The most profitable are Copiers ($55,618, 37.20% margin, no loss-making
--            lines), Phones ($44,516) and Accessories ($41,937). Paper, Labels and
--            Envelopes earn 42%-44% margins on small sales.
--          - Chairs and Phones are the two largest sub-categories by sales
--            ($328,168 and $330,007), but Chairs earns only an 8.11% margin.
--          - Binders combines a 37.23% average discount with a 14.86% margin, so a
--            high average discount does not always mean a loss.
-- Caveats: Average discount is per line, not weighted by sales.
-- Next: look into Supplies, the only low-discount loss-maker.
SELECT
    T.category,
    T.sub_category,
    COUNT(*)                                                           AS order_lines,
    ROUND(SUM(T.sales), 2)                                             AS total_sales,
    ROUND(SUM(T.profit), 2)                                            AS total_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)                       AS profit_margin_pct,
    ROUND(AVG(T.discount) * 100, 2)                                    AS avg_discount_pct,
    ROUND(AVG(CASE WHEN T.is_loss THEN 1 ELSE 0 END) * 100, 2)         AS pct_loss_lines
FROM orders_clean AS T
GROUP BY T.category, T.sub_category
ORDER BY total_profit ASC;
