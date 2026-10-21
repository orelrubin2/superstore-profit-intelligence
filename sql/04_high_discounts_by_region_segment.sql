-- Business question: Which regions and customer segments receive high discounts
--                    (above 20%), and how much profit do those discounts cost?
-- Why we checked: Query 03 showed that lines with discounts above 20% lose money in
--                 every category. The next question is where those discounts are given.
-- Insight: High discounts are concentrated by region, not by customer segment.
--          Region:
--          - Central gives by far the most high discounts: 27.68% of its lines are
--            discounted above 20%, vs. only 3.68% in West (about 7.5x more).
--          - Central also has the lowest margin (7.92%), and its high-discount lines
--            lose $52,392, more than its entire total profit ($39,706).
--          - West has the lowest discounts and the highest margin (14.94%).
--          - The link is not perfect: South has fewer high-discount lines than East
--            (10.56% vs. 16.16%) but a lower margin (11.93% vs. 13.49%).
--          Segment:
--          - The three segments get almost identical discounts (13.30%-14.16% of lines
--            above 20%), so customer type does not explain the losses.
--          - Consumer has the largest loss in dollars (-$71,890) only because it has
--            the most order lines, not because it is treated differently.
--          Check: high-discount losses add up to -$135,364 in both tables, matching query 03.
-- Next: drill into Central to find which states give the high discounts.
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