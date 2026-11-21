-- Business question: Which states lose money across all regions, and are they the
--                    states that give high discounts?
-- Why we checked: Query 05 found that Texas and Illinois cause Central's losses through
--                 high discounts. We need to know whether the same pattern appears in
--                 the other regions.
-- Insight: Losses sit in 10 of 49 states, and every one of them discounts heavily.
--          - The 10 loss-making states lose $98,235 together and bring 30.71% of sales.
--            They are spread over all four regions: Texas and Illinois (Central), Ohio
--            and Pennsylvania (East), North Carolina, Tennessee and Florida (South),
--            Colorado, Arizona and Oregon (West).
--          - All 10 have average discounts of 28.35%-39.00%; 34.64% of their lines are
--            discounted above 20%, vs. 0.80% of lines in the 39 profitable states.
--          - Large markets are among them: Texas ($170,188 in sales, -$25,729) and
--            Pennsylvania ($116,512, -$15,560) are the 3rd and 5th largest states.
--          - Profit is concentrated: California and New York bring 33.46% of sales but
--            52.52% of profit, with low average discounts (7.28% and 5.53%).
--          This extends query 05: the high-discount policy is state-level in every
--          region, not a Central problem.
-- Caveats: Several profitable states have very few lines (e.g. Wyoming: 1, West
--          Virginia: 4), so their margins are not reliable.
-- Next: show the state view in the EDA dashboard next to the regions (query 04).
SELECT
    T.region,
    T.state,
    COUNT(*)                                                           AS order_lines,
    ROUND(SUM(T.sales), 2)                                             AS total_sales,
    ROUND(SUM(T.profit), 2)                                            AS total_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)                       AS profit_margin_pct,
    ROUND(AVG(T.discount) * 100, 2)                                    AS avg_discount_pct,
    ROUND(AVG(CASE WHEN T.discount > 0.2 THEN 1 ELSE 0 END) * 100, 2)   AS pct_high_discount_lines
FROM orders_clean AS T
GROUP BY T.region, T.state
ORDER BY total_profit ASC;
