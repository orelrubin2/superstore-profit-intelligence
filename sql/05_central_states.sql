-- Insight: Central's discount problem comes from just two states: Texas and Illinois.
--          - They are the only Central states that give discounts at all (average
--            37.02% and 39.00%); the other 11 states give almost none (0%-0.71%).
--          - Both lose money (-$25,729 and -$12,608; margins of about -15%), while
--            every other Central state is profitable with margins of 24.66%-36.24%.
--          - Their high-discount lines lose $33,955 + $18,437 = $52,392, which is all of
--            Central's high-discount loss from query 04.
--          - Texas is also Central's largest state by sales ($170,188), so the problem
--            sits in one of the region's most important markets.
--          This suggests discounts are set by state-level policy, not by product or
--          customer. Caveat: several states have very few order lines ( North Dakota: 7).
-- Next: check whether the same pattern appears in other regions, and follow how
--       sales and profit changed over time.
SELECT
    T.state,
    COUNT(*)                                                           AS order_lines,
    ROUND(AVG(T.discount) * 100, 2)                                    AS avg_discount_pct,
    ROUND(AVG(CASE WHEN T.discount > 0.2 THEN 1 ELSE 0 END) * 100, 2)   AS pct_high_discount_lines,
    ROUND(SUM(T.sales), 2)                                             AS total_sales,
    ROUND(SUM(T.profit), 2)                                            AS total_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)                       AS profit_margin_pct,
    ROUND(SUM(CASE WHEN T.discount > 0.2 THEN T.profit ELSE 0 END), 2) AS profit_from_high_discounts
FROM orders_clean AS T
WHERE T.region = 'Central'
GROUP BY T.state
ORDER BY total_profit ASC;