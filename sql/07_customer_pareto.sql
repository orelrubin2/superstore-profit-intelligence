-- Business question: How much of the total profit comes from the top customers,
--                    and are some customers unprofitable?
-- Why we checked: Queries 01-06 looked at products, discounts, regions and time.
--                 The last view is the customer: is profit concentrated in a few customers?
-- Insight: Profit is highly concentrated, and about 1 in 5 customers loses money.
--          - The top 10% of customers (80) generate 57.69% of total profit, and the
--            top 20% generate 81.90%, a classic Pareto (80/20) pattern.
--          - 155 of 793 customers (deciles 9-10) are unprofitable. Profitable customers
--            earn 124.86% of total profit, and the unprofitable ones erase about a
--            quarter of it (cumulative share falls from 124.86% back to 100%).
--          - The bottom decile is not small: it bought $278,279, the third-largest
--            sales of any decile, yet lost $65,754. These are big buyers that lose money.
-- Hypothesis: the unprofitable customers buy mostly at high discounts (links to 03-05).
-- Next: test this in the customer segmentation step (RFM) and the ML model.

WITH customer_profit AS (
    -- Step 1: one row per customer with their total sales and profit
    SELECT
        T.customer_id,
        SUM(T.sales)  AS sales,
        SUM(T.profit) AS profit
    FROM orders_clean AS T
    GROUP BY T.customer_id
),
customer_deciles AS (
    -- Step 2: split customers into 10 equal groups, from most to least profitable
    SELECT
        *,
        NTILE(10) OVER (ORDER BY profit DESC) AS decile
    FROM customer_profit
)
-- Step 3: summarize each decile
SELECT
    decile,
    COUNT(*)                                                         AS customers,
    ROUND(SUM(sales), 2)                                             AS total_sales,
    ROUND(SUM(profit), 2)                                            AS total_profit,
    ROUND(SUM(profit) / SUM(SUM(profit)) OVER () * 100, 2)           AS pct_of_profit,
    ROUND(SUM(SUM(profit)) OVER (ORDER BY decile)
          / SUM(SUM(profit)) OVER () * 100, 2)                       AS cumulative_pct_of_profit,
    SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END)                      AS unprofitable_customers
FROM customer_deciles
GROUP BY decile
ORDER BY decile;