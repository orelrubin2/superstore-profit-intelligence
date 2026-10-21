-- Business question: How much of the total profit comes from the top customers,
--                    and are some customers unprofitable?
-- Why we checked: Queries 01-06 looked at products, discounts, regions and time.
--                 The last view is the customer: is profit concentrated in a few customers?
-- Insight: ___

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