-- Business question: How do sales and profit move month by month, and which months
--                    of the year are strong or weak?
-- Why we checked: Query 06 showed yearly growth, but yearly totals hide seasonality.
--                 Planning stock, staff and promotions needs the monthly pattern.
-- Insight: Sales are strongly seasonal, and profit is much noisier than sales.
--          - September, November and December are the strongest months every year:
--            on average 13.66%, 15.40% and 14.41% of the year's sales. September to
--            December (4 of 12 months) bring 51.86% of annual sales.
--          - January and February are the weakest (3.96% and 2.50%); together only
--            6.46% of annual sales. March is a smaller spring peak (9.06%).
--          - The pattern repeats each year: November is never below 13.04% of its
--            year's sales and February never above 3.77%.
--          - The busiest month is not the most profitable: November's margin (10.06%)
--            is below the company's 12.47%, while October (15.87%) and February
--            (17.23%) earn more per dollar. Profit swings more than sales: two months
--            lost money (2014-07: -$841, 2015-01: -$3,281).
--          - The highest monthly sales were in 2017-11 ($118,448).
-- Caveats: Four years means only four values per calendar month; the share range
--          (min/max) shows how stable each month is.
-- Next: check whether the low November margin comes from higher discounts in the
--       peak season (compare pct_high_discount_lines by month).

-- Part 1: monthly trend (one row per year-month)
SELECT
    T.order_ym,
    T.order_year,
    T.order_month,
    COUNT(*)                                                           AS order_lines,
    COUNT(DISTINCT T.order_id)                                         AS orders,
    ROUND(SUM(T.sales), 2)                                             AS total_sales,
    ROUND(SUM(T.profit), 2)                                            AS total_profit,
    ROUND(SUM(T.profit) / SUM(T.sales) * 100, 2)                       AS profit_margin_pct,
    ROUND(AVG(CASE WHEN T.discount > 0.2 THEN 1 ELSE 0 END) * 100, 2)   AS pct_high_discount_lines
FROM orders_clean AS T
GROUP BY T.order_ym, T.order_year, T.order_month
ORDER BY T.order_ym;


-- Part 2: seasonality (one row per calendar month, averaged over the years)
WITH monthly AS (
    -- Step 1: sales and profit per year and calendar month
    SELECT
        T.order_year,
        T.order_month,
        SUM(T.sales)  AS sales,
        SUM(T.profit) AS profit
    FROM orders_clean AS T
    GROUP BY T.order_year, T.order_month
),
shares AS (
    -- Step 2: each month's share of its year's sales
    SELECT
        *,
        sales / SUM(sales) OVER (PARTITION BY order_year) AS share_of_year
    FROM monthly
)
-- Step 3: average over the years
SELECT
    order_month,
    COUNT(*)                                       AS years,
    ROUND(AVG(sales), 2)                           AS avg_monthly_sales,
    ROUND(AVG(profit), 2)                          AS avg_monthly_profit,
    ROUND(SUM(profit) / SUM(sales) * 100, 2)       AS profit_margin_pct,
    ROUND(AVG(share_of_year) * 100, 2)             AS avg_pct_of_year_sales,
    ROUND(MIN(share_of_year) * 100, 2)             AS min_pct_of_year_sales,
    ROUND(MAX(share_of_year) * 100, 2)             AS max_pct_of_year_sales
FROM shares
GROUP BY order_month
ORDER BY order_month;
