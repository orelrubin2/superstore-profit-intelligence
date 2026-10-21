-- Business question: Do high discounts lead to losses in every category,
--                    or only in Furniture?
-- Why we checked: In query 02, the two loss-making Furniture sub-categories (Tables,
--                 Bookcases) also had the highest average discounts. That raised a
--                 suspicion that discounts drive losses. But it could have been a
--                 Furniture-only problem, so we tested the same pattern in all categories.
-- Insight: The pattern holds in every category, so this is a company-wide discount issue.
--          - With no discount, 0% of order lines lose money in all three categories
--            (margins of 22.7%-34.0%).
--          - At 1-20%, lines stay profitable overall, but margins drop sharply
--            (e.g. Technology 34.0% -> 14.7%).
--          - Above 20%, every category loses money: 78%-100% of lines are loss-making,
--            and Office Supplies at >40% loses more than its revenue (-119% margin).
--          - Lines with discounts above 20% lose a combined $135,364, equal to about 47%
--            of the company's total profit ($286,409).
--          - Furniture is also weak at 1-20% (5.1% margin vs. 14.7%-16.5% elsewhere),
--            so discounts explain part, but not all, of its low profitability.
-- Caveats: Technology >40% is based on only 35 lines. This shows a strong association,
--          not proof that discounts cause the losses.
-- Next: test the discount effect while controlling for product and price (ML step),
--       and look at which regions and customer segments receive the high discounts.
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