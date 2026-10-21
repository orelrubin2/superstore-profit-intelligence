---
name: profit-drilldown
description: Drill into where a slice of the Superstore business makes or loses money (e.g. "Furniture in the East", "Texas in 2017", "Binders"). Use when the user asks why profit is low somewhere, where losses come from, or for a profitability breakdown of a category, region, state, segment or year.
---

# Profit drill-down

Answer "where does this slice lose money, and why?" with the project's standard
drill-down, using only numbers returned by the `superstore` MCP tools.

## Rules
- Every number in the answer must come from a tool result in this session. Never
  estimate or reuse numbers from memory. Anything not computed is labeled
  *inference*.
- Margins are `SUM(profit) / SUM(sales)`, never `AVG(profit_margin)`. Count orders
  with `COUNT(DISTINCT order_id)`. "High discount" means `discount > 0.2`.
- Query `orders_clean` (one row per order line). Call `describe_table` first if a
  column name is uncertain.

## Steps
1. **Restate the slice** as a SQL filter (e.g. `category = 'Furniture' AND region = 'East'`,
   `order_year = 2017`). If the request is ambiguous, ask once.
2. **Size it** with `run_sql`: order lines, orders, sales, profit, margin, and the
   share of company sales and profit it represents.
3. **Drill down** with one `run_sql` call per level, each showing sales, profit,
   margin and share of loss-making lines (`AVG(is_loss::INT)`), sorted by profit:
   1. sub-category
   2. discount band (`discount_band`; sort by `MIN(discount)`, the text sorts wrongly)
   3. state
   4. customers: the five largest loss-making customers in the slice
4. **Isolate the discount effect**: profit on lines with `discount > 0.2` vs. the rest.
5. **Compare with the company baseline** where useful (margin, high-discount share).
   Saved query `03` (`run_saved_query("03")`) has margins by discount band per category.

## Output
- A short answer first: where the slice loses money and the main driver, with the
  key numbers.
- Then one compact table per drill-down level (only the rows that matter).
- End with a 2-4 sentence business insight and any caveats (small counts below ~30
  lines, association not cause). Label recommendations as inferences.
