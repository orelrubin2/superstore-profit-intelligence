---
name: sql-review
description: Review a business query in sql/*.sql for the Superstore project - check it against the project's SQL conventions, run it, and verify that every number in its header comments matches the result. Use when the user asks to review, check or verify a SQL file or its insight.
---

# SQL review

Review one `.sql` file the way step 2 was reviewed: correctness first, then
conventions, then whether the written insight matches the data. **Do not rewrite
the query or the insight yourself**; report findings and suggest changes, and let
the user decide.

## 1. Run it
- Run the file with `python -m src.run_sql sql/<file>.sql`, or with the MCP tool
  `run_saved_query("<number>")` for a saved query. Report any error first.

## 2. Check the SQL
| Check | Correct pattern |
|---|---|
| Margins | `SUM(profit) / SUM(sales)`, not `AVG(profit_margin)` (average of ratios over-weights small lines) |
| Order counts | `COUNT(DISTINCT order_id)`; one row of `orders_clean` is an order **line** |
| High discount | `discount > 0.2`, consistent with queries 03-05 |
| Discount bands | ordered by `MIN(discount)`; `discount_band` text sorts alphabetically |
| Shares | window over the right partition, e.g. `SUM(SUM(x)) OVER ()` |
| Reserved words | `"check"` and `"column"` quoted in `data_quality_log` |
| Dates | `TIMESTAMP`; `year()` / `date_trunc()` as needed |
| Filters | `WHERE` vs. `HAVING` used correctly; no silent row loss from inner joins |

Look also for logic errors: wrong grouping level, a filter that changes the
question, division by zero, NULL handling.

## 3. Check the header
The header should have `Business question:`, `Insight:` and, where relevant,
`Why we checked:`, `Caveats:`, `Next:`.
- Verify **every number** in the insight against the result, including sums and
  differences computed from it (recompute them with a query if needed). List each
  as ✅ matches or ❌ with the actual value.
- Flag claims the result does not support (causal wording, "all", "only", rankings).
- Check that the header describes *this* query (a header pasted from another file
  is a known past mistake).

## Output
A table of findings ordered by severity (wrong numbers and logic errors first, then
conventions, then wording), each with the line and a suggested fix. End with a
one-line verdict: ready to commit, or what must change first.
