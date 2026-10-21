---
name: verify-project
description: Verify the Superstore project end to end - rebuild the database from the raw CSV, run all tests, and check that the key numbers in the README still match the queries and data. Use before a release or commit of analysis changes, or when the user asks to verify, re-check or health-check the project.
---

# Verify the project

Confirm that the pipeline rebuilds, the tests pass, and the README's headline
numbers are still true. Report results faithfully: if something fails, show the
output; do not fix analysis text without the user's approval.

## 1. Rebuild and test
Run from the project root with the virtual environment's Python
(`.venv\Scripts\python.exe` on Windows):
1. `python -m src.build_database` - must end with "All checks passed".
2. `python -m pytest` - report the pass/fail count. The MCP tests need
   `requirements-mcp.txt`; if `mcp` is missing they are skipped, say so.
3. `git status --short` - the rebuild must not change tracked files (in particular
   `data/processed/orders_clean.parquet`).

## 2. Check the README's key numbers
Recompute each headline number with a query (MCP `run_sql`, or `python -m src.run_sql`)
and compare it with the README. At minimum:

| README claim | How to recompute |
|---|---|
| Lines above 20% discount lose −$135,364, ~47% of total profit ($286,409) | `SUM(profit)` where `discount > 0.2`; total `SUM(profit)` |
| Furniture ~a third of sales, ~6% of profit, 2.5% margin | category shares in `orders_clean` (saved query `01`) |
| Central 27.68% vs. West 3.68% of lines above 20% discount | saved query `04` |
| Texas and Illinois account for all $52,392 of Central's high-discount loss | saved query `05` |
| Top 20% of customers = 81.90% of profit; 155 of 793 unprofitable | saved query `07` |
| 43 HV-loss customers, −$52,784 | `customer_features` grouped by `volume_group` |
| HV-loss 73% vs. HV-profit 15% of gross sales above 20% discount | `customer_mix` (discount_band) joined to `customer_features` |

Step 4 model numbers (PR-AUC 0.75 / 0.92 / 0.95 / 0.96, 89% of loss dollars) come from
`notebooks/04_loss_risk_model.ipynb`; check the notebook's saved outputs instead of
retraining, unless the user asks for a full rerun (about 10 minutes).

## Output
A checklist: rebuild ✅/❌, tests (n passed / failed / skipped), working tree clean
✅/❌, then each README number ✅ matches or ❌ with the actual value. End with a
one-line verdict.
