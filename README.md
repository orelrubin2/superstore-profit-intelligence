# Superstore Profit Intelligence

An end-to-end retail analytics project on the Sample Superstore dataset (9,994 US
order lines, 2014–2017). It covers data cleaning, a DuckDB star schema, SQL business
analysis, and next machine learning, a dashboard and an AI query layer.

**Core business question: where does the company lose money, and how can it grow profit?**

## Key findings (SQL analysis)

- **Discounts above 20% lose money in every category.** With no discount, no order lines
  lose money. Above 20%, 78%–100% of lines lose money, for a combined **−$135,364**:
  about **47% of total profit** ($286,409).
- **Furniture is the weak category.** It brings about a third of sales but only **~6% of
  profit** (2.5% margin vs. ~17% elsewhere). Tables and Bookcases lose money and carry
  the highest average discounts (26.13% and 21.11%).
- **High discounts depend on region, not on customer segment.** **27.68%** of Central's
  lines are discounted above 20%, vs. **3.68%** in West. The three customer segments get
  almost identical discounts (13.30%–14.16%).
- **Inside Central it comes down to two states.** Texas and Illinois are the only Central
  states that discount (37.02% and 39.00% on average). Both lose money, and their
  high-discount lines account for all **$52,392** of Central's high-discount loss.
- **Growth is real, but 2017 was less profitable.** Sales grew about **+51.5%** from 2014
  to 2017. In 2017 the margin fell from 13.43% to 12.74%, while the share of high-discount
  lines stayed at ~14% every year.
- **Profit is concentrated in few customers.** The top 20% of customers generate
  **81.90%** of profit, while **155 of 793** customers lose money.

**So far:** most losses come from discounts above 20%, given mainly in specific states
rather than to specific customer types, under a policy that has stayed the same for
four years.

| Discount vs. margin | Central region by state |
|---|---|
| ![Profit margin by discount band](reports/figures/02_03_discount_impact_by_category.png) | ![Central states margin](reports/figures/02_05_central_states.png) |
| **Category shares** | **Customer Pareto** |
| ![Category share of sales and profit](reports/figures/02_01_category_profitability.png) | ![Customer profit by decile](reports/figures/02_07_customer_pareto.png) |

Full walkthrough with every query, table and chart:
[`notebooks/02_sql_findings.ipynb`](notebooks/02_sql_findings.ipynb).
Data quality and cleaning: [`notebooks/01_data_quality.ipynb`](notebooks/01_data_quality.ipynb).

## Project structure

```
data/raw/              original CSV (never modified)
data/processed/        orders_clean.parquet + superstore.duckdb (built locally)
src/cleaning.py        load, validate and clean the raw data; data-quality log
src/build_database.py  DuckDB star schema + integrity checks
src/run_sql.py         run any .sql file and print the result
sql/01-07_*.sql        business queries, each with question + insight in its header
sql/SCHEMA.md          every table, column and type in the database
tests/                 pytest checks: unique keys, joins and totals reconcile with raw data
notebooks/             01 data quality, 02 SQL findings
reports/figures/       charts used in this README
```

**Data model:** `fact_order_lines` with `dim_customer`, `dim_product`, `dim_location` and
`dim_date`, plus a wide `orders_clean` table and an order-level `orders` view.

**Tech stack:** Python 3.12, pandas, DuckDB (SQL), matplotlib, Jupyter, pytest, pyarrow.
All free and open source.

## How to run (Windows PowerShell)

```powershell
git clone https://github.com/orelrubin2/superstore-profit-intelligence.git
cd superstore-profit-intelligence
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.build_database                            # clean data + build DuckDB (~3 s)
python -m pytest                                        # integrity tests
python -m src.run_sql sql/03_discount_impact_by_category.sql   # run any query
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Roadmap

| Step | Status |
|---|---|
| 1. Data cleaning and DuckDB database | ✅ Done |
| 2. SQL business analysis | ✅ Done |
| 3. Deeper analysis: discount impact, RFM segmentation, cohorts | Planned |
| 4. ML: loss-risk classifier with SHAP explanations, monthly sales forecast | Planned |
| 5. Streamlit dashboard | Planned |
| 6. AI layer: Agent Skills and an MCP server for querying the data with Claude | Planned |

## How I worked with AI

I led the project and used Claude as a collaborator for both thinking and execution:

- **My role:** I set the goal and the rules (every finding must come from code in
  this repository, no assumed results), reviewed and approved each plan, chose
  which direction to take after each result, and checked every number and insight
  against the query output before committing.
- **Claude as an analysis partner:** we developed the drill-down together
  (category, sub-category, discount, region, state, time, customers). Claude
  proposed next questions and drafted queries 03-07, including the more advanced
  SQL (window functions, `LAG`, CTEs, `NTILE`), with explanations. I wrote the early
  queries (01-02) myself, and I ran, verified and interpreted all seven.
- **Claude Code as a pair programmer:** it built the pipeline under my review:
  cleaning, the DuckDB schema, integrity tests, the `run_sql` runner and the notebooks.

The project rules for the AI are in [`CLAUDE.md`](CLAUDE.md).

## Data

Sample Superstore
(Kaggle). The file is Windows-1252 encoded. 449 postal codes lose their leading zero
if read as numbers, and one duplicate row is removed; all of this is documented in
`notebooks/01_data_quality.ipynb`.

License: MIT.
