# Superstore Profit Intelligence

## Project goal
Portfolio project for Data Analyst roles. An end-to-end retail
analytics project on the Sample Superstore dataset: data cleaning, SQL analysis,
customer segmentation, machine learning (loss-risk prediction), an interactive
dashboard, and AI-assisted analysis via Claude Code, Agent Skills and an MCP server.

Core business question: **Where does the company lose money, and how can it
grow profit?** All findings must come from code in this repository.
Do not assume results in advance.

## Dataset
- File: `data/raw/superstore.csv` (US retail orders, Kaggle "Sample Superstore")
- 9,994 rows, 21 columns, orders from 2014-01-03 to 2017-12-30
- One row = one order line (a product within an order), not a whole order
- Encoding: Windows-1252 (read with `encoding="cp1252"`); latin-1 turns the
  file's smart quotes (bytes 0x93/0x94) into invisible control characters
- Dates (`Order Date`, `Ship Date`) are strings in M/D/YYYY format
- Key columns: Order ID, Order Date, Ship Date, Ship Mode, Customer ID,
  Segment, Region, State, City, Category, Sub-Category, Product ID,
  Sales, Quantity, Discount (0 to 0.8), Profit (can be negative)

## Project structure
- `data/raw/` – original data. **Never modify these files.**
- `data/processed/` – cleaned data and the DuckDB database
- `sql/` – business-question queries (`.sql` files), run with DuckDB
- `src/` – reusable Python code (cleaning, features, models)
- `notebooks/` – exploration and analysis notebooks
- `reports/figures/` – saved charts used in the README

## Environment
- Windows, PowerShell, VS Code
- Python 3.12 in a virtual environment: activate with `.venv\Scripts\Activate.ps1`
- Dependencies are listed in `requirements.txt` with exact versions (it is also what
  Streamlit Community Cloud installs for the live dashboard). When adding a package,
  install it inside `.venv` and pin it in `requirements.txt`.
- The MCP server's extra dependency lives in `requirements-mcp.txt` (it includes
  `requirements.txt`), so the dashboard's install stays unchanged.

## Conventions
- Use relative paths from the project root (e.g. with `pathlib`), never
  absolute paths like `C:\Users\...`
- Save every chart to `reports/figures/` as PNG, don't rely on `plt.show()`
- Every analysis should end with a short written business insight, not just a table or chart
- Code and comments in English, with clear names and short docstrings
- Use only free, open-source tools. No paid APIs or services.
- Before large changes, explain the plan briefly and keep steps small
- In step 2 (SQL business analysis) the user leads the SQL analysis: Claude may
  propose questions and draft queries with explanations, but the user runs,
  verifies and interprets every query before committing it
- From step 3 on, the user proposes the analysis questions and the direction;
  Claude may refine them, draft code and explain it, and the user verifies every
  result before committing

## Roadmap
1. Data cleaning and a DuckDB database
2. SQL business analysis (profitability, discounts, regions, customers)
3. Customer segmentation: RFM scoring and K-Means clustering
   - Business question: "Who are our high-volume customers that generate net
     losses, and how do they differ from profitable high-volume customers in
     recency, frequency, discounts, products and regions?"
4. ML: loss-risk classifier with Random Forest, compared with a simple discount
   rule baseline and Logistic Regression; SHAP to rank the strongest predictors
   (Random Forest's built-in importance is biased toward high-cardinality
   features such as state)
   - Business question: "Which features known at order time (e.g. product
     category, region, ship mode, discount level) are the strongest predictors of
     loss-making order lines, and can we flag high-risk transactions before
     fulfillment better than a simple discount rule?"
5. Streamlit dashboard
6. AI layer: Agent Skills and an MCP server for querying the data with Claude

## Future ideas
Not on the roadmap; only if time allows after step 6.
- Monthly sales forecast