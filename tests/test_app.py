"""Smoke and consistency tests for the Streamlit dashboard.

Each page is run headless with Streamlit's AppTest; the displayed numbers are
compared with the database.

Run from the project root (after `python -m src.build_database`):
    python -m pytest
"""

from pathlib import Path

import duckdb
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.build_database import DB_PATH

TIMEOUT = 180   # first run may train the model
ENTRY_POINT = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def run_page(module: str) -> AppTest:
    """Run one dashboard page's render() headless."""
    def page(module):
        import importlib
        importlib.import_module(f"app.views.{module}").render()
    return AppTest.from_function(page, args=(module,), default_timeout=TIMEOUT).run()


def metric_values(at: AppTest) -> dict:
    return {m.label: m.value for m in at.metric}


def money(value: float) -> str:
    return f"−${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def percent(value: float) -> str:
    return f"−{abs(value) * 100:.1f}%" if value < 0 else f"{value * 100:.1f}%"


@pytest.fixture(scope="module")
def con():
    if not DB_PATH.exists():
        pytest.skip("database not built; run `python -m src.build_database` first")
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    yield connection
    connection.close()


def test_entry_point_runs():
    """streamlit_app.py starts and renders the default page without errors."""
    at = AppTest.from_file(str(ENTRY_POINT), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.title[0].value == "Exploratory data analysis"


def test_overview_matches_database(con):
    """Unfiltered key numbers equal the totals in orders_clean."""
    at = run_page("overview")
    assert not at.exception
    sales, profit, high = con.execute("""
        SELECT SUM(sales), SUM(profit), SUM(profit) FILTER (WHERE discount > 0.2) FROM orders_clean
    """).fetchone()
    values = metric_values(at)
    assert values["Sales"] == money(sales)
    assert values["Profit"] == money(profit)
    assert values["Profit on lines discounted >20%"] == money(high)   # -$135,364 (query 03)


def test_overview_filters(con):
    """Filtering to Texas shows Texas totals, and the state list follows the region."""
    at = run_page("overview")
    at.multiselect[1].set_value(["Central"]).run()
    assert "California" not in at.multiselect[2].options
    at.multiselect[2].set_value(["Texas"]).run()
    assert not at.exception
    profit = con.execute("SELECT SUM(profit) FROM orders_clean WHERE state = 'Texas'").fetchone()[0]
    assert metric_values(at)["Profit"] == money(profit)


def eda_kpis_sql(con, where: str = "TRUE") -> dict:
    sales, profit, orders, high = con.execute(f"""
        SELECT SUM(sales), SUM(profit), COUNT(DISTINCT order_id), AVG((discount > 0.2)::INT)
        FROM orders_clean WHERE {where}
    """).fetchone()
    return {"Sales": money(sales), "Profit": money(profit),
            "Profit margin": percent(profit / sales), "Orders": f"{orders:,}",
            "Lines discounted >20%": percent(high)}


def test_eda_matches_database(con):
    """Default view (2017) shows 2017 KPIs, and the sales change equals query 06's growth."""
    at = run_page("eda")
    assert not at.exception
    assert at.selectbox(key="eda_year").value == 2017
    assert metric_values(at) == eda_kpis_sql(con, "order_year = 2017")
    growth = con.execute(
        "SELECT sales_growth_pct FROM (SELECT order_year, ROUND((SUM(sales) - LAG(SUM(sales)) "
        "OVER (ORDER BY order_year)) / LAG(SUM(sales)) OVER (ORDER BY order_year) * 100, 2) "
        "AS sales_growth_pct FROM orders_clean GROUP BY order_year) WHERE order_year = 2017"
    ).fetchone()[0]                                                          # 20.36% (query 06)
    sales = next(m for m in at.metric if m.label == "Sales")
    assert sales.delta == f"{growth / 100:+.1%}"


def test_eda_filters(con):
    """Year + region + category filters give the same KPIs as SQL; 'All years' has no delta."""
    at = run_page("eda")
    at.selectbox(key="eda_year").set_value(2016).run()
    at.multiselect(key="eda_region").set_value(["Central"]).run()
    at.multiselect(key="eda_category").set_value(["Furniture"]).run()
    assert not at.exception
    assert metric_values(at) == eda_kpis_sql(
        con, "order_year = 2016 AND region = 'Central' AND category = 'Furniture'")
    at.selectbox(key="eda_year").set_value("All years").run()
    assert not at.exception
    assert metric_values(at) == eda_kpis_sql(con, "region = 'Central' AND category = 'Furniture'")
    assert not any(m.delta for m in at.metric)


def test_eda_queries_match_files(con):
    """run_query on unfiltered lines returns what the saved .sql files return on the database."""
    import pandas as pd
    from app import data
    from src.run_sql import execute_sql_file

    for file in ["03_discount_impact_by_category.sql", "04_high_discounts_by_region_segment.sql",
                 "08_shipping.sql", "10_monthly_trend.sql"]:
        for number, expected in execute_sql_file(con, data.SQL_DIR / file):
            got = data.run_query(file, number)
            pd.testing.assert_frame_equal(got, expected, check_dtype=False)
    shipping = data.run_query("08_shipping.sql")
    assert len(shipping) == 4 and shipping["order_lines"].sum() == 9993


def test_eda_every_year_renders():
    """Every Year option renders; 2014 has no previous year, so its cards show no delta."""
    at = run_page("eda")
    for year in ["All years", 2014, 2015, 2016]:
        at.selectbox(key="eda_year").set_value(year).run()
        assert not at.exception
    at.selectbox(key="eda_year").set_value(2014).run()
    assert not any(m.delta for m in at.metric)


def test_risk_checker_page():
    """The checker renders a probability and both rule verdicts for the default inputs."""
    at = run_page("risk_checker")
    assert not at.exception
    values = metric_values(at)
    risk = float(values["Predicted loss risk"].rstrip("%"))
    assert 0 <= risk <= 100
    assert values["Rule A: discount > 20%"] == "No flag"     # default discount is 20%


def test_risk_checker_warns_on_unseen_sub_category_discount():
    """Binders (the default) were never sold at a 30% discount, so the page says so."""
    at = run_page("risk_checker")
    assert not any("Binders were never sold" in i.value for i in at.info)   # default 20%
    at.select_slider[0].set_value(0.3).run()
    assert not at.exception
    assert any("Binders were never sold at a 30% discount" in i.value for i in at.info)


def test_explanation_adds_up_to_prediction():
    """Base value + SHAP contributions equals the model's predicted probability."""
    import src.loss_model as lm
    from app import data

    model = data.load_model()
    line = lm.build_line(model, "Binders", "Texas", 0.8, 3, 10.0, "Standard Class", "Consumer", 11)
    base, contributions = lm.explain(model["forest"], line, model["explainer"])
    predicted = model["forest"].predict_proba(line[lm.FEATURES])[0, 1]
    assert np.isclose(base + contributions.iloc[0].sum(), predicted, atol=1e-6)
    assert model["category_of"]["Binders"] == "Office Supplies"
    assert model["region_of"]["Texas"] == "Central"


def test_eda_state_map(con):
    """The map covers every state with a unique FIPS code, every metric renders, and the
    sentence under the map matches SQL (default year 2017)."""
    from app import data
    from app.views import eda

    states = eda.state_frame(data.run_query("09_state_profitability.sql"))
    assert len(states) == 49 and states["fips"].is_unique
    at = run_page("eda")
    for option in ["Profit", "Margin", "Sales"]:
        at.radio(key="eda_map_metric").set_value(option).run()
        assert not at.exception
    losing, winning = con.execute("""
        WITH per_state AS (
            SELECT state, SUM(profit) AS profit, COUNT(*) AS n,
                   SUM((discount > 0.2)::INT) AS high
            FROM orders_clean WHERE order_year = 2017 GROUP BY state
        )
        SELECT SUM(high) FILTER (WHERE profit < 0) / SUM(n) FILTER (WHERE profit < 0),
               SUM(high) FILTER (WHERE profit >= 0) / SUM(n) FILTER (WHERE profit >= 0)
        FROM per_state
    """).fetchone()
    expected = (f"Loss-making states discount **{percent(losing)}** of their lines above 20%, "
                f"profitable states **{percent(winning)}**.")
    assert any(m.value == expected for m in at.markdown)
