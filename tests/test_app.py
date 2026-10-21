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
    assert at.title[0].value == "Where does the company lose money?"


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


def test_customers_page(con):
    at = run_page("customers")
    assert not at.exception
    hv_loss = con.execute(
        "SELECT COUNT(*) FROM customer_features WHERE volume_group = 'HV-loss'").fetchone()[0]
    assert metric_values(at)["HV-loss customers"] == f"{hv_loss:,}"


def test_risk_checker_page():
    """The checker renders a probability and both rule verdicts for the default inputs."""
    at = run_page("risk_checker")
    assert not at.exception
    values = metric_values(at)
    risk = float(values["Predicted loss risk"].rstrip("%"))
    assert 0 <= risk <= 100
    assert values["Rule A: discount > 20%"] == "No flag"     # default discount is 20%


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
