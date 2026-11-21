"""Tests for the step 4b profit model (src/profit_model.py).

The leakage tests guarantee that the model reads only order-time columns from
orders_clean: nothing derived from profit, no IDs or names, and no step 3
customer table. Other tests check that baselines and the top-4 ranking learn
only from the training years.

Run from the project root (after `python -m src.build_database`):
    python -m pytest
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

import src.profit_model as pm
from src.build_database import DB_PATH
from src.loss_model import split_by_year

CUSTOMER_TABLES = ["customer_features", "customer_mix", "customer_rfm", "customer_clusters"]
PROFIT_DERIVED = {"profit", "profit_margin", "is_loss", "is_outlier_profit"}


@pytest.fixture(scope="module")
def con():
    if not DB_PATH.exists():
        pytest.skip("database not built; run `python -m src.build_database` first")
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def lines(con):
    return pm.load_lines(con)


def columns_of(con, table: str) -> set[str]:
    return {r[0] for r in con.execute(
        f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"
    ).fetchall()}


# ---------- leakage guard ----------

def test_no_forbidden_or_profit_derived_features():
    """No feature is the target, derived from it, an outlier flag, an ID or a name."""
    assert PROFIT_DERIVED <= pm.FORBIDDEN_FEATURES
    assert not set(pm.ALL_FEATURES) & pm.FORBIDDEN_FEATURES
    assert pm.TARGET not in pm.ALL_FEATURES
    assert not any(f.endswith("_id") or f.endswith("_name") for f in pm.ALL_FEATURES)


def test_reads_only_orders_clean():
    """The loading query touches orders_clean and no customer table."""
    sql = pm.LINES_SQL.lower()
    assert "from orders_clean" in sql
    assert not any(table in sql for table in CUSTOMER_TABLES)


def test_features_are_order_line_columns(con):
    """Every feature is a plain orders_clean column, none is a customer-level column."""
    line_columns = columns_of(con, "orders_clean")
    assert set(pm.ALL_FEATURES) <= line_columns
    for table in CUSTOMER_TABLES:
        assert not set(pm.ALL_FEATURES) & (columns_of(con, table) - line_columns), table


def test_models_never_see_the_target(lines):
    """Fitted pipelines take exactly the chosen features as input columns."""
    train, _ = split_by_year(lines)
    small = train.sample(500, random_state=0)
    forest = pm.make_forest(pm.ALL_FEATURES, n_estimators=5).fit(small[pm.ALL_FEATURES],
                                                                   small[pm.TARGET])
    assert list(forest.feature_names_in_) == pm.ALL_FEATURES


# ---------- split and baselines ----------

def test_split_by_year(lines):
    train, test = split_by_year(lines)
    assert set(train["order_year"]) == {2014, 2015, 2016}
    assert set(test["order_year"]) == {2017}
    assert len(train) + len(test) == len(lines)


def test_median_baseline_is_the_training_median(lines):
    train, test = split_by_year(lines)
    pred = pm.MedianBaseline().fit(train).predict(test)
    assert (pred == train["profit"].median()).all()


def test_margin_rule_learns_only_from_training_data(lines):
    """Baseline (b): cell margin = training SUM(profit) / SUM(sales); test profits are unused."""
    train, test = split_by_year(lines)
    rule = pm.MarginRuleBaseline().fit(train)
    before = rule.predict(test)
    scrambled = test.assign(profit=np.random.default_rng(0).permutation(test["profit"].to_numpy()))
    assert np.allclose(before, rule.predict(scrambled))
    cell = train[(train["discount_band"] == "0%") & (train["sub_category"] == "Binders")]
    row = test[(test["discount_band"] == "0%") & (test["sub_category"] == "Binders")].head(1)
    expected = row["sales"].iloc[0] * cell["profit"].sum() / cell["sales"].sum()
    assert np.isclose(rule.predict(row)[0], expected)


def test_margin_rule_falls_back_to_the_band():
    """A band x sub-category never seen in training uses the band's margin."""
    train = pd.DataFrame({"discount_band": ["0%", "0%"], "sub_category": ["A", "B"],
                          "sales": [100.0, 300.0], "profit": [10.0, 90.0]})
    new = pd.DataFrame({"discount_band": ["0%"], "sub_category": ["C"], "sales": [50.0]})
    assert np.isclose(pm.MarginRuleBaseline().fit(train).predict(new)[0], 50 * 100 / 400)


def test_ranking_ignores_the_test_year(lines):
    """The top-4 ranking is the same when 2017's profit is scrambled."""
    params = {"max_depth": 8, "min_samples_leaf": 5, "n_estimators": 50}
    scrambled = lines.copy()
    in_2017 = scrambled["order_year"] == 2017
    scrambled.loc[in_2017, "profit"] = np.random.default_rng(0).permutation(
        scrambled.loc[in_2017, "profit"].to_numpy())
    a = pm.rank_features(lines, params)
    b = pm.rank_features(scrambled, params)
    pd.testing.assert_frame_equal(a, b)


# ---------- margin x sales model ----------

def test_sales_can_be_derived_from_page_inputs(lines):
    """sales = list price x quantity x (1 - discount), so the page can compute it."""
    assert np.allclose(pm.derived_sales(lines), lines["sales"], atol=1e-6)
    assert pm.sales_derivable(pm.PAGE_FEATURES + ["unit_price"])
    assert not pm.sales_derivable(pm.PAGE_FEATURES)


def test_margin_forest_needs_sales():
    with pytest.raises(ValueError):
        pm.MarginForest(pm.PAGE_FEATURES)


def test_margin_forest_shap_adds_up_in_dollars(lines):
    """sales x base + the dollar SHAP values = the predicted profit."""
    train, test = split_by_year(lines)
    features = pm.PAGE_FEATURES + ["unit_price"]
    model = pm.MarginForest(features, max_depth=6, n_estimators=20).fit(train)
    sample = test.head(50)
    base, per_feature = pm.explain_margin_forest(model, sample)
    assert set(per_feature.columns) == set(features)
    assert np.allclose(base + per_feature.sum(axis=1).to_numpy(), model.predict(sample), atol=1e-6)


# ---------- helpers ----------

def test_original_feature_mapping():
    assert pm.original_feature("cat__sub_category_Binders") == "sub_category"
    assert pm.original_feature("cat__category_Furniture") == "category"
    assert pm.original_feature("cat__state_New York") == "state"
    assert pm.original_feature("num__sales") == "sales"
    assert pm.original_feature("num__unit_price") == "unit_price"


def test_metrics_on_toy_data():
    result = pm.metrics(np.array([0.0, 10.0, -10.0]), np.array([0.0, 0.0, 0.0]))
    assert np.isclose(result["mae"], 20 / 3)
    assert np.isclose(result["rmse"], np.sqrt(200 / 3))
    assert result["median_ae"] == 10


def test_bootstrap_mae_difference_sign():
    """A perfect predictor beats a constant one: difference and interval below 0."""
    toy = pd.DataFrame({"order_id": [f"o{i // 2}" for i in range(40)],
                        "profit": np.arange(40, dtype=float)})
    result = pm.bootstrap_mae_difference(toy, toy["profit"].to_numpy(), np.zeros(40), n_boot=200)
    assert result["difference"] < 0 and result["ci_high"] < 0
