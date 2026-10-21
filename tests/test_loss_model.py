"""Tests for the step 4 loss-risk model (src/loss_model.py).

The leakage tests guarantee that the model reads only order-time columns from
orders_clean and never the step 3 customer tables, which summarize future orders.

Run from the project root (after `python -m src.build_database`):
    python -m pytest
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

import src.loss_model as lm
from src.build_database import DB_PATH

CUSTOMER_TABLES = ["customer_features", "customer_mix", "customer_rfm", "customer_clusters"]


@pytest.fixture(scope="module")
def con():
    if not DB_PATH.exists():
        pytest.skip("database not built; run `python -m src.build_database` first")
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def lines(con):
    return lm.load_lines(con)


# ---------- leakage ----------

def test_no_forbidden_features():
    """No feature defines the target, leaks the label, or comes from step 3."""
    assert not set(lm.FEATURES) & lm.FORBIDDEN_FEATURES


def test_reads_only_orders_clean():
    """The loading query touches orders_clean and no customer table."""
    sql = lm.LINES_SQL.lower()
    assert "from orders_clean" in sql
    assert not any(table in sql for table in CUSTOMER_TABLES)


def test_step3_customer_columns_not_used(con):
    """No feature is a step 3 customer-level column (beyond plain line columns like quantity)."""
    line_columns = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'orders_clean'"
    ).fetchall()}
    for table in CUSTOMER_TABLES:
        customer_only = {r[0] for r in con.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"
        ).fetchall()} - line_columns
        assert not set(lm.FEATURES) & customer_only, table
    assert set(lm.FEATURES) <= line_columns


# ---------- split ----------

def test_split_by_year(lines):
    """Train is 2014-2016, test is 2017, and together they are every line exactly once."""
    train, test = lm.split_by_year(lines)
    assert set(train["order_year"]) == {2014, 2015, 2016}
    assert set(test["order_year"]) == {2017}
    assert len(train) + len(test) == len(lines)
    assert not set(train["row_id"]) & set(test["row_id"])


# ---------- baselines ----------

def test_baseline_a_is_the_discount_rule(lines):
    scores = lm.baseline_a_scores(lines)
    assert ((scores == 1) == (lines["discount"] > 0.2)).all()


def test_baseline_b_learns_only_from_training_data(lines):
    """Changing test labels does not change Baseline B's test scores."""
    train, test = lm.split_by_year(lines)
    model = lm.BaselineB(0.3).fit(train)
    shuffled = test.assign(is_loss=np.random.default_rng(0).permutation(test["is_loss"]))
    np.testing.assert_array_equal(model.scores(test), model.scores(shuffled))
    # mid-band score is the training loss rate of the sub-category in that band
    mid_train = train[lm.in_mid_band(train)]
    binders = mid_train.loc[mid_train.sub_category == "Binders", "is_loss"].mean()
    row = test[lm.in_mid_band(test) & (test.sub_category == "Binders")].head(1)
    assert np.isclose(model.scores(row)[0], binders)


def test_baseline_b_includes_rule_a(lines):
    """Every line Baseline A flags, Baseline B flags too; undiscounted lines score 0."""
    train, test = lm.split_by_year(lines)
    model = lm.BaselineB(0.5).fit(train)
    assert model.flags(test)[test["discount"] > 0.2].all()
    assert (model.scores(test)[test["discount"] == 0] == 0).all()


# ---------- evaluation helpers ----------

def test_flag_top_flags_exactly_n():
    scores = np.array([0.1, 0.9, 0.5, 0.9, 0.0])
    flags = lm.flag_top(scores, 2)
    assert flags.sum() == 2 and flags[1] and flags[3]


def test_operating_point_on_toy_data():
    toy = pd.DataFrame({"is_loss": [1, 1, 0, 0], "profit": [-100.0, -300.0, 50.0, 10.0]})
    result = lm.operating_point(toy, np.array([True, False, True, False]))
    assert result["precision"] == 0.5 and result["recall"] == 0.5
    assert np.isclose(result["loss_dollars_caught"], 0.25)


def test_original_feature_mapping():
    assert lm.original_feature("cat__sub_category_Binders") == "sub_category"
    assert lm.original_feature("cat__category_Furniture") == "category"
    assert lm.original_feature("cat__state_New York") == "state"
    assert lm.original_feature("num__discount") == "discount"


def test_grouped_shap_keeps_additivity(lines):
    """Summing one-hot SHAP columns per feature keeps each line's total SHAP unchanged,
    and base value + total SHAP equals the predicted loss probability."""
    import shap

    train, test = lm.split_by_year(lines)
    forest = lm.make_forest(max_depth=6, n_estimators=20).fit(train[lm.FEATURES], train[lm.TARGET])
    sample = test.head(200)
    per_feature, ranking = lm.grouped_shap(forest, sample)
    assert set(per_feature.columns) == set(lm.FEATURES) == set(ranking["feature"])

    model = forest.named_steps["model"]
    base = shap.TreeExplainer(model).expected_value
    base = base[1] if np.ndim(base) else base
    predicted = forest.predict_proba(sample[lm.FEATURES])[:, 1]
    np.testing.assert_allclose(base + per_feature.sum(axis=1), predicted, atol=1e-6)
