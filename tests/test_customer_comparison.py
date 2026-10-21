"""Consistency tests for the step 3 comparison (src/customer_comparison.py).

These check that the comparison functions agree with each other and with
independent totals, so a wrong join or denominator would be caught.

Run from the project root (after `python -m src.build_database`):
    python -m pytest
"""

import duckdb
import numpy as np
import pytest

import src.customer_comparison as cc
from src.build_database import DB_PATH


@pytest.fixture(scope="module")
def con():
    """Read-only connection to the built database."""
    if not DB_PATH.exists():
        pytest.skip("database not built; run `python -m src.build_database` first")
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def customers(con):
    return cc.load_customers(con)


def test_overview_adds_up_to_company_totals(con, customers):
    """The three volume groups cover every customer and all profit."""
    overview = cc.group_overview(customers)
    customers_total, profit_total = con.execute(
        "SELECT COUNT(DISTINCT customer_id), SUM(profit) FROM orders_clean"
    ).fetchone()
    assert overview["customers"].sum() == customers_total
    assert np.isclose(overview["profit"].sum(), profit_total)
    assert np.isclose(overview["share_of_total_profit"].sum(), 1)


@pytest.mark.parametrize("dimension", ["discount_band", "category", "sub_category", "region", "state"])
def test_group_mix_shares_sum_to_one(con, dimension):
    """Within each group, the gross shares of a dimension sum to 1."""
    mix = cc.group_mix(con, dimension)
    sums = mix.groupby("volume_group")["gross_share"].sum()
    np.testing.assert_allclose(sums, 1)


def test_group_mix_matches_group_profit(con, customers):
    """Profit summed over any mix dimension equals the group's total profit."""
    overview = cc.group_overview(customers)
    mix = cc.group_mix(con, "state").groupby("volume_group")["profit"].sum()
    for group in cc.GROUPS:
        assert np.isclose(mix[group], overview.loc[group, "profit"])


def test_mix_vs_rate_actuals_match_overview(con, customers):
    """'Actual' profit per gross $ equals group profit / group gross sales."""
    overview = cc.group_overview(customers)
    result = cc.mix_vs_rate(con).set_index("scenario")["profit_per_gross_dollar"]
    for group in cc.GROUPS:
        expected = overview.loc[group, "profit"] / overview.loc[group, "gross_sales"]
        assert np.isclose(result[f"{group} actual"], expected)


def test_worst_lines_one_per_customer(con, customers):
    """Exactly one worst line per HV-loss customer, equal to worst_line_profit."""
    lines = cc.worst_lines(con).set_index("customer_id")
    hv_loss = customers[customers.volume_group == "HV-loss"].set_index("customer_id")
    assert set(lines.index) == set(hv_loss.index) and lines.index.is_unique
    np.testing.assert_allclose(lines.loc[hv_loss.index, "profit"], hv_loss["worst_line_profit"])


def test_segment_crosstab_counts(customers):
    """Crosstab counts add up to the volume groups; lift of the whole is 1."""
    table = cc.segment_crosstab(customers, "cluster")
    counts = customers["volume_group"].value_counts()
    for group in ["HV-loss", "HV-profit", "Other"]:
        assert table[group].sum() == counts[group]
    weighted_lift = (table["lift"] * table["share of all"]).sum()
    assert np.isclose(weighted_lift, 1)


def test_share_ci_brackets_estimate(con):
    """The bootstrap interval contains the point estimate and is reproducible."""
    first = cc.share_ci(con, "state", ["Texas", "Illinois"])
    second = cc.share_ci(con, "state", ["Texas", "Illinois"])
    assert first == second
    assert first["ci_low (pp)"] <= first["difference (pp)"] <= first["ci_high (pp)"]


def test_sensitivity_primary_row_matches_groups(con, customers):
    """The primary definition in the sensitivity table reproduces the main groups."""
    primary = cc.sensitivity(con, customers).iloc[0]
    counts = customers["volume_group"].value_counts()
    assert primary["HV-loss n"] == counts["HV-loss"]
    assert primary["HV-profit n"] == counts["HV-profit"]
