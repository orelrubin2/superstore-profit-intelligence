"""Tests for RFM scoring and the R x F segment grid (src/segmentation.py).

Run from the project root (after `python -m src.build_database`):
    python -m pytest
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.build_database import DB_PATH
from src.segmentation import (
    CLUSTER_FEATURES,
    SEGMENT_ORDER,
    SEGMENT_RULES,
    check_cluster_table,
    check_rfm_table,
    cluster_customers,
    cluster_matrix,
    name_main_clusters,
    percentile_score,
    rfm_segment,
)


# ---------- unit tests on small, hand-checkable inputs ----------

def test_ties_share_a_score():
    """Equal values always get the same score, even when they straddle a quintile edge."""
    values = pd.Series([1, 2, 2, 2, 2, 3, 4, 5, 6, 7])
    scores = percentile_score(values)
    assert scores[values == 2].nunique() == 1


def test_scores_cover_one_to_five_and_follow_order():
    """Distinct values spread over 1..5, and a larger value never scores lower."""
    values = pd.Series(range(100))
    scores = percentile_score(values)
    assert sorted(scores.unique()) == [1, 2, 3, 4, 5]
    assert scores.is_monotonic_increasing
    assert (scores.value_counts() == 20).all()


def test_recency_is_inverted():
    """With higher_is_better=False the smallest value (most recent) scores 5."""
    days = pd.Series([1, 10, 100, 1000, 5000])
    scores = percentile_score(days, higher_is_better=False)
    assert scores.iloc[0] == 5 and scores.iloc[-1] == 1


def test_grid_covers_every_cell_exactly_once():
    """The six rules assign each of the 25 R x F cells to exactly one segment."""
    for r in range(1, 6):
        for f in range(1, 6):
            matches = [name for name, (rl, rh), (fl, fh) in SEGMENT_RULES
                       if rl <= r <= rh and fl <= f <= fh]
            assert len(matches) == 1, (r, f, matches)
    assert len(set(SEGMENT_ORDER)) == 6


def test_named_cells():
    """Spot-check the corners of the grid."""
    assert rfm_segment(5, 5) == "Champions"
    assert rfm_segment(5, 1) == "Promising"
    assert rfm_segment(3, 5) == "Loyal"
    assert rfm_segment(3, 1) == "Needs attention"
    assert rfm_segment(1, 5) == "Frequent, quiet lately"
    assert rfm_segment(1, 1) == "Infrequent, quiet lately"


# ---------- tests on the built database ----------

@pytest.fixture(scope="module")
def con():
    """Read-only connection to the built database."""
    if not DB_PATH.exists():
        pytest.skip("database not built; run `python -m src.build_database` first")
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    yield connection
    connection.close()


def test_rfm_table_valid(con):
    """One row per customer, scores in 1..5, no tie split, segments match the rules."""
    check_rfm_table(con)


def test_rfm_matches_customer_features(con):
    """R, F, M inputs in customer_rfm are copied unchanged from customer_features."""
    mismatches = con.execute("""
        SELECT COUNT(*) FROM customer_rfm r
        JOIN customer_features f USING (customer_id)
        WHERE r.recency_days <> f.recency_days
           OR r.frequency    <> f.frequency
           OR r.sales        <> f.sales
    """).fetchone()[0]
    assert mismatches == 0


def test_every_segment_used(con):
    """All six segments occur, and they add up to every customer."""
    counts = con.execute(
        "SELECT rfm_segment, COUNT(*) FROM customer_rfm GROUP BY 1"
    ).fetchall()
    assert {name for name, _ in counts} == set(SEGMENT_ORDER)
    total = con.execute("SELECT COUNT(*) FROM customer_features").fetchone()[0]
    assert sum(n for _, n in counts) == total


# ---------- K-Means ----------

def test_cluster_matrix_is_logged_and_scaled():
    """Skewed columns are log1p-transformed, then every column has mean 0 and std 1."""
    df = pd.DataFrame({
        "recency_days": [1, 10, 100, 1000],
        "frequency": [1, 2, 4, 8],
        "gross_sales": np.expm1([1.0, 2.0, 3.0, 4.0]),   # log1p gives 1, 2, 3, 4
        "high_discount_gross_share": [0.0, 0.1, 0.5, 1.0],
    })
    x = cluster_matrix(df, CLUSTER_FEATURES)
    np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(x.std(axis=0), 1, atol=1e-12)
    # log1p(gross_sales) is evenly spaced, so its z-scores are symmetric around 0
    np.testing.assert_allclose(x[:, 2], -x[::-1, 2], atol=1e-9)


def test_cluster_names_follow_profiles_not_label_numbers():
    """Swapping K-Means label numbers does not change which customer gets which name."""
    df = pd.DataFrame({
        "gross_sales": [100, 120, 5000, 5200, 4000, 4100],
        "high_discount_gross_share": [0.0, 0.0, 0.05, 0.02, 0.7, 0.8],
    })
    a = name_main_clusters(df, np.array([0, 0, 1, 1, 2, 2]))
    b = name_main_clusters(df, np.array([2, 2, 0, 0, 1, 1]))
    assert (a == b).all()
    assert list(a) == ["Quiet, low volume"] * 2 + ["Active, regular"] * 2 + ["Active, heavy-discount"] * 2


def test_cluster_table_valid(con):
    """One row per customer, k named clusters each, every cluster above the minimum size."""
    check_cluster_table(con)


def test_clustering_is_reproducible(con):
    """Refitting from customer_features gives exactly the stored clusters."""
    features = con.sql("SELECT * FROM customer_features").df()
    refit = cluster_customers(features).set_index("customer_id")
    stored = con.sql("SELECT * FROM customer_clusters").df().set_index("customer_id")
    assert (refit.loc[stored.index] == stored).all().all()
