"""Tests for RFM scoring and the R x F segment grid (src/segmentation.py).

Run from the project root (after `python -m src.build_database`):
    python -m pytest
"""

import duckdb
import pandas as pd
import pytest

from src.build_database import DB_PATH
from src.segmentation import (
    SEGMENT_ORDER,
    SEGMENT_RULES,
    check_rfm_table,
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
