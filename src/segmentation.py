"""Customer segmentation for step 3: RFM scoring (and later K-Means).

RFM scoring (agreed in the step 3 plan):
    - Each of Recency, Frequency and Monetary gets a 1-5 score from its
      average-rank percentile: score = ceil(5 * percentile rank), where tied
      values share their average rank. Tied customers therefore always get
      the same score, and groups are close to, but not exactly, 20% each
      (frequency has only 14 distinct values).
    - Recency is inverted: 5 = most recent.
    - Segments come from an R x F grid; Monetary is kept for profiling only,
      because high-volume (gross sales) is close to M.
    - Segment names are labels, not findings. The "quiet lately" names are
      neutral on purpose: with a median of 6 orders in 4 years, months
      without an order are normal and do not mean a customer has churned.

WARNING (leakage): scores summarize each customer's full 2014-2017 history.
Do not use them as features in the step 4 order-line model.
"""

import duckdb
import numpy as np
import pandas as pd

# Segment name, R scores, F scores. Together the rules cover all 25 cells once.
SEGMENT_RULES = [
    ("Champions",                (4, 5), (4, 5)),
    ("Promising",                (4, 5), (1, 3)),
    ("Loyal",                    (3, 3), (4, 5)),
    ("Needs attention",          (3, 3), (1, 3)),
    ("Frequent, quiet lately",   (1, 2), (4, 5)),
    ("Infrequent, quiet lately", (1, 2), (1, 3)),
]
SEGMENT_ORDER = [name for name, _, _ in SEGMENT_RULES]


def percentile_score(values: pd.Series, bins: int = 5, higher_is_better: bool = True) -> pd.Series:
    """Score values 1..bins from their average-rank percentile; ties share a score."""
    pct = values.rank(method="average", pct=True)
    score = np.ceil(pct * bins).clip(1, bins).astype(int)
    return score if higher_is_better else bins + 1 - score


def rfm_segment(r_score: int, f_score: int) -> str:
    """Return the R x F segment name for one pair of scores."""
    for name, (r_low, r_high), (f_low, f_high) in SEGMENT_RULES:
        if r_low <= r_score <= r_high and f_low <= f_score <= f_high:
            return name
    raise ValueError(f"no segment rule for R={r_score}, F={f_score}")


def rfm_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Add R, F and M scores and the R x F segment to customer features.

    Expects columns customer_id, recency_days, frequency and sales.
    """
    out = features[["customer_id", "recency_days", "frequency", "sales"]].copy()
    out["r_score"] = percentile_score(out["recency_days"], higher_is_better=False)
    out["f_score"] = percentile_score(out["frequency"])
    out["m_score"] = percentile_score(out["sales"])
    out["rfm_segment"] = [rfm_segment(r, f) for r, f in zip(out["r_score"], out["f_score"])]
    return out


def create_rfm_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create customer_rfm from customer_features."""
    features = con.execute(
        "SELECT customer_id, recency_days, frequency, sales FROM customer_features"
    ).df()
    rfm = rfm_scores(features).sort_values("customer_id")
    con.register("_rfm", rfm)
    con.execute("CREATE OR REPLACE TABLE customer_rfm AS SELECT * FROM _rfm")
    con.unregister("_rfm")


def check_rfm_table(con: duckdb.DuckDBPyConnection) -> None:
    """Assert every customer has valid scores, ties share scores, and segments match the rules."""
    rfm = con.execute("SELECT * FROM customer_rfm").df()
    customers = con.execute("SELECT COUNT(*) FROM customer_features").fetchone()[0]
    assert len(rfm) == rfm["customer_id"].nunique() == customers, "customer_rfm is not one row per customer"

    for value, score in [("recency_days", "r_score"), ("frequency", "f_score"), ("sales", "m_score")]:
        assert rfm[score].between(1, 5).all(), f"{score} outside 1..5"
        split = rfm.groupby(value)[score].nunique()
        assert (split == 1).all(), f"tied {value} values received different {score}"

    expected = [rfm_segment(r, f) for r, f in zip(rfm["r_score"], rfm["f_score"])]
    assert (rfm["rfm_segment"] == expected).all(), "rfm_segment does not match the R x F rules"
