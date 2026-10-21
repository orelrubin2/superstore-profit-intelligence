"""Customer segmentation for step 3: RFM scoring and K-Means clustering.

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

K-Means (agreed in the step 3 plan):
    - Behavior features only; profit is left out so that profitability can be
      used afterwards to test the clusters, not to define them.
    - log1p on recency, frequency and gross sales (skewed), discount share left
      as is (already 0-1), then StandardScaler so no unit dominates distances.
    - k is chosen with four checks (elbow, silhouette, minimum cluster size,
      stability) in evaluate_k(); the chosen k is fixed in N_CLUSTERS.
    - A contrast run without the discount feature tests whether discount
      behavior is what sets loss-making customers apart.

WARNING (leakage): scores and clusters summarize each customer's full
2014-2017 history. Do not use them as features in the step 4 order-line model.
"""

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["recency_days", "frequency", "gross_sales", "high_discount_gross_share"]
CONTRAST_FEATURES = ["recency_days", "frequency", "gross_sales"]   # no discount feature
LOG_FEATURES = {"recency_days", "frequency", "gross_sales"}
RANDOM_STATE = 42
MIN_CLUSTER_SHARE = 0.05      # every cluster must hold at least 5% of customers
MIN_STABILITY_ARI = 0.80      # mean Adjusted Rand Index across seeds and bootstraps

# Chosen with evaluate_k() on the four checks, judged on the clustering features
# only (profit and the volume groups were not looked at):
#   main     k=2,3,4 pass size and stability; k=3 has the best silhouette (0.271)
#            and is the inertia elbow. k=2 was rejected because it reproduces the
#            no-discount contrast (ARI 0.925), i.e. it ignores the discount feature.
#   contrast only k=2 passes (bootstrap ARI 0.66-0.69 for k=3..5).
N_CLUSTERS = 3
N_CLUSTERS_CONTRAST = 2

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


def cluster_matrix(features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return the scaled feature matrix: log1p on skewed columns, then z-scores."""
    data = features[columns].astype(float).copy()
    for column in columns:
        if column in LOG_FEATURES:
            data[column] = np.log1p(data[column])
    return StandardScaler().fit_transform(data)


def fit_kmeans(x: np.ndarray, k: int, seed: int = RANDOM_STATE) -> KMeans:
    """Fit K-Means with several initializations and a fixed seed."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)


def stability(x: np.ndarray, k: int, n_seeds: int = 20, n_boot: int = 50) -> tuple[float, float]:
    """Mean Adjusted Rand Index of the reference labels vs. other seeds and bootstraps.

    Seeds: refit on all customers with a different random_state.
    Bootstraps: refit on a resample of customers, then label every customer
    with that model, so all runs are compared on the same customers.
    """
    reference = fit_kmeans(x, k).labels_
    seed_ari = [adjusted_rand_score(reference, fit_kmeans(x, k, seed).labels_)
                for seed in range(1, n_seeds + 1)]
    rng = np.random.default_rng(RANDOM_STATE)
    boot_ari = []
    for _ in range(n_boot):
        sample = x[rng.integers(0, len(x), len(x))]
        model = fit_kmeans(sample, k)
        boot_ari.append(adjusted_rand_score(reference, model.predict(x)))
    return float(np.mean(seed_ari)), float(np.mean(boot_ari))


def evaluate_k(x: np.ndarray, ks=range(2, 9)) -> pd.DataFrame:
    """Score each k on the four checks: inertia, silhouette, minimum size, stability."""
    rows = []
    for k in ks:
        model = fit_kmeans(x, k)
        seed_ari, boot_ari = stability(x, k)
        smallest = np.bincount(model.labels_).min() / len(x)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(x, model.labels_),
            "smallest_cluster_share": smallest,
            "seed_ari": seed_ari,
            "bootstrap_ari": boot_ari,
            "passes_size": smallest >= MIN_CLUSTER_SHARE,
            "passes_stability": min(seed_ari, boot_ari) >= MIN_STABILITY_ARI,
        })
    return pd.DataFrame(rows)


def name_main_clusters(features: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """Name the k=3 clusters from their profiles, independent of K-Means label numbers.

    Rule: the cluster with the highest median high-discount share is
    "Active, heavy-discount"; of the other two, the one with lower median gross
    sales is "Quiet, low volume" and the other "Active, regular".
    """
    medians = features.groupby(labels)[["gross_sales", "high_discount_gross_share"]].median()
    discount = medians["high_discount_gross_share"].idxmax()
    rest = medians.drop(index=discount).sort_values("gross_sales")
    names = {discount: "Active, heavy-discount",
             rest.index[0]: "Quiet, low volume",
             rest.index[1]: "Active, regular"}
    return pd.Series(labels, index=features.index).map(names)


def name_contrast_clusters(features: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """Name the k=2 contrast clusters: lower median gross sales = "Quiet", other = "Active"."""
    medians = features.groupby(labels)["gross_sales"].median().sort_values()
    names = {medians.index[0]: "Quiet", medians.index[1]: "Active"}
    return pd.Series(labels, index=features.index).map(names)


def cluster_customers(features: pd.DataFrame) -> pd.DataFrame:
    """Return customer_id with the named main (k=3) and contrast (k=2) clusters."""
    features = features.sort_values("customer_id").reset_index(drop=True)
    main = fit_kmeans(cluster_matrix(features, CLUSTER_FEATURES), N_CLUSTERS).labels_
    contrast = fit_kmeans(cluster_matrix(features, CONTRAST_FEATURES), N_CLUSTERS_CONTRAST).labels_
    return pd.DataFrame({
        "customer_id": features["customer_id"],
        "cluster": name_main_clusters(features, main),
        "cluster_contrast": name_contrast_clusters(features, contrast),
    })


def create_cluster_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create customer_clusters from customer_features."""
    features = con.execute("SELECT * FROM customer_features").df()
    clusters = cluster_customers(features)
    con.register("_clusters", clusters)
    con.execute("CREATE OR REPLACE TABLE customer_clusters AS SELECT * FROM _clusters")
    con.unregister("_clusters")


def check_cluster_table(con: duckdb.DuckDBPyConnection) -> None:
    """Assert one row per customer, all cluster names present, and minimum sizes."""
    clusters = con.execute("SELECT * FROM customer_clusters").df()
    customers = con.execute("SELECT COUNT(*) FROM customer_features").fetchone()[0]
    assert len(clusters) == clusters["customer_id"].nunique() == customers, (
        "customer_clusters is not one row per customer"
    )
    for column, k in [("cluster", N_CLUSTERS), ("cluster_contrast", N_CLUSTERS_CONTRAST)]:
        sizes = clusters[column].value_counts()
        assert clusters[column].notna().all() and len(sizes) == k, f"{column}: expected {k} named clusters"
        assert sizes.min() >= MIN_CLUSTER_SHARE * customers, f"{column}: a cluster is below the minimum size"
