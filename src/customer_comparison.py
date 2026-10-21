"""Step 3 comparison: loss-making vs. profitable high-volume customers.

Business question: "Who are our high-volume customers that generate net losses,
and how do they differ from profitable high-volume customers in recency,
frequency, discounts, products and regions?"

Every function takes an open DuckDB connection and returns a DataFrame, so the
notebook can display and chart the results directly. Group-level shares
(product, region, discount mix) are ratios of totals over the group's gross
sales; per-customer metrics use medians because of skew. Differences come with
bootstrap 95% confidence intervals (customers resampled within each group).
"""

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import adjusted_rand_score

GROUPS = ["HV-loss", "HV-profit"]
N_BOOT = 2000
SEED = 42


def load_customers(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """customer_features joined with RFM segment and clusters."""
    return con.execute("""
        SELECT f.*, r.r_score, r.f_score, r.m_score, r.rfm_segment,
               c.cluster, c.cluster_contrast
        FROM customer_features f
        JOIN customer_rfm r USING (customer_id)
        JOIN customer_clusters c USING (customer_id)
        ORDER BY customer_id
    """).df()


def group_overview(customers: pd.DataFrame) -> pd.DataFrame:
    """Size, gross sales, sales and profit of each volume group."""
    g = customers.groupby("volume_group")
    out = pd.DataFrame({
        "customers": g.size(),
        "gross_sales": g["gross_sales"].sum(),
        "sales": g["sales"].sum(),
        "profit": g["profit"].sum(),
    })
    out["profit_margin"] = out["profit"] / out["sales"]
    out["share_of_total_profit"] = out["profit"] / out["profit"].sum()
    return out.loc[["HV-loss", "HV-profit", "Other"]]


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, stat=np.median) -> tuple[float, float]:
    """95% bootstrap interval for stat(a) - stat(b), resampling each group."""
    rng = np.random.default_rng(SEED)
    diffs = [stat(rng.choice(a, len(a))) - stat(rng.choice(b, len(b))) for _ in range(N_BOOT)]
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def compare_customer_metrics(customers: pd.DataFrame) -> pd.DataFrame:
    """Medians of per-customer metrics for HV-loss vs. HV-profit, with CI and p-value."""
    metrics = {
        "recency_days": "Recency (days since last order)",
        "frequency": "Frequency (orders)",
        "tenure_days": "Tenure (days since first order)",
        "gross_sales": "Gross sales ($)",
        "avg_discount": "Average discount",
        "high_discount_gross_share": "Share of gross sales at >20% discount",
        "worst_line_profit": "Worst single line profit ($)",
    }
    rows = []
    for column, label in metrics.items():
        a = customers.loc[customers.volume_group == "HV-loss", column].to_numpy(float)
        b = customers.loc[customers.volume_group == "HV-profit", column].to_numpy(float)
        low, high = _bootstrap_diff(a, b)
        rows.append({
            "metric": label,
            "HV-loss median": np.median(a),
            "HV-profit median": np.median(b),
            "difference": np.median(a) - np.median(b),
            "ci_low": low,
            "ci_high": high,
            "mann_whitney_p": mannwhitneyu(a, b).pvalue,
        })
    return pd.DataFrame(rows)


def group_mix(con: duckdb.DuckDBPyConnection, dimension: str) -> pd.DataFrame:
    """Each group's gross-sales share and profit for every value of a mix dimension.

    Shares are group totals (sum of the value's gross sales / group gross sales),
    so large customers weigh more, as they do in the business.
    """
    return con.execute("""
        SELECT f.volume_group, m.value,
               SUM(m.gross_sales) AS gross_sales,
               SUM(m.sales)       AS sales,
               SUM(m.profit)      AS profit,
               SUM(m.gross_sales) / SUM(SUM(m.gross_sales)) OVER (PARTITION BY f.volume_group)
                                  AS gross_share
        FROM customer_mix m
        JOIN customer_features f USING (customer_id)
        WHERE m.dimension = ? AND f.volume_group IN ('HV-loss', 'HV-profit')
        GROUP BY f.volume_group, m.value
    """, [dimension]).df()


def mix_comparison(con: duckdb.DuckDBPyConnection, dimension: str) -> pd.DataFrame:
    """Gross-sales share per value, HV-loss vs. HV-profit, plus HV-loss profit by value."""
    mix = group_mix(con, dimension)
    share = mix.pivot(index="value", columns="volume_group", values="gross_share").fillna(0)
    profit = mix.pivot(index="value", columns="volume_group", values="profit").fillna(0)
    out = pd.DataFrame({
        "HV-loss share": share["HV-loss"],
        "HV-profit share": share["HV-profit"],
        "difference (pp)": (share["HV-loss"] - share["HV-profit"]) * 100,
        "HV-loss profit": profit["HV-loss"],
        "HV-profit profit": profit["HV-profit"],
    })
    return out.sort_values("difference (pp)", ascending=False)


def share_ci(con: duckdb.DuckDBPyConnection, dimension: str, values: list[str]) -> dict:
    """Group-level gross share of the given values, with a bootstrap CI of the difference."""
    data = con.execute(f"""
        SELECT f.customer_id, f.volume_group, f.gross_sales AS total,
               COALESCE(SUM(m.gross_sales) FILTER (WHERE m.value IN ({', '.join('?' * len(values))})), 0) AS part
        FROM customer_features f
        LEFT JOIN customer_mix m ON m.customer_id = f.customer_id AND m.dimension = ?
        WHERE f.volume_group IN ('HV-loss', 'HV-profit')
        GROUP BY ALL
        ORDER BY f.customer_id   -- fixed order, so the seeded bootstrap is reproducible
    """, values + [dimension]).df()
    rng = np.random.default_rng(SEED)
    parts = {g: data[data.volume_group == g][["part", "total"]].to_numpy() for g in GROUPS}
    share = {g: parts[g][:, 0].sum() / parts[g][:, 1].sum() for g in GROUPS}
    diffs = []
    for _ in range(N_BOOT):
        s = {}
        for g in GROUPS:
            sample = parts[g][rng.integers(0, len(parts[g]), len(parts[g]))]
            s[g] = sample[:, 0].sum() / sample[:, 1].sum()
        diffs.append(s["HV-loss"] - s["HV-profit"])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"values": " + ".join(values), "HV-loss share": share["HV-loss"],
            "HV-profit share": share["HV-profit"],
            "difference (pp)": (share["HV-loss"] - share["HV-profit"]) * 100,
            "ci_low (pp)": low * 100, "ci_high (pp)": high * 100}


def margin_by_discount_band(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Mix vs. rate: each group's profit margin and gross share within each discount band."""
    mix = group_mix(con, "discount_band")
    mix["margin"] = mix["profit"] / mix["sales"]
    out = mix.pivot(index="value", columns="volume_group", values=["gross_share", "margin"])
    out.columns = [f"{group} {metric}" for metric, group in out.columns]
    return out.reindex(["0%", "1-20%", "21-40%", ">40%"])


def loss_concentration(customers: pd.DataFrame) -> pd.DataFrame:
    """How far one bad line explains each HV-loss customer's losses."""
    hv_loss = customers[customers.volume_group == "HV-loss"]
    without_worst = hv_loss["profit"] - hv_loss["worst_line_profit"]
    return pd.DataFrame({
        "HV-loss customers": [len(hv_loss)],
        "median worst-line share of losses": [hv_loss["worst_line_share_of_losses"].median()],
        "worst line > 50% of losses": [(hv_loss["worst_line_share_of_losses"] > 0.5).sum()],
        "profitable without their worst line": [(without_worst > 0).sum()],
        "median worst line profit ($)": [hv_loss["worst_line_profit"].median()],
    })


def segment_crosstab(customers: pd.DataFrame, column: str) -> pd.DataFrame:
    """Customers per segment for each volume group, with HV-loss lift.

    Lift = segment's share of HV-loss customers / segment's share of all
    customers (2 = twice as concentrated as chance).
    """
    counts = pd.crosstab(customers[column], customers["volume_group"])
    out = counts[["HV-loss", "HV-profit", "Other"]].copy()
    out["all customers"] = counts.sum(axis=1)
    out["share of HV-loss"] = out["HV-loss"] / out["HV-loss"].sum()
    out["share of all"] = out["all customers"] / out["all customers"].sum()
    out["lift"] = out["share of HV-loss"] / out["share of all"]
    hv = out["HV-loss"] + out["HV-profit"]
    out["loss rate among HV"] = (out["HV-loss"] / hv).where(hv > 0)
    return out.sort_values("lift", ascending=False)


def segmentation_agreement(customers: pd.DataFrame) -> pd.DataFrame:
    """Adjusted Rand Index between RFM segments, main clusters and contrast clusters."""
    pairs = [("rfm_segment", "cluster"), ("rfm_segment", "cluster_contrast"),
             ("cluster", "cluster_contrast")]
    return pd.DataFrame([{"a": a, "b": b, "ari": adjusted_rand_score(customers[a], customers[b])}
                         for a, b in pairs])


def sensitivity(con: duckdb.DuckDBPyConnection, customers: pd.DataFrame) -> pd.DataFrame:
    """Repeat the key comparisons under the alternative high-volume definitions."""
    tx_il = con.execute("""
        SELECT customer_id, SUM(gross_sales) AS tx_il_gross
        FROM customer_mix WHERE dimension = 'state' AND value IN ('Texas', 'Illinois')
        GROUP BY customer_id
    """).df()
    data = customers.merge(tx_il, on="customer_id", how="left").fillna({"tx_il_gross": 0})
    data["hd_gross"] = data["high_discount_gross_share"] * data["gross_sales"]

    definitions = {
        "gross sales top 25% (primary)": data["is_high_volume"],
        "gross sales top 20%": data["hv_gross_p80"],
        "net sales top 25%": data["hv_net_sales_p75"],
        "quantity top 25%": data["hv_quantity_p75"],
        "primary, excluding |profit| < $100": data["is_high_volume"] & ~data["is_near_zero_profit"],
    }
    rows = []
    for name, is_hv in definitions.items():
        row = {"definition": name}
        for label, mask in [("loss", is_hv & data.is_net_loss), ("profit", is_hv & ~data.is_net_loss)]:
            group = data[mask]
            row[f"HV-{label} n"] = len(group)
            row[f"HV-{label} >20% disc share"] = group["hd_gross"].sum() / group["gross_sales"].sum()
            row[f"HV-{label} TX+IL share"] = group["tx_il_gross"].sum() / group["gross_sales"].sum()
            row[f"HV-{label} median recency"] = group["recency_days"].median()
            row[f"HV-{label} median orders"] = group["frequency"].median()
        loss = data[is_hv & data.is_net_loss]
        row["HV-loss in heavy-discount cluster"] = (loss["cluster"] == "Active, heavy-discount").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def subcategory_buyers(con: duckdb.DuckDBPyConnection, sub_categories: list[str]) -> pd.DataFrame:
    """For HV-loss customers: how many buy each sub-category, how many lose on it,
    and how much of its gross sales comes from the single largest customer."""
    placeholders = ", ".join("?" * len(sub_categories))
    return con.execute(f"""
        SELECT m.value                                   AS sub_category,
               COUNT(*)                                  AS customers_buying,
               COUNT(*) FILTER (WHERE m.profit < 0)      AS customers_losing,
               SUM(m.profit)                             AS profit,
               SUM(m.gross_sales)                        AS gross_sales,
               MAX(m.gross_sales) / SUM(m.gross_sales)   AS largest_customer_share
        FROM customer_mix m
        JOIN customer_features f USING (customer_id)
        WHERE f.volume_group = 'HV-loss' AND m.dimension = 'sub_category'
          AND m.value IN ({placeholders})
        GROUP BY m.value
        ORDER BY profit
    """, sub_categories).df()


def worst_lines(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Each HV-loss customer's single worst order line (one row per customer)."""
    return con.execute("""
        SELECT customer_id, order_id, sub_category, discount, discount_band, sales, profit
        FROM (
            SELECT o.*, ROW_NUMBER() OVER (PARTITION BY o.customer_id
                                           ORDER BY o.profit, o.row_id) AS rn
            FROM orders_clean o
            JOIN customer_features f USING (customer_id)
            WHERE f.volume_group = 'HV-loss'
        )
        WHERE rn = 1
        ORDER BY profit
    """).df()


def worst_line_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """HV-loss customers' worst lines grouped by sub-category and discount band."""
    lines = worst_lines(con)
    return (lines.groupby(["sub_category", "discount_band"])
            .agg(customers=("customer_id", "size"),
                 median_profit=("profit", "median"),
                 median_discount=("discount", "median"))
            .sort_values("customers", ascending=False)
            .reset_index())


def state_discount_levels(con: duckdb.DuckDBPyConnection, states: list[str]) -> pd.DataFrame:
    """Discount level and margin of the given states, over all customers."""
    placeholders = ", ".join("?" * len(states))
    return con.execute(f"""
        SELECT state, region,
               COUNT(*)                                             AS order_lines,
               AVG(discount)                                        AS avg_discount,
               COALESCE(SUM(gross_sales) FILTER (WHERE discount > 0.2), 0)
                   / SUM(gross_sales)                               AS gross_share_above_20,
               SUM(profit) / SUM(sales)                             AS profit_margin
        FROM orders_clean
        WHERE state IN ({placeholders})
        GROUP BY state, region
        ORDER BY gross_share_above_20 DESC
    """, states).df()


def mix_vs_rate(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Split the profit gap per $ of gross sales into discount mix vs. within-band rate.

    profit per gross $ = sum over discount bands of (gross share x profit per gross $).
    Swapping only the mix (or only the rates) between the groups shows how much
    of the gap the discount mix explains. Both orderings are reported because
    the split depends on which group's rates are held fixed.
    """
    bands = group_mix(con, "discount_band")
    bands["rate"] = bands["profit"] / bands["gross_sales"]
    share = bands.pivot(index="value", columns="volume_group", values="gross_share").fillna(0)
    rate = bands.pivot(index="value", columns="volume_group", values="rate").fillna(0)

    def per_gross(mix_of: str, rates_of: str) -> float:
        return float((share[mix_of] * rate[rates_of]).sum())

    loss, profit = per_gross("HV-loss", "HV-loss"), per_gross("HV-profit", "HV-profit")
    gap = profit - loss
    swap_mix = per_gross("HV-profit", "HV-loss")      # HV-loss rates, HV-profit mix
    swap_rates = per_gross("HV-loss", "HV-profit")    # HV-profit rates, HV-loss mix
    return pd.DataFrame([
        {"scenario": "HV-loss actual", "profit_per_gross_dollar": loss},
        {"scenario": "HV-profit actual", "profit_per_gross_dollar": profit},
        {"scenario": "HV-loss rates, HV-profit discount mix", "profit_per_gross_dollar": swap_mix,
         "share_of_gap_from_mix": (swap_mix - loss) / gap},
        {"scenario": "HV-profit rates, HV-loss discount mix", "profit_per_gross_dollar": swap_rates,
         "share_of_gap_from_mix": (profit - swap_rates) / gap},
    ])
