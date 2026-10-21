"""Page 2: loss-making vs. profitable high-volume customers, and where they fall in the clusters."""

import altair as alt
import pandas as pd
import streamlit as st

import src.customer_comparison as cc
from app import data
from app.style import BAND_COLORS, BLUE, GREY, LOSS, money, pct


def render() -> None:
    st.title("Loss-making high-volume customers")
    st.caption("High-volume = top 25% of customers by gross sales (before discount). "
               "HV-loss customers lost money in total over 2014-2017; HV-profit customers did not.")
    customers = data.load_customers()
    overview = cc.group_overview(customers)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HV-loss customers", f"{overview.loc['HV-loss', 'customers']:,}")
    c2.metric("Their total profit", money(overview.loc["HV-loss", "profit"]),
              help="Share of company profit: " + pct(overview.loc["HV-loss", "share_of_total_profit"]))
    c3.metric("HV-profit customers", f"{overview.loc['HV-profit', 'customers']:,}")
    c4.metric("Their total profit", money(overview.loc["HV-profit", "profit"]),
              help="Share of company profit: " + pct(overview.loc["HV-profit", "share_of_total_profit"]))

    left, right = st.columns(2)

    mix = data.discount_mix_by_group()
    mix_chart = (alt.Chart(mix)
                 .mark_bar(size=42, stroke="#fcfcfb", strokeWidth=2)
                 .encode(y=alt.Y("volume_group:N", title=None, sort=["HV-loss", "HV-profit"]),
                         x=alt.X("gross_share:Q", stack="normalize", title="Share of gross sales",
                                 axis=alt.Axis(format="%")),
                         color=alt.Color("value:N", title="Discount",
                                         scale=alt.Scale(domain=data.DISCOUNT_BANDS, range=BAND_COLORS),
                                         legend=alt.Legend(orient="bottom")),
                         order=alt.Order("value:N", sort="ascending"),
                         tooltip=[alt.Tooltip("volume_group:N", title="Group"),
                                  alt.Tooltip("value:N", title="Discount band"),
                                  alt.Tooltip("gross_share:Q", title="Share", format=".1%")]))
    left.subheader("What they buy at a deep discount")
    left.altair_chart(mix_chart.properties(height=220), width="stretch")

    table = cc.segment_crosstab(customers, "cluster").reset_index()
    base_rate = (customers["volume_group"] == "HV-loss").sum() / customers["is_high_volume"].sum()
    table["label"] = [f"{r:.0%} ({int(l)}/{int(l + p)})" for r, l, p in
                      zip(table["loss rate among HV"], table["HV-loss"], table["HV-profit"])]
    bars = (alt.Chart(table)
            .mark_bar(size=48, cornerRadiusEnd=4)
            .encode(x=alt.X("cluster:N", title=None, sort="-y", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("loss rate among HV:Q", title="Loss-making share of high-volume customers",
                            axis=alt.Axis(format="%")),
                    color=alt.condition(alt.datum.cluster == "Active, heavy-discount",
                                        alt.value(LOSS), alt.value(BLUE)),
                    tooltip=[alt.Tooltip("cluster:N", title="Cluster"),
                             alt.Tooltip("HV-loss:Q", title="HV-loss customers"),
                             alt.Tooltip("HV-profit:Q", title="HV-profit customers"),
                             alt.Tooltip("all customers:Q", title="All customers"),
                             alt.Tooltip("lift:Q", title="HV-loss lift", format=".1f")]))
    text = bars.mark_text(dy=-8, fontSize=12).encode(text="label:N", color=alt.value("#0b0b0b"))
    rule = (alt.Chart(pd.DataFrame({"y": [base_rate]}))
            .mark_rule(strokeDash=[5, 4], color=GREY).encode(y="y:Q"))
    right.subheader("Where they fall in the K-Means clusters")
    right.altair_chart((bars + text + rule).properties(height=260), width="stretch")
    right.caption(f"Dashed line: all high-volume customers ({base_rate:.1%} lose money).")

    st.markdown(
        "**Step 3 finding:** these customers are not smaller, newer or lapsing. What sets them "
        "apart is that 73% of what they buy is discounted above 20% (vs. 15%), in high-discount "
        "states and in Machines and Binders. RFM would rank half of them as top customers; the "
        "heavy-discount cluster holds 91% of them. Details: `notebooks/03_customer_segmentation.ipynb`."
    )

    st.subheader("HV-loss customers")
    clusters = ["All clusters"] + sorted(customers["cluster"].unique())
    chosen = st.selectbox("Cluster", clusters)
    hv_loss = customers[customers["volume_group"] == "HV-loss"]
    if chosen != "All clusters":
        hv_loss = hv_loss[hv_loss["cluster"] == chosen]
    columns = {"customer_name": "Customer", "segment": "Segment", "cluster": "Cluster",
               "rfm_segment": "RFM segment", "gross_sales": "Gross sales", "profit": "Profit",
               "high_discount_gross_share": "Share >20% discount", "frequency": "Orders",
               "recency_days": "Days since last order"}
    st.dataframe(hv_loss.sort_values("profit")[list(columns)].rename(columns=columns)
                 .style.format({"Gross sales": "${:,.0f}", "Profit": "${:,.0f}",
                                "Share >20% discount": "{:.0%}"}),
                 hide_index=True, width="stretch")
    st.caption(f"{len(hv_loss)} customers shown.")
