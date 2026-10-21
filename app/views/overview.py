"""Page 1: key numbers and margin by discount band, with category / region / state filters."""

import altair as alt
import pandas as pd
import streamlit as st

from app import data
from app.style import BLUE, LOSS, money, pct

HIGH_DISCOUNT = 0.20


def filter_lines(lines: pd.DataFrame, categories, regions, states) -> pd.DataFrame:
    """Keep lines matching every non-empty filter (an empty filter means 'all')."""
    mask = pd.Series(True, index=lines.index)
    for column, chosen in [("category", categories), ("region", regions), ("state", states)]:
        if chosen:
            mask &= lines[column].isin(chosen)
    return lines[mask]


def band_summary(lines: pd.DataFrame) -> pd.DataFrame:
    """Lines, sales, profit, margin and loss-line share per discount band."""
    out = (lines.groupby("discount_band")
           .agg(order_lines=("row_id", "size"), sales=("sales", "sum"),
                profit=("profit", "sum"), loss_line_share=("is_loss", "mean"))
           .reindex(data.DISCOUNT_BANDS).dropna(subset=["order_lines"]))
    out["margin"] = out["profit"] / out["sales"]
    return out.reset_index()


def render() -> None:
    st.title("Where does the company lose money?")
    st.caption("Sample Superstore, 9,993 order lines, 2014-2017. Empty filters mean 'all'.")
    lines = data.load_lines()

    c1, c2, c3 = st.columns(3)
    categories = c1.multiselect("Category", sorted(lines["category"].unique()),
                                placeholder="All categories")
    regions = c2.multiselect("Region", sorted(lines["region"].unique()), placeholder="All regions")
    state_pool = lines[lines["region"].isin(regions)] if regions else lines
    states = c3.multiselect("State", sorted(state_pool["state"].unique()), placeholder="All states")

    selected = filter_lines(lines, categories, regions, states)
    if selected.empty:
        st.warning("No order lines match these filters.")
        return

    high = selected[selected["discount"] > HIGH_DISCOUNT]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sales", money(selected["sales"].sum()))
    k2.metric("Profit", money(selected["profit"].sum()))
    k3.metric("Profit margin", pct(selected["profit"].sum() / selected["sales"].sum()))
    k4.metric("Profit on lines discounted >20%", money(high["profit"].sum()),
              help="Sum of profit on order lines with a discount above 20% (query 03).")

    bands = band_summary(selected)
    chart = (alt.Chart(bands)
             .mark_bar(cornerRadiusEnd=4, size=48)
             .encode(x=alt.X("discount_band:N", sort=data.DISCOUNT_BANDS, title="Discount band"),
                     y=alt.Y("margin:Q", title="Profit margin", axis=alt.Axis(format="%")),
                     color=alt.condition("datum.margin >= 0", alt.value(BLUE), alt.value(LOSS)),
                     tooltip=[alt.Tooltip("discount_band:N", title="Band"),
                              alt.Tooltip("order_lines:Q", title="Order lines", format=","),
                              alt.Tooltip("sales:Q", title="Sales", format="$,.0f"),
                              alt.Tooltip("profit:Q", title="Profit", format="$,.0f"),
                              alt.Tooltip("margin:Q", title="Margin", format=".1%"),
                              alt.Tooltip("loss_line_share:Q", title="Loss-making lines", format=".0%")]))
    labels = chart.mark_text(dy=-8, fontSize=12).encode(
        text=alt.Text("margin:Q", format=".1%"), color=alt.value("#0b0b0b"))
    st.subheader("Profit margin by discount band")
    st.altair_chart((chart + labels).properties(height=320), width="stretch")

    if high.empty:
        st.info("No order lines in this selection are discounted above 20%.")
    else:
        st.markdown(
            f"In this selection, **{len(high):,} order lines** are discounted above 20%. "
            f"They bring {money(high['sales'].sum())} in sales and "
            f"**{money(high['profit'].sum())} in profit**, "
            f"and {pct(high['is_loss'].mean(), 0)} of them lose money."
        )

    with st.expander("Numbers behind the chart"):
        st.dataframe(bands.style.format({"order_lines": "{:,.0f}", "sales": "${:,.0f}",
                                         "profit": "${:,.0f}", "margin": "{:.1%}",
                                         "loss_line_share": "{:.1%}"}),
                     hide_index=True, width="stretch")
