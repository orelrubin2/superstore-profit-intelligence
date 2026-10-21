"""Page 2: exploratory KPI dashboard (map, cities, monthly trend, categories, segments)."""

import altair as alt
import pandas as pd
import streamlit as st

from app import data
from app.style import (BLUE, BORDER, DIVERGING, LOSS, REGION_COLORS, SEQUENTIAL, SURFACE,
                       money, pct)

# US state outlines, loaded by the viewer's browser (pinned version of vega-datasets)
US_TOPOJSON = "https://cdn.jsdelivr.net/npm/vega-datasets@2.8.0/data/us-10m.json"

# State name -> (postal code, FIPS code); FIPS is the 'id' of each state in US_TOPOJSON
STATES = {
    "Alabama": ("AL", 1), "Arizona": ("AZ", 4), "Arkansas": ("AR", 5), "California": ("CA", 6),
    "Colorado": ("CO", 8), "Connecticut": ("CT", 9), "Delaware": ("DE", 10),
    "District of Columbia": ("DC", 11), "Florida": ("FL", 12), "Georgia": ("GA", 13),
    "Idaho": ("ID", 16), "Illinois": ("IL", 17), "Indiana": ("IN", 18), "Iowa": ("IA", 19),
    "Kansas": ("KS", 20), "Kentucky": ("KY", 21), "Louisiana": ("LA", 22), "Maine": ("ME", 23),
    "Maryland": ("MD", 24), "Massachusetts": ("MA", 25), "Michigan": ("MI", 26),
    "Minnesota": ("MN", 27), "Mississippi": ("MS", 28), "Missouri": ("MO", 29),
    "Montana": ("MT", 30), "Nebraska": ("NE", 31), "Nevada": ("NV", 32),
    "New Hampshire": ("NH", 33), "New Jersey": ("NJ", 34), "New Mexico": ("NM", 35),
    "New York": ("NY", 36), "North Carolina": ("NC", 37), "North Dakota": ("ND", 38),
    "Ohio": ("OH", 39), "Oklahoma": ("OK", 40), "Oregon": ("OR", 41), "Pennsylvania": ("PA", 42),
    "Rhode Island": ("RI", 44), "South Carolina": ("SC", 45), "South Dakota": ("SD", 46),
    "Tennessee": ("TN", 47), "Texas": ("TX", 48), "Utah": ("UT", 49), "Vermont": ("VT", 50),
    "Virginia": ("VA", 51), "Washington": ("WA", 53), "West Virginia": ("WV", 54),
    "Wisconsin": ("WI", 55), "Wyoming": ("WY", 56),
}

MONEY_TOOLTIPS = [alt.Tooltip("sales:Q", title="Sales", format="$,.0f"),
                  alt.Tooltip("profit:Q", title="Profit", format="$,.0f"),
                  alt.Tooltip("margin:Q", title="Margin", format=".1%")]
TABLE_FORMAT = {"sales": "${:,.0f}", "profit": "${:,.0f}", "margin": "{:.1%}"}


# ---------------------------------------------------------------- aggregations

def summarize(lines: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sales, profit and margin (SUM(profit) / SUM(sales)) per group."""
    out = lines.groupby(by, as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
    out["margin"] = out["profit"] / out["sales"]
    return out


def kpis(lines: pd.DataFrame) -> dict:
    """Headline numbers of a selection."""
    sales, profit = lines["sales"].sum(), lines["profit"].sum()
    return {"sales": sales, "profit": profit, "margin": profit / sales,
            "orders": lines["order_id"].nunique(), "customers": lines["customer_id"].nunique()}


def by_state(lines: pd.DataFrame) -> pd.DataFrame:
    """Sales, profit and margin per state, with postal and FIPS codes for the map."""
    out = summarize(lines, ["state"])
    out["code"] = out["state"].map(lambda s: STATES[s][0])
    out["fips"] = out["state"].map(lambda s: STATES[s][1])
    return out


def top_cities(lines: pd.DataFrame, rank_by: str = "sales", n: int = 10) -> pd.DataFrame:
    """Top n cities by sales or profit. Cities are keyed by city + state (names repeat)."""
    out = summarize(lines, ["city", "state"])
    out["label"] = out["city"] + ", " + out["state"].map(lambda s: STATES[s][0])
    return out.sort_values(rank_by, ascending=False).head(n).reset_index(drop=True)


def monthly_by_region(lines: pd.DataFrame) -> pd.DataFrame:
    """Sales and profit per region and calendar month."""
    out = summarize(lines, ["order_ym", "region"])
    out["month"] = pd.to_datetime(out["order_ym"] + "-01")
    return out.drop(columns="order_ym")


# ---------------------------------------------------------------- charts

def state_map(states: pd.DataFrame, metric: str) -> alt.LayerChart:
    """Choropleth of one metric; sequential for sales, diverging around 0 otherwise."""
    fmt = ".0%" if metric == "margin" else "$,.0f"
    if metric == "sales":
        scale = alt.Scale(range=SEQUENTIAL, interpolate="lab")
    else:
        scale = alt.Scale(domainMid=0, range=DIVERGING, interpolate="lab")
    topo = alt.topo_feature(US_TOPOJSON, "states")
    background = alt.Chart(topo).mark_geoshape(fill=SURFACE, stroke=BORDER, strokeWidth=0.75)
    shapes = (alt.Chart(topo)
              .mark_geoshape(stroke=SURFACE, strokeWidth=0.75)
              .transform_lookup(lookup="id", from_=alt.LookupData(
                  states, "fips", ["state", "sales", "profit", "margin"]))
              .transform_filter("isValid(datum.state)")
              .encode(color=alt.Color(f"{metric}:Q", scale=scale, title=metric.capitalize(),
                                      legend=alt.Legend(format=fmt, orient="right")),
                      tooltip=[alt.Tooltip("state:N", title="State"), *MONEY_TOOLTIPS]))
    return (background + shapes).project(type="albersUsa").properties(height=430)


def city_bars(cities: pd.DataFrame, rank_by: str) -> alt.LayerChart:
    """Horizontal bars of the ranked metric, red where it is negative."""
    bars = (alt.Chart(cities)
            .mark_bar(cornerRadiusEnd=4, size=20)
            .encode(y=alt.Y("label:N", sort="-x", title=None),
                    x=alt.X(f"{rank_by}:Q", title=rank_by.capitalize(),
                            axis=alt.Axis(format="$,.0f")),
                    color=alt.condition(f"datum.{rank_by} >= 0", alt.value(BLUE), alt.value(LOSS)),
                    tooltip=[alt.Tooltip("label:N", title="City"), *MONEY_TOOLTIPS]))
    text = bars.mark_text(align="left", dx=4, fontSize=12).encode(
        text=alt.Text(f"{rank_by}:Q", format="$,.0f"), color=alt.value("#0b0b0b"))
    return (bars + text).properties(height=360)


def end_labels(monthly: pd.DataFrame, measure: str, gap: float = 0.06) -> pd.DataFrame:
    """Last-month value per region, nudged up so labels sit at least `gap` of the y-range apart."""
    last = monthly[monthly["month"] == monthly["month"].max()].sort_values(measure)
    min_gap = gap * (monthly[measure].max() - min(monthly[measure].min(), 0))
    positions = []
    for value in last[measure]:
        positions.append(value if not positions else max(value, positions[-1] + min_gap))
    return last.assign(label_y=positions)


def trend_lines(monthly: pd.DataFrame, measure: str) -> alt.LayerChart:
    """One line per region with a hover crosshair and direct labels at the last month."""
    regions = sorted(monthly["region"].unique())
    color = alt.Color("region:N", title="Region",
                      scale=alt.Scale(domain=regions, range=[REGION_COLORS[r] for r in regions]),
                      legend=alt.Legend(orient="top"))
    base = alt.Chart(monthly).encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=0, tickCount=8)))
    lines = base.mark_line(strokeWidth=2).encode(
        y=alt.Y(f"{measure}:Q", title=measure.capitalize(), axis=alt.Axis(format="$,.0f")),
        color=color)
    hover = alt.selection_point(fields=["month"], nearest=True, on="pointerover",
                                clear="pointerout", empty=False)
    rule = (base.transform_pivot("region", value=measure, groupby=["month"])
            .mark_rule(color=BORDER, strokeWidth=1)
            .encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                    tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"),
                             *[alt.Tooltip(f"{r}:Q", format="$,.0f") for r in regions]])
            .add_params(hover))
    points = lines.mark_point(filled=True, size=50).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    labels = (alt.Chart(end_labels(monthly, measure))
              .mark_text(align="left", dx=6, fontSize=12)
              .encode(x="month:T", y="label_y:Q", text="region:N", color=color))
    return (lines + points + rule + labels).properties(height=340)


def sub_category_bars(subs: pd.DataFrame, measure: str) -> alt.FacetChart:
    """Bars of one measure per sub-category, one row per category, sorted by sales."""
    bars = (alt.Chart(subs)
            .mark_bar(cornerRadiusEnd=4, size=16)
            .encode(y=alt.Y("sub_category:N", title=None,
                            sort=alt.EncodingSortField("sales", order="descending")),
                    x=alt.X(f"{measure}:Q", title=measure.capitalize(),
                            axis=alt.Axis(format="$,.0f")),
                    color=alt.condition(f"datum.{measure} >= 0", alt.value(BLUE), alt.value(LOSS)),
                    tooltip=[alt.Tooltip("category:N", title="Category"),
                             alt.Tooltip("sub_category:N", title="Sub-category"), *MONEY_TOOLTIPS])
            .properties(width=280))
    return (bars.facet(row=alt.Row("category:N", title=None,
                                   header=alt.Header(labelAngle=0, labelAlign="left",
                                                     labelFontWeight="bold")))
            .resolve_scale(y="independent"))


def segment_bars(segments: pd.DataFrame) -> alt.LayerChart:
    """Sales per customer segment (one series, so one color)."""
    bars = (alt.Chart(segments)
            .mark_bar(cornerRadiusEnd=4, size=28, color=BLUE)
            .encode(y=alt.Y("segment:N", sort="-x", title=None),
                    x=alt.X("sales:Q", title="Sales", axis=alt.Axis(format="$,.0f")),
                    tooltip=[alt.Tooltip("segment:N", title="Segment"), *MONEY_TOOLTIPS]))
    text = bars.mark_text(align="left", dx=4, fontSize=12).encode(
        text=alt.Text("sales:Q", format="$,.0f"), color=alt.value("#0b0b0b"))
    return (bars + text).properties(height=180)


def numbers_table(df: pd.DataFrame, columns: list[str]) -> None:
    with st.expander("Numbers behind the chart"):
        st.dataframe(df[columns].style.format(TABLE_FORMAT), hide_index=True, width="stretch")


# ---------------------------------------------------------------- page

def render() -> None:
    st.title("Explore the data")
    st.caption("Sales and profit by state, city, month, product and customer segment. "
               "The filters apply to the whole page; empty filters mean 'all'.")
    lines = data.load_lines()

    f1, f2, f3, f4 = st.columns(4)
    years = f1.multiselect("Year", sorted(lines["order_year"].unique()), placeholder="All years")
    regions = f2.multiselect("Region", sorted(lines["region"].unique()), placeholder="All regions")
    categories = f3.multiselect("Category", sorted(lines["category"].unique()),
                                placeholder="All categories")
    segments = f4.multiselect("Segment", sorted(lines["segment"].unique()),
                              placeholder="All segments")
    selected = data.filter_lines(lines, order_year=years, region=regions, category=categories,
                                 segment=segments)
    if selected.empty:
        st.warning("No order lines match these filters.")
        return

    k = kpis(selected)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sales", money(k["sales"]))
    k2.metric("Profit", money(k["profit"]))
    k3.metric("Profit margin", pct(k["margin"]))
    k4.metric("Orders", f"{k['orders']:,}")
    k5.metric("Customers", f"{k['customers']:,}")

    # ---- map
    st.subheader("Sales and profit by state")
    metric = st.radio("Map metric", ["Sales", "Profit", "Margin"], horizontal=True,
                      key="map_metric").lower()
    states = by_state(selected)
    st.altair_chart(state_map(states, metric), width="stretch")
    losing = states[states["profit"] < 0].sort_values("profit")
    if losing.empty:
        st.markdown(f"All **{len(states)} states** in this selection are profitable.")
    else:
        worst = losing.iloc[0]
        st.markdown(
            f"**{len(losing)} of {len(states)} states** lose money in this selection, "
            f"together **{money(losing['profit'].sum())}**. The largest loss is in "
            f"**{worst['state']}** ({money(worst['profit'])}, margin {pct(worst['margin'])}).")
    st.caption("Grey-outlined white states have no orders in this selection. "
               "Profit and margin: red = loss, blue = profit, grey = break-even.")
    numbers_table(states.sort_values("profit"), ["state", "sales", "profit", "margin"])

    # ---- cities
    st.subheader("Top 10 cities")
    rank_by = st.radio("Rank by", ["Sales", "Profit"], horizontal=True, key="city_rank").lower()
    cities = top_cities(selected, rank_by)
    st.altair_chart(city_bars(cities, rank_by), width="stretch")
    n_cities = selected.groupby(["city", "state"]).ngroups
    text = (f"These {len(cities)} of {n_cities:,} cities bring "
            f"**{pct(cities['sales'].sum() / k['sales'], 0)} of sales**")
    if k["profit"] > 0:
        text += f" and **{pct(cities['profit'].sum() / k['profit'], 0)} of profit**"
    st.markdown(text + " in this selection.")
    numbers_table(cities, ["label", "sales", "profit", "margin"])

    # ---- monthly trend
    st.subheader("Monthly trend by region")
    measure = st.radio("Show", ["Sales", "Profit"], horizontal=True, key="trend_measure").lower()
    monthly = monthly_by_region(selected)
    st.altair_chart(trend_lines(monthly, measure), width="stretch")
    regions_total = summarize(selected, ["region"]).sort_values("sales", ascending=False)
    lead, low = regions_total.iloc[0], regions_total.sort_values("margin").iloc[0]
    q4 = selected.loc[selected["order_month"] >= 9, "sales"].sum() / k["sales"]
    st.markdown(
        f"**{lead['region']}** has the most sales ({pct(lead['sales'] / k['sales'], 0)}), and "
        f"**{low['region']}** the lowest margin ({pct(low['margin'])}). "
        f"September-December (4 of 12 months) bring **{pct(q4, 0)}** of sales.")
    table = monthly.assign(month=monthly["month"].dt.strftime("%b %Y"))
    numbers_table(table, ["month", "region", "sales", "profit", "margin"])

    # ---- categories and sub-categories
    st.subheader("Sales and profit by category and sub-category")
    subs = summarize(selected, ["category", "sub_category"])
    c1, c2 = st.columns(2)
    c1.altair_chart(sub_category_bars(subs, "sales"))
    c2.altair_chart(sub_category_bars(subs, "profit"))
    losers = subs[subs["profit"] < 0].sort_values("profit")
    if losers.empty:
        st.markdown("Every sub-category is profitable in this selection.")
    else:
        st.markdown(
            f"**{len(losers)} sub-categories lose money**: "
            + ", ".join(f"{r.sub_category} ({money(r.profit)})" for r in losers.itertuples())
            + f". Together they bring {pct(losers['sales'].sum() / k['sales'], 0)} of sales.")
    numbers_table(subs.sort_values(["category", "sales"], ascending=[True, False]),
                  ["category", "sub_category", "sales", "profit", "margin"])

    # ---- segments
    st.subheader("Sales by customer segment")
    segs = summarize(selected, ["segment"]).sort_values("sales", ascending=False)
    st.altair_chart(segment_bars(segs), width="stretch")
    top = segs.iloc[0]
    spread = segs["margin"].max() - segs["margin"].min()
    st.markdown(
        f"**{top['segment']}** brings {pct(top['sales'] / k['sales'], 0)} of sales. Margins by "
        f"segment range from {pct(segs['margin'].min())} to {pct(segs['margin'].max())}"
        + (", so the segments differ in size much more than in profitability."
           if len(segs) > 1 and spread < 0.03 else "."))
    numbers_table(segs, ["segment", "sales", "profit", "margin"])
