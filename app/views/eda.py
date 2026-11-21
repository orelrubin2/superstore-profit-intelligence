"""Page 3: exploratory data analysis (discounts, geography, time, products, shipping).

Every number comes from a saved query in sql/, run on the filtered order lines
with data.run_query. Card titles are written from the query result, so they
state the finding for the current selection (or fall back to neutral wording).
"""

import altair as alt
import pandas as pd
import streamlit as st

from app import data
from app.style import (BLUE, BORDER, CARD, CARD_CSS, CATEGORY_COLORS, DIVERGING, GRID, LOSS,
                       MUTED, ORANGE, PAGE_CSS, SEQUENTIAL, TEXT, card, money, pct)

ALL_YEARS = "All years"
YEARS = [2014, 2015, 2016, 2017]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NEUTRAL = "#4a5670"   # de-emphasised bars
CHART_HEIGHT = 200    # most cards
MAP_HEIGHT = 240      # state map (row 1)
REGION_HEIGHT = 215   # regions card, next to the map

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

# Compact cards (on top of style.PAGE_CSS): the page should fit, or nearly fit, one
# 1920x1080 screen. Fonts stay at 11 px or more; chart heights are fixed per card.
COMPACT_CSS = """
<style>
[class*="st-key-card-"] { padding: 0.55rem 0.8rem 0.45rem 0.8rem; gap: 0.2rem !important; }
[class*="st-key-card-"] [data-testid="stVerticalBlock"] { gap: 0.2rem; }
/* Streamlit pairs a 1rem bottom margin on paragraphs with -1rem on their wrapper; zero both */
[class*="st-key-card-"] [data-testid="stMarkdownContainer"],
[class*="st-key-card-"] [data-testid="stCaptionContainer"] { margin-bottom: 0 !important; }
[class*="st-key-card-"] [data-testid="stMarkdownContainer"] p { font-size: 13.5px;
    line-height: 1.35; margin-bottom: 0; }
[class*="st-key-card-"] [data-testid="stCaptionContainer"] p { font-size: 12px; line-height: 1.3;
    margin-bottom: 0; }
[class*="st-key-card-"] [data-testid="stExpander"] summary { padding: 0.15rem 0.6rem; min-height: 0; }
[class*="st-key-card-"] [data-testid="stExpander"] summary p { font-size: 12px; }
[class*="st-key-card-"] [data-testid="stRadio"] label p { font-size: 12.5px; }
[class*="st-key-card-kpi"] { padding: 0.45rem 0.8rem 0.3rem 0.8rem; }
[data-testid="stMetricChart"] { height: 28px; overflow: hidden; margin-top: 2px; }
[data-testid="stMetricChart"] svg { transform: scaleY(0.54); transform-origin: top left; }
</style>
"""

# KPI label, key, value format, change type, monthly sparkline column (sql/10 part 1)
KPIS = [("Sales", "sales", money, "growth", "total_sales"),
        ("Profit", "profit", money, "growth", "total_profit"),
        ("Profit margin", "margin", pct, "pp", "profit_margin_pct"),
        ("Orders", "orders", "{:,}".format, "growth", "orders"),
        ("Lines discounted >20%", "high_share", pct, "pp", "pct_high_discount_lines")]


# ---------------------------------------------------------------- helpers

def esc(text: str) -> str:
    """Escape $ so Streamlit markdown does not read money amounts as LaTeX."""
    return text.replace("$", "\\$")


def weighted(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean of a per-group percentage (e.g. pct_loss_lines by order_lines), as a fraction."""
    return (values * weights).sum() / weights.sum() / 100


def header(section: str, title: str, subtitle: str | None = None) -> None:
    """Two lines: the finding in bold, then 'Section · what the chart shows'."""
    st.markdown(f"**{esc(title)}**")
    st.caption(esc(f"{section} · {subtitle}" if subtitle else section))


def show(chart: alt.TopLevelMixin) -> None:
    """Render a chart with light grey axes and faint gridlines for the dark cards."""
    st.altair_chart(chart.configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID,
                                         labelColor=MUTED, titleColor=MUTED, labelFontSize=11,
                                         titleFontSize=11, titleFontWeight="normal")
                    .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFontSize=11,
                                      titleFontSize=11, symbolSize=60, padding=0, offset=4)
                    .configure_view(strokeWidth=0)
                    .configure(background=CARD), width="stretch")


def numbers(df: pd.DataFrame, file: str, statement: int = 1) -> None:
    """Expander with the query result as returned by sql/<file>."""
    with st.expander("Numbers behind the chart"):
        st.caption(f"Source: sql/{file}" + (f", part {statement}" if statement > 1 else ""))
        fmt = {c: "${:,.0f}" for c in df.columns
               if c.startswith(("total_", "avg_monthly_", "profit_from"))}
        fmt |= {c: "{:.2f}%" for c in df.columns if c.endswith("_pct") or "pct_" in c}
        st.dataframe(df.style.format(fmt), hide_index=True, width="stretch")


def zero_rule() -> alt.Chart:
    return alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color=MUTED, strokeDash=[4, 4]) \
        .encode(y="zero:Q")


def signed_bars(df: pd.DataFrame, y: str, x: str, label: str, x_title: str | None,
                x_format: str = "$,.0f", tooltip: list | None = None,
                sort: list | None = None, size: int = 16,
                x_domain: list | None = None) -> alt.LayerChart:
    """Horizontal bars, blue when positive and red when negative, with a text label per bar.

    Rows are sorted by `x`, largest first, unless an explicit `sort` list is given.
    Labels sit at the bar end for positive values and right of zero for negative ones.
    """
    df = df.assign(label_x=df[x].clip(lower=0))
    sort = sort or df.sort_values(x, ascending=False)[y].tolist()
    bars = (alt.Chart(df)
            .mark_bar(cornerRadiusEnd=3, size=size)
            .encode(y=alt.Y(f"{y}:N", sort=sort, title=None, axis=alt.Axis(labelOverlap=False)),
                    x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(format=x_format),
                            scale=alt.Scale(domain=x_domain) if x_domain else alt.Undefined),
                    color=alt.condition(f"datum['{x}'] >= 0", alt.value(BLUE), alt.value(LOSS)),
                    tooltip=tooltip or [f"{y}:N", alt.Tooltip(f"{x}:Q", format=x_format)]))
    text = (alt.Chart(df)
            .mark_text(align="left", dx=4, fontSize=11, color=TEXT)
            .encode(y=alt.Y(f"{y}:N", sort=sort), x="label_x:Q", text=f"{label}:N"))
    return bars + text


# ---------------------------------------------------------------- KPIs

def kpi_values(monthly: pd.DataFrame) -> dict:
    """Headline numbers summed from monthly rows of sql/10 part 1 (all columns are additive)."""
    sales, profit = monthly["total_sales"].sum(), monthly["total_profit"].sum()
    return {"sales": sales, "profit": profit, "margin": profit / sales,
            "orders": int(monthly["orders"].sum()),
            "high_share": weighted(monthly["pct_high_discount_lines"], monthly["order_lines"])}


def change(now: float, before: float, kind: str) -> str:
    """'+20.4%' growth or '-0.7 pp' difference (ASCII sign so st.metric picks the arrow)."""
    if kind == "growth":
        return f"{(now - before) / abs(before):+.1%}"
    return f"{(now - before) * 100:+.1f} pp"


def kpi_row(monthly: pd.DataFrame, year: int | None) -> str:
    """Five KPI cards; returns the note on what the change and sparklines compare."""
    current = monthly if year is None else monthly[monthly["order_year"] == year]
    previous = monthly[monthly["order_year"] == year - 1] if year else monthly.iloc[0:0]
    now = kpi_values(current)
    before = kpi_values(previous) if not previous.empty else None
    for column, (label, key, fmt, kind, spark) in zip(st.columns(5), KPIS):
        delta = change(now[key], before[key], kind) if before else None
        with column, card(f"kpi-{key}"):
            st.metric(label, fmt(now[key]), delta,
                      delta_color="inverse" if key == "high_share" else "normal",
                      delta_description=f"vs {year - 1}" if delta else None,
                      chart_data=current[spark].tolist(), chart_type="line")
    if year is None:
        return ("All four years; sparklines show the 48 months. "
                "Pick a single year to see the change vs. the previous year.")
    if before is None:
        return (f"Sparklines: monthly values in {year}. {year} is the first year, "
                "so there is no previous year to compare with.")
    return f"Change vs. {year - 1}. Sparklines: monthly values in {year}."


# ---------------------------------------------------------------- sections

def discounts(years: tuple, regions: tuple, categories: tuple) -> None:
    file = "03_discount_impact_by_category.sql"
    df = data.run_query(file, 1, years, regions, categories)
    high = df[df["discount_band"].isin(["21-40%", ">40%"])]
    bands = (df.groupby("discount_band")[["total_sales", "total_profit"]].sum()
             .reindex(data.DISCOUNT_BANDS).dropna())
    margins = bands["total_profit"] / bands["total_sales"]
    cats = sorted(df["category"].unique())
    if not high.empty and (high["profit_margin_pct"] < 0).all():
        who = "every category loses" if len(cats) > 1 else f"{cats[0]} loses"
        title = f"Above 20% discount, {who} money"
    else:
        title = (f"Margin goes from {pct(margins.iloc[0])} at {margins.index[0]} discount "
                 f"to {pct(margins.iloc[-1])} at {margins.index[-1]}")
    with card("discounts"):
        header("Discounts", title, "margin by discount band, one line per category")
        chart_df = df.assign(margin=df["profit_margin_pct"] / 100,
                             loss_share=df["pct_loss_lines"] / 100)
        color = alt.Color("category:N", title=None, legend=alt.Legend(orient="top"),
                          scale=alt.Scale(domain=cats, range=[CATEGORY_COLORS[c] for c in cats]))
        lines = (alt.Chart(chart_df)
                 .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(filled=True, size=50))
                 .encode(x=alt.X("discount_band:N", sort=data.DISCOUNT_BANDS, title=None,
                                 axis=alt.Axis(labelAngle=0)),
                         y=alt.Y("margin:Q", title="Profit margin", axis=alt.Axis(format="%")),
                         color=color,
                         tooltip=[alt.Tooltip("category:N", title="Category"),
                                  alt.Tooltip("discount_band:N", title="Band"),
                                  alt.Tooltip("order_lines:Q", title="Order lines", format=","),
                                  alt.Tooltip("total_sales:Q", title="Sales", format="$,.0f"),
                                  alt.Tooltip("total_profit:Q", title="Profit", format="$,.0f"),
                                  alt.Tooltip("margin:Q", title="Margin", format=".1%"),
                                  alt.Tooltip("loss_share:Q", title="Loss-making lines",
                                              format=".0%")]))
        show((zero_rule() + lines).properties(height=CHART_HEIGHT - 10))
        if high.empty:
            st.markdown("No order lines in this selection are discounted above 20%.")
        else:
            st.markdown(esc(
                f"Order lines discounted above 20% bring {money(high['total_sales'].sum())} in "
                f"sales and **{money(high['total_profit'].sum())} in profit**; "
                f"{pct(weighted(high['pct_loss_lines'], high['order_lines']), 0)} of them "
                "lose money."))
        numbers(df, file)


def regions_card(years: tuple, regions: tuple, categories: tuple) -> None:
    file = "04_high_discounts_by_region_segment.sql"
    df = data.run_query(file, 1, years, regions, categories)
    low = df.sort_values("profit_margin_pct").iloc[0]
    most = df.sort_values("pct_high_discount_lines").iloc[-1]
    if len(df) == 1:
        title = f"{low['region']} earns a {pct(low['profit_margin_pct'] / 100)} margin"
    elif low["region"] == most["region"]:
        title = (f"{low['region']} has the lowest margin "
                 f"({pct(low['profit_margin_pct'] / 100)}) and the most high discounts")
    else:
        title = f"{low['region']} has the lowest margin ({pct(low['profit_margin_pct'] / 100)})"
    with card("regions"):
        header("Regions", title, "profit by region; label = margin")
        chart_df = df.assign(margin=df["profit_margin_pct"] / 100,
                             label=(df["profit_margin_pct"] / 100).map(lambda m: pct(m)))
        tooltip = [alt.Tooltip("region:N", title="Region"),
                   alt.Tooltip("total_profit:Q", title="Profit", format="$,.0f"),
                   alt.Tooltip("margin:Q", title="Margin", format=".1%"),
                   alt.Tooltip("pct_high_discount_lines:Q", title="% lines >20% discount",
                               format=".1f")]
        show(signed_bars(chart_df, "region", "total_profit", "label", None, tooltip=tooltip,
                         size=22).properties(height=REGION_HEIGHT))
        st.markdown(esc(
            f"**{most['region']}** discounts the most lines above 20% "
            f"({most['pct_high_discount_lines']:.1f}%), and those lines bring "
            f"**{money(most['profit_from_high_discounts'])}** in profit."))
        numbers(df, file)


def state_frame(df: pd.DataFrame) -> pd.DataFrame:
    """sql/09 result with margin as a fraction and the FIPS code each state has in US_TOPOJSON."""
    return df.assign(margin=df["profit_margin_pct"] / 100,
                     code=df["state"].map(lambda s: STATES[s][0]),
                     fips=df["state"].map(lambda s: STATES[s][1]))


def state_map(states: pd.DataFrame, metric: str) -> alt.LayerChart:
    """Choropleth of one metric: dim -> bright blue for sales, red <- slate -> blue otherwise.

    States with no orders in the selection keep the card color with a visible outline.
    """
    field = {"sales": "total_sales", "profit": "total_profit", "margin": "margin"}[metric]
    fmt = ".0%" if metric == "margin" else "$,.0f"
    if metric == "sales":
        scale = alt.Scale(range=SEQUENTIAL, interpolate="lab")
    else:
        scale = alt.Scale(domainMid=0, range=DIVERGING, interpolate="lab")
    topo = alt.topo_feature(US_TOPOJSON, "states")
    background = alt.Chart(topo).mark_geoshape(fill=CARD, stroke=BORDER, strokeWidth=0.75)
    shapes = (alt.Chart(topo)
              .mark_geoshape(stroke=CARD, strokeWidth=0.75)
              .transform_lookup(lookup="id", from_=alt.LookupData(
                  states, "fips", ["state", "total_sales", "total_profit", "margin"]))
              .transform_filter("isValid(datum.state)")
              .encode(color=alt.Color(f"{field}:Q", scale=scale, title=metric.capitalize(),
                                      legend=alt.Legend(format=fmt, orient="right",
                                                        gradientLength=150)),
                      tooltip=[alt.Tooltip("state:N", title="State"),
                               alt.Tooltip("total_sales:Q", title="Sales", format="$,.0f"),
                               alt.Tooltip("total_profit:Q", title="Profit", format="$,.0f"),
                               alt.Tooltip("margin:Q", title="Margin", format=".1%")]))
    # Red outline on loss-making states, so small losses do not read as break-even slate
    losses = (alt.Chart(topo)
              .mark_geoshape(filled=False, stroke=LOSS, strokeWidth=1.5)
              .transform_lookup(lookup="id", from_=alt.LookupData(states, "fips", ["total_profit"]))
              .transform_filter("isValid(datum.total_profit) && datum.total_profit < 0"))
    return (background + shapes + losses).project(type="albersUsa").properties(height=MAP_HEIGHT)


def states_card(years: tuple, regions: tuple, categories: tuple) -> None:
    file = "09_state_profitability.sql"
    df = data.run_query(file, 1, years, regions, categories)
    losing, winning = df[df["total_profit"] < 0], df[df["total_profit"] >= 0]
    if losing.empty:
        title = f"All {len(df)} states are profitable"
    else:
        title = (f"{len(losing)} of {len(df)} states lose money, "
                 f"together {money(losing['total_profit'].sum())}")
    with card("states"):
        head, pick = st.columns([3, 1], vertical_alignment="center")
        with head:
            header("States", title, "red outline = state loses money; uncolored = no orders. "
                   "Profit and margin: red = loss, slate = break-even, blue = profit")
        metric = pick.radio("Map metric", ["Sales", "Profit", "Margin"], horizontal=True,
                            key="eda_map_metric", label_visibility="collapsed").lower()
        show(state_map(state_frame(df), metric))
        if not losing.empty and not winning.empty:
            share_losing = weighted(losing["pct_high_discount_lines"], losing["order_lines"])
            share_winning = weighted(winning["pct_high_discount_lines"], winning["order_lines"])
            st.markdown(f"Loss-making states discount **{pct(share_losing)}** of their lines "
                        f"above 20%, profitable states **{pct(share_winning)}**.")
        numbers(df, file)


def trend_card(monthly: pd.DataFrame, year: int | None) -> None:
    file = "10_monthly_trend.sql"
    df = monthly.assign(month=pd.to_datetime(monthly["order_ym"] + "-01"))
    yearly = df.groupby("order_year")["total_sales"].sum()
    first, last = yearly.index.min(), yearly.index.max()
    growth = yearly[last] / yearly[first] - 1
    title = (f"Sales {'grew' if growth >= 0 else 'fell'} {pct(abs(growth), 0)} "
             f"from {first} to {last}")
    subtitle = "all four years, ignores the Year filter"
    if year:
        subtitle += f"; {year} shaded"
    domain = [(df["month"].min() - pd.Timedelta(days=16)).isoformat(),
              (df["month"].max() + pd.Timedelta(days=16)).isoformat()]
    x = alt.X("month:T", title=None, scale=alt.Scale(domain=domain),
              axis=alt.Axis(format="%Y", tickCount="year", labelAngle=0))
    x_hidden = alt.X("month:T", title=None, scale=alt.Scale(domain=domain),
                     axis=alt.Axis(labels=False, ticks=False, tickCount="year"))
    tooltip = [alt.Tooltip("month:T", title="Month", format="%B %Y"),
               alt.Tooltip("total_sales:Q", title="Sales", format="$,.0f"),
               alt.Tooltip("total_profit:Q", title="Profit", format="$,.0f")]

    def shade() -> alt.Chart:
        band = pd.DataFrame({"start": [pd.Timestamp(year or first, 1, 1) - pd.Timedelta(days=15)],
                             "end": [pd.Timestamp(year or first, 12, 1) + pd.Timedelta(days=15)]})
        return alt.Chart(band).mark_rect(color=MUTED, opacity=0.15 if year else 0) \
            .encode(x="start:T", x2="end:T")

    with card("trend"):
        header("Monthly trend", title, subtitle)
        sales = alt.Chart(df).mark_line(color=BLUE, strokeWidth=2).encode(
            x=x_hidden, y=alt.Y("total_sales:Q", title="Sales",
                                axis=alt.Axis(format="$~s", tickCount=3)),
            tooltip=tooltip)
        show((shade() + sales).properties(height=100))
        profit = alt.Chart(df).mark_bar(size=4).encode(
            x=x, y=alt.Y("total_profit:Q", title="Profit",
                         axis=alt.Axis(format="$~s", tickCount=2)),
            color=alt.condition("datum.total_profit >= 0", alt.value(BLUE), alt.value(LOSS)),
            tooltip=tooltip)
        show((shade() + profit).properties(height=72))
        negative = int((df["total_profit"] < 0).sum())
        st.markdown(
            f"Profit is noisier than sales: monthly margins range from "
            f"{pct(df['profit_margin_pct'].min() / 100)} to "
            f"{pct(df['profit_margin_pct'].max() / 100)}"
            + (f", and **{negative} of {len(df)} months lost money**." if negative else "."))
        numbers(monthly, file)


def seasonality_card(monthly: pd.DataFrame, regions: tuple, categories: tuple,
                     year: int | None) -> None:
    file = "10_monthly_trend.sql"
    df = data.run_query(file, 2, (), regions, categories)
    df = df.assign(month=df["order_month"].map(lambda m: MONTHS[m - 1]),
                   share=df["avg_pct_of_year_sales"] / 100)
    top = df.nlargest(3, "share").sort_values("order_month")
    weak = df.nsmallest(2, "share").sort_values("order_month")
    role = df["order_month"].map(lambda m: "Strongest 3" if m in top["order_month"].values
                                 else "Weakest 2" if m in weak["order_month"].values else "Other")
    title = (f"{', '.join(top['month'][:-1])} and {top['month'].iloc[-1]} bring "
             f"{pct(top['share'].sum(), 0)} of annual sales; {' and '.join(weak['month'])} "
             f"only {pct(weak['share'].sum(), 0)}")
    subtitle = "avg. share of each year's sales 2014-2017; ignores the Year filter"
    if year:
        subtitle += f"; ticks = {year}"
    with card("seasonality"):
        header("Seasonality", title, subtitle)
        chart_df = df.assign(role=role)
        bars = (alt.Chart(chart_df).mark_bar(cornerRadiusEnd=3)
                .encode(x=alt.X("month:N", sort=MONTHS, title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("share:Q", title=None, axis=alt.Axis(format="%", tickCount=4)),
                        color=alt.Color("role:N", title=None, legend=alt.Legend(orient="top"),
                                        scale=alt.Scale(domain=["Strongest 3", "Weakest 2", "Other"],
                                                        range=[BLUE, ORANGE, NEUTRAL])),
                        tooltip=[alt.Tooltip("month:N", title="Month"),
                                 alt.Tooltip("share:Q", title="Average share", format=".1%"),
                                 alt.Tooltip("min_pct_of_year_sales:Q", title="Lowest year %",
                                             format=".1f"),
                                 alt.Tooltip("max_pct_of_year_sales:Q", title="Highest year %",
                                             format=".1f"),
                                 alt.Tooltip("profit_margin_pct:Q", title="Margin %",
                                             format=".1f")]))
        layers = bars
        if year:
            this = monthly[monthly["order_year"] == year]
            ticks_df = this.assign(month=this["order_month"].map(lambda m: MONTHS[m - 1]),
                                   share=this["total_sales"] / this["total_sales"].sum())
            ticks = (alt.Chart(ticks_df).mark_tick(color=TEXT, thickness=2, size=14)
                     .encode(x=alt.X("month:N", sort=MONTHS), y="share:Q",
                             tooltip=[alt.Tooltip("month:N", title="Month"),
                                      alt.Tooltip("share:Q", title=f"Share in {year}",
                                                  format=".1%")]))
            layers = bars + ticks
        show(layers.properties(height=CHART_HEIGHT - 20))
        weakest = weak.sort_values("share").iloc[0]
        text = (f"{weakest['month']} was never above "
                f"{pct(weakest['max_pct_of_year_sales'] / 100)} of a year's sales.")
        if year:
            best = ticks_df.sort_values("share").iloc[-1]
            text += f" In {year}, the strongest month was **{best['month']}** ({pct(best['share'])})."
        st.markdown(text)
        numbers(df.drop(columns=["month", "share"]), file, 2)


def categories_card(years: tuple, regions: tuple, categories: tuple) -> None:
    file = "01_category_profitability.sql"
    df = data.run_query(file, 1, years, regions, categories)
    low = df.sort_values("profit_margin_pct").iloc[0]
    positive_total = df["total_profit"].sum() > 0
    if len(df) == 1:
        title = (f"{low['category']} earns a {pct(low['profit_margin_pct'] / 100)} margin on "
                 f"{money(low['total_sales'])}")
    elif positive_total and low["pct_of_profit"] < low["pct_of_sales"] / 2:
        title = (f"{low['category']} brings {low['pct_of_sales']:.0f}% of sales "
                 f"but only {low['pct_of_profit']:.0f}% of profit")
    else:
        title = f"{low['category']} has the lowest margin ({pct(low['profit_margin_pct'] / 100)})"
    with card("categories"):
        if len(df) > 1 and positive_total:
            header("Categories", title, "share of sales vs. share of profit")
            long = df.melt(id_vars=["category", "profit_margin_pct"],
                           value_vars=["pct_of_sales", "pct_of_profit"],
                           var_name="measure", value_name="share")
            long["measure"] = long["measure"].map({"pct_of_sales": "Share of sales",
                                                   "pct_of_profit": "Share of profit"})
            long["share"] = long["share"] / 100
            chart = (alt.Chart(long).mark_bar(cornerRadiusEnd=3, size=14)
                     .encode(y=alt.Y("category:N", title=None),
                             yOffset=alt.YOffset("measure:N",
                                                 sort=["Share of sales", "Share of profit"]),
                             x=alt.X("share:Q", title=None, axis=alt.Axis(format="%")),
                             color=alt.Color("measure:N", title=None,
                                             legend=alt.Legend(orient="top"),
                                             scale=alt.Scale(domain=["Share of sales",
                                                                     "Share of profit"],
                                                             range=[NEUTRAL, BLUE])),
                             tooltip=[alt.Tooltip("category:N", title="Category"),
                                      alt.Tooltip("measure:N", title="Measure"),
                                      alt.Tooltip("share:Q", title="Share", format=".1%"),
                                      alt.Tooltip("profit_margin_pct:Q", title="Margin %",
                                                  format=".1f")]))
        else:
            header("Categories", title, "profit by category; label = margin")
            chart_df = df.assign(label=(df["profit_margin_pct"] / 100).map(lambda m: pct(m)))
            chart = signed_bars(chart_df, "category", "total_profit", "label", None)
        show(chart.properties(height=CHART_HEIGHT - 20))
        st.markdown(esc(" · ".join(
            f"{r.category}: {pct(r.profit_margin_pct / 100)} margin" for r in df.itertuples())))
        numbers(df, file)


def sub_categories_card(years: tuple, regions: tuple, categories: tuple) -> None:
    file = "11_subcategory_profitability.sql"
    df = data.run_query(file, 1, years, regions, categories)
    losers = df[df["total_profit"] < 0].sort_values("total_profit")
    if losers.empty:
        title = f"All {len(df)} sub-categories are profitable"
    elif len(losers) <= 3:
        names = list(losers["sub_category"])
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        title = f"{listed} {'loses' if len(names) == 1 else 'lose'} money"
    else:
        title = f"{len(losers)} of {len(df)} sub-categories lose money"
    with card("sub-categories"):
        header("Sub-categories", title, "profit by sub-category; label = margin")
        chart_df = df.assign(label=(df["profit_margin_pct"] / 100).map(lambda m: pct(m)))
        tooltip = [alt.Tooltip("sub_category:N", title="Sub-category"),
                   alt.Tooltip("category:N", title="Category"),
                   alt.Tooltip("total_sales:Q", title="Sales", format="$,.0f"),
                   alt.Tooltip("total_profit:Q", title="Profit", format="$,.0f"),
                   alt.Tooltip("avg_discount_pct:Q", title="Avg discount %", format=".1f")]
        show(signed_bars(chart_df, "sub_category", "total_profit", "label", None,
                         tooltip=tooltip, size=11).properties(height=16 * len(df) + 34))
        if not losers.empty:
            st.markdown(esc(
                f"Together they lose **{money(losers['total_profit'].sum())}** on "
                f"{pct(losers['total_sales'].sum() / df['total_sales'].sum(), 0)} of sales."))
        numbers(df, file)


def shipping_card(years: tuple, regions: tuple, categories: tuple) -> None:
    file = "08_shipping.sql"
    df = data.run_query(file, 1, years, regions, categories)
    order = list(df["ship_mode"])   # the query sorts by average ship days
    lo, hi = df["profit_margin_pct"].min() / 100, df["profit_margin_pct"].max() / 100
    if len(df) > 1 and hi - lo < 0.03:
        title = f"Faster shipping does not change margin: {pct(lo)} to {pct(hi)} in every mode"
    else:
        best = df.sort_values("profit_margin_pct").iloc[-1]
        worst = df.sort_values("profit_margin_pct").iloc[0]
        title = (f"{best['ship_mode']} has the highest margin ({pct(hi)}), "
                 f"{worst['ship_mode']} the lowest ({pct(lo)})")
    with card("shipping"):
        header("Shipping", title, "margin by ship mode, fastest first; label = margin · days")
        chart_df = df.assign(
            margin=df["profit_margin_pct"] / 100,
            margin_label=(df["profit_margin_pct"] / 100).map(lambda m: pct(m)),
            days_label=df["avg_ship_days"].map(lambda d: f"{d:.1f} days"))
        chart_df["label"] = chart_df["margin_label"] + " · " + chart_df["days_label"]
        tooltip = [alt.Tooltip("ship_mode:N", title="Ship mode"),
                   alt.Tooltip("order_lines:Q", title="Order lines", format=","),
                   alt.Tooltip("total_sales:Q", title="Sales", format="$,.0f"),
                   alt.Tooltip("total_profit:Q", title="Profit", format="$,.0f"),
                   alt.Tooltip("margin:Q", title="Margin", format=".1%"),
                   alt.Tooltip("avg_ship_days:Q", title="Avg ship days", format=".2f"),
                   alt.Tooltip("avg_discount_pct:Q", title="Avg discount %", format=".1f")]
        # the x domain leaves room for the 'margin · days' label right of each bar
        show(signed_bars(chart_df, "ship_mode", "margin", "label", None, x_format="%",
                         tooltip=tooltip, sort=order, size=20,
                         x_domain=[min(0.0, lo), hi * 1.8]).properties(height=CHART_HEIGHT - 20))
        main = df.sort_values("pct_of_sales").iloc[-1]
        fastest = df.iloc[0]
        st.markdown(esc(
            f"**{main['ship_mode']}** carries {main['pct_of_sales']:.0f}% of sales and ships in "
            f"{main['avg_ship_days']:.1f} days on average; {fastest['ship_mode']} ships in "
            f"{fastest['avg_ship_days']:.1f}. Average discounts range from "
            f"{df['avg_discount_pct'].min():.1f}% to {df['avg_discount_pct'].max():.1f}% by mode."))
        numbers(df, file)


# ---------------------------------------------------------------- page

def render() -> None:
    st.markdown(CARD_CSS + PAGE_CSS + COMPACT_CSS, unsafe_allow_html=True)
    lines = data.load_lines()
    head, f1, f2, f3 = st.columns([2.2, 1, 1.2, 1.2], vertical_alignment="bottom")
    with head:
        st.title("Exploratory data analysis")
        note = st.empty()
    choice = f1.selectbox("Year", [ALL_YEARS, *YEARS], index=len(YEARS), key="eda_year")
    regions = f2.multiselect("Region", sorted(lines["region"].unique()), placeholder="All regions",
                             key="eda_region")
    categories = f3.multiselect("Category", sorted(lines["category"].unique()),
                                placeholder="All categories", key="eda_category")
    year = None if choice == ALL_YEARS else int(choice)
    years, regions, categories = ((year,) if year else ()), tuple(regions), tuple(categories)

    if data.filter_lines(lines, order_year=list(years), region=list(regions),
                         category=list(categories)).empty:
        st.warning("No order lines match these filters.")
        return

    monthly = data.run_query("10_monthly_trend.sql", 1, (), regions, categories)
    note.caption("Every chart comes from a saved query in sql/, run on the filtered order "
                 "lines; empty filters mean 'all'. " + kpi_row(monthly, year))

    # row 1: state map (2/3) | regions (1/3)
    wide, narrow = st.columns([2, 1])
    with wide:
        states_card(years, regions, categories)
    with narrow:
        regions_card(years, regions, categories)
    # row 2: discounts | monthly trend | seasonality
    c1, c2, c3 = st.columns(3)
    with c1:
        discounts(years, regions, categories)
    with c2:
        trend_card(monthly, year)
    with c3:
        seasonality_card(monthly, regions, categories, year)
    # row 3: categories | sub-categories | shipping
    c1, c2, c3 = st.columns(3)
    with c1:
        categories_card(years, regions, categories)
    with c2:
        sub_categories_card(years, regions, categories)
    with c3:
        shipping_card(years, regions, categories)
