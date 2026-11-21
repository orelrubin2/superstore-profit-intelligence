"""Order-line checker: loss risk (step 4) and profit forecast (step 4b) for one order line."""

import altair as alt
import pandas as pd
import streamlit as st

import src.loss_model as lm
import src.profit_model as pm
from app import data
from app.style import BLUE, LOSS, TEXT, pct, signed_pp

FEATURE_LABELS = {"discount": "Discount", "sub_category": "Sub-category", "state": "State",
                  "category": "Category", "region": "Region", "unit_price": "List price",
                  "quantity": "Quantity", "ship_mode": "Ship mode", "segment": "Segment",
                  "order_month": "Order month", "sales": "Sales"}


def dollars(value: float) -> str:
    """$1,234.56 or −$1,234.56 (true minus sign)."""
    return f"−${abs(value):,.2f}" if value < 0 else f"${value:,.2f}"


def esc(text: str) -> str:
    """Escape $ so Streamlit markdown does not read money amounts as LaTeX."""
    return text.replace("$", "\\$")


def describe(feature: str, line: pd.Series) -> str:
    """Readable 'Feature = value' label for the SHAP charts."""
    value = line[feature]
    if feature == "discount":
        value = f"{value:.0%}"
    elif feature in ("unit_price", "sales"):
        value = dollars(value)
    return f"{FEATURE_LABELS[feature]} = {value}"


def contribution_chart(shown: pd.DataFrame, x_title: str, color_if_positive: str,
                       color_if_negative: str) -> alt.LayerChart:
    """Horizontal bars of per-feature contributions (in percentage points), labelled."""
    chart = (alt.Chart(shown)
             .mark_bar(size=26, cornerRadiusEnd=4)
             .encode(y=alt.Y("factor:N", sort=None, title=None),
                     x=alt.X("contribution:Q", title=x_title, axis=alt.Axis(format="+%")),
                     color=alt.condition("datum.contribution >= 0", alt.value(color_if_positive),
                                         alt.value(color_if_negative)),
                     tooltip=["factor:N", alt.Tooltip("contribution:Q", format="+.1%")]))
    labels = chart.mark_text(align="left", dx=4, fontSize=12).encode(
        text="label:N", color=alt.value(TEXT))
    return (chart + labels).properties(height=200)


def top_contributions(row: pd.Series, line: pd.Series, n: int = 4) -> pd.DataFrame:
    top = row.reindex(row.abs().sort_values(ascending=False).index).head(n)
    shown = pd.DataFrame({"feature": top.index,
                          "factor": [describe(f, line) for f in top.index],
                          "contribution": top.to_numpy()})
    shown["label"] = shown["contribution"].map(signed_pp)
    return shown


# Loss-risk features the profit model already reflects: its own inputs, plus quantity
# (through Sales) and category (fixed by the sub-category).
PROFIT_MODEL_COVERS = set(pm.RECOMMENDED_FEATURES) | {"quantity", "category"}


def risk_raisers(contributions: pd.Series, line: pd.Series) -> list[str]:
    """Loss-risk SHAP factors that push the risk up but are not used by the profit model,
    largest first, e.g. ['state = Texas (+12.3 pp)', 'region = Central (+4.5 pp)']."""
    raising = contributions[(contributions > 0) & ~contributions.index.isin(PROFIT_MODEL_COVERS)]
    return [f"{describe(f, line).lower()} ({signed_pp(v)})"
            for f, v in raising.sort_values(ascending=False).items()]


def disagreement_note(risk: float, profit: float, expected_error: float, stats: dict,
                      base_rate: float | None = None,
                      raisers: list[str] | None = None) -> str | None:
    """Explain when the loss risk and the profit forecast point different ways.

    Two cases: a real disagreement (risk >= 50% vs. the profit sign), and an elevated risk
    (above the average line, below 50%) next to a profit clearly above zero (by more than
    the expected error). `raisers` are the loss-risk factors the profit model does not use
    (see risk_raisers). Returns None when the two point the same way.
    """
    raisers = raisers or []
    unused = (f" The factors pushing the risk up that the profit model does not use: "
              f"{', '.join(raisers)}; the profit model does not see them."
              if raisers else "")
    risk_says_loss, profit_says_loss = risk >= 0.5, profit < 0
    if risk_says_loss == profit_says_loss:
        elevated = base_rate is not None and base_rate <= risk < 0.5
        if elevated and profit > expected_error:
            return (f"The loss risk is elevated ({pct(risk, 0)}, above the average line's "
                    f"{pct(base_rate, 1)}), but the profit forecast is clearly positive "
                    f"({dollars(profit)}, more than its expected error of "
                    f"{dollars(expected_error)}).{unused} The profit model works only from the "
                    "list price, Sales, discount and sub-category, so it predicts the margin "
                    "of the product line itself; the extra risk comes from where and how it "
                    "is sold.")
        return None
    if risk_says_loss:
        text = (f"The two models disagree: the loss-risk model expects a loss ({pct(risk, 0)}), "
                f"but the profit forecast is positive ({dollars(profit)}).")
    else:
        text = (f"The two models disagree: the profit forecast is a loss ({dollars(profit)}), "
                f"but the loss risk is below 50% ({pct(risk, 0)}).")
    text += (" They use different inputs: the loss-risk model also weighs state, ship mode, "
             "segment and month, while the profit model works from the list price, Sales, "
             "discount and sub-category.")
    if risk_says_loss:
        text += unused
    if abs(profit) <= expected_error:
        text += (f" The forecast is within its expected error ({dollars(expected_error)}) of "
                 "break-even, so treat this line as a thin margin either way.")
    text += (f" In 2017 the two disagreed on {pct(stats['share'])} of lines; on those lines the "
             f"profit sign matched the actual outcome {pct(stats['profit_sign_right'], 0)} of the "
             f"time and the loss-risk flag {pct(stats['loss_model_right'], 0)}.")
    return text


def render() -> None:
    st.title("Will this order line make or lose money?")
    st.caption("Two models, using only what is known when the order is placed, both trained on "
               "2014-2016 and tested on 2017: the loss risk (Random Forest, PR-AUC 0.96; "
               "notebooks/04_loss_risk_model.ipynb) and the profit forecast (margin x Sales "
               "Random Forest; notebooks/05_profit_model.ipynb).")
    model = data.load_model()
    profit_model = data.load_profit_model()
    train = model["train"]
    lines = data.load_lines()

    c1, c2, c3, c4, c5 = st.columns(5)
    sub_categories = sorted(train["sub_category"].unique())
    sub_category = c1.selectbox("Sub-category", sub_categories, index=sub_categories.index("Binders"))
    states = sorted(train["state"].unique())
    state = c2.selectbox("State", states, index=states.index("Texas"))
    discount = c3.select_slider("Discount", options=sorted(lines["discount"].unique()), value=0.2,
                                format_func=lambda d: f"{d:.0%}")
    quantity = c4.number_input("Quantity", min_value=1, max_value=14, value=3)
    prices = profit_model["prices"].loc[sub_category]
    default_price = round(float(prices["median"]), 2)
    price_range = (f"Typical {sub_category} list price: {dollars(prices['q25'])}-"
                   f"{dollars(prices['q75'])} (middle 50% of 2014-2016 lines), "
                   f"median {dollars(prices['median'])}.")
    unit_price = c5.number_input("List price ($)", min_value=0.01, value=default_price, step=1.0,
                                 format="%.2f", key=f"price_{sub_category}",
                                 help=price_range + " Prefilled with the median; enter the real "
                                      "list price per unit, before discount.")
    st.caption(esc(price_range))
    if not prices["min"] <= unit_price <= prices["max"]:
        st.info(esc(f"{dollars(unit_price)} is outside the {sub_category} list prices seen in "
                    f"2014-2016 ({dollars(prices['min'])}-{dollars(prices['max'])}); both models "
                    "can only extrapolate, so treat the results with caution."))

    with st.expander("Advanced: other inputs for the loss-risk model "
                     "(defaults shown; they barely affect the prediction)"):
        a1, a2, a3 = st.columns(3)
        ship_modes = sorted(train["ship_mode"].unique())
        ship_mode = a1.selectbox("Ship mode", ship_modes,
                                 index=ship_modes.index(model["default_ship_mode"]))
        segments = sorted(train["segment"].unique())
        segment = a2.selectbox("Segment", segments, index=segments.index(model["default_segment"]))
        order_month = a3.selectbox("Order month", list(range(1, 13)),
                                   index=model["default_month"] - 1)

    line = lm.build_line(model, sub_category, state, discount, quantity, unit_price,
                         ship_mode, segment, order_month)
    base, contributions = lm.explain(model["forest"], line, model["explainer"])
    risk = base + float(contributions.iloc[0].sum())

    profit_line = pm.build_line(sub_category, discount, quantity, unit_price)
    sales = float(profit_line["sales"].iloc[0])
    margin_base, margin_contributions, margin = pm.explain_line(profit_model, profit_line)
    profit = margin * sales
    band = pm.expected_error(profit_model["error_bands"], sales)

    # ---- loss risk and the two rules
    st.subheader("Loss risk")
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted loss risk", pct(risk, 0),
              delta=f"{signed_pp(risk - model['base_rate'])} vs. average line",
              delta_color="inverse")
    rule_a = discount > lm.HIGH_DISCOUNT
    r2.metric("Rule A: discount > 20%", "Flag" if rule_a else "No flag")
    rule_b = bool(model["baseline_b"].flags(line)[0])
    r3.metric("Rule B: + sub-category in the 1-20% band", "Flag" if rule_b else "No flag")

    if risk >= 0.5:
        st.error(f"High risk: the model expects this line to lose money ({pct(risk, 0)}).")
    elif risk >= model["base_rate"]:
        st.warning(f"Elevated risk: above the average line ({pct(model['base_rate'], 1)}).")
    else:
        st.success(f"Low risk: below the average line ({pct(model['base_rate'], 1)}).")

    # ---- what actually happened to similar lines
    similar = lines[(lines["sub_category"] == sub_category) & (lines["state"] == state)
                    & (lines["discount"] == discount)]
    state_discount_seen = ((lines["state"] == state) & (lines["discount"] == discount)).any()
    sub_category_discount_seen = ((lines["sub_category"] == sub_category)
                                  & (lines["discount"] == discount)).any()
    if not state_discount_seen:
        st.info(f"{state} never sold anything at a {discount:.0%} discount in 2014-2017, so the model "
                "has no direct evidence for this combination; treat the prediction with caution.")
    if not sub_category_discount_seen:
        st.info(f"{sub_category} were never sold at a {discount:.0%} discount in any state in "
                "2014-2017, so the model has no direct evidence for this combination; treat the "
                "prediction with caution.")
    if similar.empty:
        if state_discount_seen and sub_category_discount_seen:
            st.caption(f"No past {sub_category} lines in {state} at {discount:.0%} discount.")
    else:
        st.caption(f"In 2014-2017, {len(similar)} {sub_category} line(s) in {state} at {discount:.0%} "
                   f"discount were sold; {pct(similar['is_loss'].mean(), 0)} of them lost money.")

    # ---- profit forecast
    st.subheader("Profit forecast")
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted profit", dollars(profit))
    p2.metric("Sales", dollars(sales),
              help="List price x quantity x (1 - discount): "
                   f"{dollars(unit_price)} x {quantity} x {1 - discount:.2f}.")
    p3.metric("Predicted margin", pct(margin))
    st.caption(esc(f"Expected error: for 2017 lines of similar size (Sales {dollars(band['sales_low'])}"
                   f"-{dollars(band['sales_high'])}), the forecast missed by "
                   f"{dollars(band['mae'])} on average."))
    if abs(unit_price - default_price) < 0.005:
        st.warning(esc(
            f"This forecast uses a typical {sub_category} list price ({dollars(default_price)}, the "
            "2014-2016 median), not the real one. Enter the real list price for a reliable "
            f"forecast: with typical prices the 2017 error was about "
            f"{dollars(profit_model['mae_default_price'])} per line, instead of "
            f"{dollars(profit_model['mae_real_price'])} with the real price."))
    note = disagreement_note(risk, profit, float(band["mae"]),
                             profit_model.get("disagreement", {}), model["base_rate"],
                             risk_raisers(contributions.iloc[0], line.iloc[0]))
    if note:
        st.info(esc(note))

    # ---- explanations
    st.subheader("Why these predictions")
    left, right = st.columns(2)
    row = contributions.iloc[0]
    shown = top_contributions(row, line.iloc[0])
    with left:
        st.markdown("**Loss risk**")
        st.altair_chart(contribution_chart(shown, "Change in predicted loss risk", LOSS, BLUE),
                        width="stretch")
        parts = [f"{describe(f, line.iloc[0]).lower()} ({signed_pp(v)})"
                 for f, v in zip(shown["feature"], shown["contribution"])]
        rest = float(row.drop(shown["feature"]).sum())
        st.markdown(
            f"Starting from the average training line ({pct(base, 1)}), the biggest factors are "
            + ", ".join(parts)
            + f". The other features together add {signed_pp(rest)}, giving {pct(risk, 0)}.")
        st.caption("Red pushes the risk up, blue pulls it down.")
    with right:
        st.markdown("**Profit margin**")
        margin_shown = top_contributions(margin_contributions, profit_line.iloc[0])
        st.altair_chart(contribution_chart(margin_shown, "Change in predicted margin", BLUE, LOSS),
                        width="stretch")
        parts = [f"{describe(f, profit_line.iloc[0]).lower()} ({signed_pp(v)})"
                 for f, v in zip(margin_shown["feature"], margin_shown["contribution"])]
        st.markdown(esc(
            f"Starting from the average training line's margin ({pct(margin_base, 1)}): "
            + ", ".join(parts)
            + f". Predicted margin {pct(margin)} x Sales {dollars(sales)} = "
              f"predicted profit {dollars(profit)}."))
        st.caption("Blue raises the margin, red lowers it.")
    st.caption("Both charts show SHAP values from the Random Forests (one-hot columns summed per "
               "feature). The models show association, not cause.")
