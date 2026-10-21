"""Page 3: predicted loss risk of a single order line, with a short SHAP explanation."""

import altair as alt
import pandas as pd
import streamlit as st

import src.loss_model as lm
from app import data
from app.style import BLUE, LOSS, pct, signed_pp

FEATURE_LABELS = {"discount": "Discount", "sub_category": "Sub-category", "state": "State",
                  "category": "Category", "region": "Region", "unit_price": "List unit price",
                  "quantity": "Quantity", "ship_mode": "Ship mode", "segment": "Segment",
                  "order_month": "Order month"}


def build_line(model: dict, sub_category: str, state: str, discount: float, quantity: int,
               unit_price: float, ship_mode: str, segment: str, order_month: int) -> pd.DataFrame:
    """One order line in the model's feature format; category and region are derived."""
    return pd.DataFrame([{
        "category": model["category_of"][sub_category], "sub_category": sub_category,
        "region": model["region_of"][state], "state": state, "ship_mode": ship_mode,
        "segment": segment, "order_month": str(order_month),
        "discount": discount, "quantity": quantity, "unit_price": unit_price,
    }])


def describe(feature: str, line: pd.Series) -> str:
    """Readable 'Feature = value' label for the SHAP chart."""
    value = line[feature]
    if feature == "discount":
        value = f"{value:.0%}"
    elif feature == "unit_price":
        value = f"${value:,.2f}"
    return f"{FEATURE_LABELS[feature]} = {value}"


def most_common(series: pd.Series):
    return series.mode().iloc[0]


def render() -> None:
    st.title("Loss-risk checker")
    st.caption("Predicts whether an order line will lose money, using only what is known when "
               "the order is placed. Random Forest trained on 2014-2016 and tested on 2017 "
               "(PR-AUC 0.96; see notebooks/04_loss_risk_model.ipynb).")
    model = data.load_model()
    train = model["train"]
    lines = data.load_lines()

    c1, c2, c3, c4 = st.columns(4)
    sub_categories = sorted(train["sub_category"].unique())
    sub_category = c1.selectbox("Sub-category", sub_categories, index=sub_categories.index("Binders"))
    states = sorted(train["state"].unique())
    state = c2.selectbox("State", states, index=states.index("Texas"))
    discount = c3.select_slider("Discount", options=sorted(lines["discount"].unique()), value=0.2,
                                format_func=lambda d: f"{d:.0%}")
    quantity = c4.number_input("Quantity", min_value=1, max_value=14, value=3)

    with st.expander("Advanced: other inputs (defaults shown; they barely affect the prediction)"):
        a1, a2, a3, a4 = st.columns(4)
        unit_price = a1.number_input("List unit price ($)", min_value=0.5,
                                     value=round(float(model["median_price"][sub_category]), 2),
                                     key=f"price_{sub_category}",
                                     help="Default: median list price of this sub-category.")
        ship_modes = sorted(train["ship_mode"].unique())
        ship_mode = a2.selectbox("Ship mode", ship_modes,
                                 index=ship_modes.index(most_common(train["ship_mode"])))
        segments = sorted(train["segment"].unique())
        segment = a3.selectbox("Segment", segments, index=segments.index(most_common(train["segment"])))
        order_month = a4.selectbox("Order month", list(range(1, 13)),
                                   index=int(most_common(train["order_month"])) - 1)

    line = build_line(model, sub_category, state, discount, quantity, unit_price,
                      ship_mode, segment, order_month)
    base, contributions = lm.explain(model["forest"], line, model["explainer"])
    risk = base + float(contributions.iloc[0].sum())

    # ---- prediction and the two rules
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
    if not state_discount_seen:
        st.info(f"{state} never sold anything at a {discount:.0%} discount in 2014-2017, so the model "
                "has no direct evidence for this combination; treat the prediction with caution.")
    elif similar.empty:
        st.caption(f"No past {sub_category} lines in {state} at {discount:.0%} discount.")
    else:
        st.caption(f"In 2014-2017, {len(similar)} {sub_category} line(s) in {state} at {discount:.0%} "
                   f"discount were sold; {pct(similar['is_loss'].mean(), 0)} of them lost money.")

    # ---- SHAP explanation
    st.subheader("Why this prediction")
    row = contributions.iloc[0]
    top = row.reindex(row.abs().sort_values(ascending=False).index).head(4)
    shown = pd.DataFrame({"factor": [describe(f, line.iloc[0]) for f in top.index],
                          "contribution": top.to_numpy()})
    shown["label"] = shown["contribution"].map(signed_pp)
    chart = (alt.Chart(shown)
             .mark_bar(size=26, cornerRadiusEnd=4)
             .encode(y=alt.Y("factor:N", sort=None, title=None),
                     x=alt.X("contribution:Q", title="Change in predicted loss risk",
                             axis=alt.Axis(format="+%")),
                     color=alt.condition("datum.contribution >= 0", alt.value(LOSS), alt.value(BLUE)),
                     tooltip=["factor:N", alt.Tooltip("contribution:Q", format="+.1%")]))
    labels = chart.mark_text(align="left", dx=4, fontSize=12).encode(
        text="label:N", color=alt.value("#0b0b0b"))
    st.altair_chart((chart + labels).properties(height=200), width="stretch")

    parts = [f"{describe(f, line.iloc[0]).lower()} ({signed_pp(v)})" for f, v in top.items()]
    rest = float(row.drop(top.index).sum())
    st.markdown(
        f"Starting from the average training line ({pct(base, 1)}), the biggest factors are "
        + ", ".join(parts)
        + f". The other features together add {signed_pp(rest)}, giving {pct(risk, 0)}."
    )
    st.caption("Red pushes the risk up, blue pulls it down (SHAP values from the Random Forest; "
               "one-hot columns summed per feature). The model shows association, not cause.")
