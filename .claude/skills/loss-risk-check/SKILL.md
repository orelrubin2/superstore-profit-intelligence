---
name: loss-risk-check
description: Assess whether a proposed order, deal or set of order lines is likely to lose money, using the project's step 4 loss-risk model, the two discount rules and historical outcomes. Use when the user describes a planned sale or discount (e.g. "10 binders to Texas at 30% off") and asks if it is risky.
---

# Loss-risk check

Assess each proposed order line with the `superstore` MCP tool `predict_loss_risk`
and explain the result in business terms.

## Steps
1. **Parse the request into lines**: sub-category, state, discount (as a fraction,
   0.3 = 30%) and quantity for each. Optional: list unit price, ship mode, segment,
   order month. Ask once if sub-category, state or discount is missing. Unknown
   names return an error listing the valid options; use it to correct spelling.
2. **Call `predict_loss_risk`** once per line.
3. **For each line report**:
   - predicted loss risk vs. the average line (`average_line_risk`)
   - Rule A (`discount > 20%`) and Rule B flags
   - the top 2-3 SHAP factors in plain words, using the returned contributions
     (e.g. "the 30% discount raises the risk by +N pp")
   - history: how many similar past lines there were and the share that lost money
   - every warning returned (combination never seen, discount level not in the data)
4. **Suggest a safer alternative** if the risk is high, by re-running the tool with a
   lower discount (e.g. the next level at or below 20%) and reporting how the risk
   changes. Only suggest what the tool output supports.

## Rules
- Use only the tool's numbers. The model shows **association, not cause**: say "lines
  like this usually lose money", not "this discount will cause a loss".
- The model was trained on 2014-2016 and tested on 2017 (PR-AUC 0.96). Its ranking is
  by probability, not loss size.
- With several lines, end with a summary table (line, risk, rules, main factor) and
  the overall verdict.
