"""Step 4b: predict the profit (in dollars) of an order line from order-time features.

Goal: give the Loss-risk checker a predicted profit next to the step 4 loss-risk
probability. This module is analysis only; src/loss_model.py is not changed.

Design (agreed in the step 4b plan):
    - One row per order line from orders_clean ONLY. Target: profit in dollars.
    - Allowed features are known when the order is placed: sales, quantity,
      discount, list price (unit_price), category, sub-category, region, state,
      ship mode, segment and order month. Unlike step 4, sales IS allowed here: it
      is the price of the line, known at order time. Everything derived from
      profit, the outlier flags, IDs and names, and every step 3 customer table
      are forbidden.
    - Train on 2014-2016, test once on 2017. Hyperparameters and the top-4
      feature ranking use only the training years (expanding windows
      2014 -> 2015 and 2014-15 -> 2016, as in step 4).
    - Baselines: (a) the training median profit; (b) sales x the training margin
      of the line's discount band and sub-category.
    - Models: Linear Regression and Random Forest on profit, tuned separately per
      feature set. Agreed rule: because the direct-profit forest did not beat
      baseline (b), a forest on margin x sales (MarginForest) is added for the
      feature sets where sales is known or can be computed.
    - Main metric: MAE on 2017 (dollars per line, robust to the few huge lines);
      RMSE and R2 are reported as well. Paired order-level bootstrap for MAE
      differences.
    - Importance: permutation importance (MAE increase) and SHAP, with one-hot
      columns summed back into their original feature.
"""

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.loss_model import CV_FOLDS, RANDOM_STATE, split_by_year
from src.loss_model import FORBIDDEN_FEATURES as STEP4_FORBIDDEN

TARGET = "profit"
CATEGORICAL_FEATURES = ["category", "sub_category", "region", "state", "ship_mode",
                        "segment", "order_month"]
NUMERIC_FEATURES = ["sales", "quantity", "discount", "unit_price"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
PAGE_FEATURES = ["sub_category", "state", "discount", "quantity"]   # the checker's 4 inputs

# Never allowed: the target and anything derived from it, the outlier flags (analysis
# only), IDs and names, and the step 3 customer tables (they summarize future orders).
# Step 4's list is reused except `sales`, which is an allowed order-time feature here.
FORBIDDEN_FEATURES = (STEP4_FORBIDDEN - {"sales"}) | {
    "profit", "profit_margin", "is_loss", "is_outlier_profit", "is_outlier_sales",
    "row_id", "order_id", "customer_id", "customer_name", "product_id", "product_name",
    "postal_code", "location_id",
}

LINES_SQL = f"""
    SELECT row_id, order_id, order_year, discount_band, {', '.join(ALL_FEATURES)}, {TARGET}
    FROM orders_clean
    ORDER BY row_id
"""

FOREST_GRID = [{"max_depth": d, "min_samples_leaf": leaf}
               for d in (None, 8, 16) for leaf in (1, 5, 20)]


def load_lines(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Order lines with all allowed features, the target, and order/band for evaluation."""
    lines = con.execute(LINES_SQL).df()
    lines["order_month"] = lines["order_month"].astype(str)
    return lines


# ---------------------------------------------------------------- baselines

class MedianBaseline:
    """Baseline (a): every line gets the training median profit."""

    def fit(self, train: pd.DataFrame) -> "MedianBaseline":
        self.value_ = float(train[TARGET].median())
        return self

    def predict(self, lines: pd.DataFrame) -> np.ndarray:
        return np.full(len(lines), self.value_)


class MarginRuleBaseline:
    """Baseline (b): sales x training margin of the line's discount band and sub-category.

    Margin = SUM(profit) / SUM(sales) in the training lines of that cell; cells never
    seen in training fall back to the margin of the discount band.
    """

    def fit(self, train: pd.DataFrame) -> "MarginRuleBaseline":
        cells = train.groupby(["discount_band", "sub_category"])[["profit", "sales"]].sum()
        bands = train.groupby("discount_band")[["profit", "sales"]].sum()
        self.cell_margin_ = cells["profit"] / cells["sales"]
        self.band_margin_ = bands["profit"] / bands["sales"]
        return self

    def margins(self, lines: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_frame(lines[["discount_band", "sub_category"]])
        cell = pd.Series(self.cell_margin_.reindex(keys).to_numpy(), index=lines.index)
        band = lines["discount_band"].map(self.band_margin_)
        return cell.fillna(band).to_numpy()

    def predict(self, lines: pd.DataFrame) -> np.ndarray:
        return lines["sales"].to_numpy() * self.margins(lines)


# ---------------------------------------------------------------- models

def make_preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    """One-hot the categorical features; scale the numeric ones for the linear model."""
    cats = [f for f in features if f in CATEGORICAL_FEATURES]
    nums = [f for f in features if f in NUMERIC_FEATURES]
    steps = []
    if cats:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats))
    if nums:
        steps.append(("num", StandardScaler() if scale else "passthrough", nums))
    return ColumnTransformer(steps, verbose_feature_names_out=True)


def make_linear(features: list[str]) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(features, scale=True)),
                     ("model", LinearRegression())])


def make_forest(features: list[str], max_depth=None, min_samples_leaf: int = 1,
                n_estimators: int = 300) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(features, scale=False)),
                     ("model", RandomForestRegressor(
                         n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, n_jobs=-1,
                         random_state=RANDOM_STATE))])


class MarginForest:
    """Forest on the line's margin (profit / sales), then predicted profit = margin x sales.

    Same structure as baseline (b), but the margin is learned from the features.
    Training lines are weighted by sales, so the forest focuses on the margins of
    big lines, where a margin error costs the most dollars. If sales is not one of
    the features it is computed from them as list price x quantity x (1 - discount),
    exactly as the dashboard would; the feature set must allow that
    (see sales_derivable).
    """

    def __init__(self, features: list[str], **params):
        if not sales_derivable(features):
            raise ValueError("MarginForest needs sales, or list price + quantity + discount")
        self.features = features
        self.forest = make_forest(features, **params)

    def sales(self, lines: pd.DataFrame) -> np.ndarray:
        if "sales" in self.features:
            return lines["sales"].to_numpy()
        return derived_sales(lines)

    def fit(self, lines: pd.DataFrame) -> "MarginForest":
        sales = self.sales(lines)
        self.forest.fit(lines[self.features], lines[TARGET].to_numpy() / sales,
                        model__sample_weight=sales)
        return self

    def predict_margin(self, lines: pd.DataFrame) -> np.ndarray:
        return self.forest.predict(lines[self.features])

    def predict(self, lines: pd.DataFrame) -> np.ndarray:
        return self.predict_margin(lines) * self.sales(lines)


def derived_sales(lines: pd.DataFrame) -> np.ndarray:
    """Sales from list price x quantity x (1 - discount); equals `sales` in the data
    under the step 1 assumption that sales is recorded after discount."""
    return (lines["unit_price"] * lines["quantity"] * (1 - lines["discount"])).to_numpy()


MODELS = ("linear", "forest", "margin_forest")


def fit_predict(kind: str, features: list[str], params: dict, fit: pd.DataFrame,
                apply: pd.DataFrame) -> np.ndarray:
    """Fit one model kind on `fit` and predict profit for `apply`."""
    if kind == "linear":
        return make_linear(features).fit(fit[features], fit[TARGET]).predict(apply[features])
    if kind == "forest":
        model = make_forest(features, **params).fit(fit[features], fit[TARGET])
        return model.predict(apply[features])
    if kind == "margin_forest":
        return MarginForest(features, **params).fit(fit).predict(apply)
    raise ValueError(kind)


def cv_mae(kind: str, features: list[str], params: dict, train: pd.DataFrame) -> float:
    """Mean MAE over the expanding-window folds inside the training years."""
    maes = []
    for fit_years, val_year in CV_FOLDS:
        fit = train[train["order_year"].isin(fit_years)]
        val = train[train["order_year"] == val_year]
        maes.append(mean_absolute_error(val[TARGET],
                                        fit_predict(kind, features, params, fit, val)))
    return float(np.mean(maes))


def tune_forest(features: list[str], train: pd.DataFrame, kind: str = "forest",
                grid: list[dict] = FOREST_GRID) -> pd.DataFrame:
    """CV MAE of every forest setting for one feature set and model kind; best first."""
    rows = [{**params, "params": params, "cv_mae": cv_mae(kind, features, params, train)}
            for params in grid]
    return pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)


def sales_derivable(features: list[str]) -> bool:
    """True if sales is a feature or can be computed as list price x quantity x (1 - discount)."""
    return "sales" in features or {"unit_price", "quantity", "discount"} <= set(features)


# ---------------------------------------------------------------- feature ranking

def original_feature(column: str) -> str:
    """Map a transformed column name (e.g. 'cat__state_Texas') to its original feature."""
    name = column.split("__", 1)[1]
    for feature in sorted(ALL_FEATURES, key=len, reverse=True):
        if name == feature or name.startswith(feature + "_"):
            return feature
    raise ValueError(f"unknown column {column}")


def grouped_permutation_importance(model: Pipeline, lines: pd.DataFrame, features: list[str],
                                   n_repeats: int = 10) -> pd.DataFrame:
    """MAE increase when each original feature is shuffled (one-hot columns move together)."""
    result = permutation_importance(model, lines[features], lines[TARGET],
                                    scoring="neg_mean_absolute_error", n_repeats=n_repeats,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"feature": features, "mae_increase": result.importances_mean,
                          "std": result.importances_std})
            .sort_values("mae_increase", ascending=False).reset_index(drop=True))


def rank_features(lines: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Rank all allowed features using the training years only.

    A forest with `params` is fit on 2014-2015 and permutation importance is measured
    on 2016 (the last validation fold). 2017 is never read, so choosing the top 4
    from this ranking does not leak the test year.
    """
    train, _ = split_by_year(lines)
    fit_years, val_year = CV_FOLDS[-1]
    fit = train[train["order_year"].isin(fit_years)]
    val = train[train["order_year"] == val_year]
    forest = make_forest(ALL_FEATURES, **params).fit(fit[ALL_FEATURES], fit[TARGET])
    return grouped_permutation_importance(forest, val, ALL_FEATURES)


def explain(forest: Pipeline, lines: pd.DataFrame, features: list[str]) -> tuple[float, pd.DataFrame]:
    """Base value and per-feature SHAP values (dollars) for each line.

    base + the row sum of the per-feature values = the forest's predicted profit.
    One-hot columns are summed back into their original feature.
    """
    import shap

    prep = forest.named_steps["prep"]
    x = pd.DataFrame(prep.transform(lines[features]), columns=prep.get_feature_names_out())
    explainer = shap.TreeExplainer(forest.named_steps["model"])
    values = explainer.shap_values(x, check_additivity=False)
    base = float(np.ravel(explainer.expected_value)[0])
    per_column = pd.DataFrame(values, columns=x.columns, index=lines.index)
    return base, per_column.T.groupby(original_feature).sum().T


def explain_margin_forest(model: "MarginForest", lines: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """SHAP values of a MarginForest in dollars.

    The forest explains the margin: margin = base + sum(phi). Profit = sales x margin,
    so profit = sales x base + sum(sales x phi): each margin contribution times the
    line's sales is that feature's contribution in dollars, and the parts still add
    up to the predicted profit. Returns (sales x base per line, dollar SHAP by feature).
    """
    base, per_feature = explain(model.forest, lines, model.features)
    sales = model.sales(lines)
    return sales * base, per_feature.mul(sales, axis=0)


def shap_ranking(per_feature: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per original feature, largest first."""
    return (per_feature.abs().mean().sort_values(ascending=False)
            .rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}))


def permutation_mae_increase(predict, lines: pd.DataFrame, features: list[str],
                             n_repeats: int = 10) -> pd.DataFrame:
    """MAE increase when each feature is shuffled, for any `predict(lines)` function.

    Used for MarginForest, whose prediction also depends on sales (computed from the
    features), so shuffling e.g. quantity changes both the margin and the multiplier.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y = lines[TARGET].to_numpy()
    base_mae = mean_absolute_error(y, predict(lines))
    rows = []
    for feature in features:
        increases = []
        for _ in range(n_repeats):
            shuffled = lines.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            increases.append(mean_absolute_error(y, predict(shuffled)) - base_mae)
        rows.append({"feature": feature, "mae_increase": np.mean(increases),
                     "std": np.std(increases)})
    return pd.DataFrame(rows).sort_values("mae_increase", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- evaluation

def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """MAE (main), RMSE, R2 and median absolute error."""
    return {"mae": mean_absolute_error(y, pred),
            "rmse": float(np.sqrt(np.mean((np.asarray(y) - pred) ** 2))),
            "r2": r2_score(y, pred),
            "median_ae": median_absolute_error(y, pred)}


def bootstrap_mae_difference(lines: pd.DataFrame, pred_a: np.ndarray, pred_b: np.ndarray,
                             n_boot: int = 1000) -> dict:
    """Paired bootstrap of MAE(a) - MAE(b), resampling whole orders (lines stay together).

    A negative difference means a is better. Both are scored on the same resamples.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    orders = lines["order_id"].to_numpy()
    unique = np.unique(orders)
    index_by_order = pd.Series(np.arange(len(lines))).groupby(orders).apply(np.array)
    err_a = np.abs(lines[TARGET].to_numpy() - pred_a)
    err_b = np.abs(lines[TARGET].to_numpy() - pred_b)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate(index_by_order.loc[rng.choice(unique, len(unique))].to_numpy())
        diffs.append(err_a[idx].mean() - err_b[idx].mean())
    return {"difference": float(err_a.mean() - err_b.mean()),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5))}
