"""Step 4: predict loss-making order lines from features known at order time.

Business question: "Which features known at order time (e.g. product category,
region, ship mode, discount level) are the strongest predictors of loss-making
order lines, and can we flag high-risk transactions before fulfillment better
than a simple discount rule?"

Design (agreed in the step 4 plan):
    - One row per order line from orders_clean ONLY. Profit, margin, outlier
      flags and every step 3 customer table (customer_features, customer_mix,
      customer_rfm, customer_clusters) are excluded: they define the target or
      summarize a customer's full history, including future orders.
    - Train on 2014-2016, test on 2017. The test year is used once, for the
      final evaluation. Hyperparameters are tuned on expanding windows inside
      the training years (2014 -> 2015, 2014-15 -> 2016).
    - Baseline A: flag if discount > 20% (step 2).
      Baseline B: Baseline A, plus lines discounted 1-20% whose sub-category
      had a high loss rate in that band in the training years.
    - Logistic Regression and Random Forest on the same features.
    - Primary metric: PR-AUC on 2017. Business comparison at the same number
      of flagged lines as Baseline A. Everything is also reported for the
      1-20% discount band, where Baseline A cannot tell lines apart.
    - SHAP (Random Forest) ranks predictors; one-hot columns are summed back
      into their original feature so that high-cardinality features such as
      state are counted once.
"""

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "is_loss"
CATEGORICAL_FEATURES = ["category", "sub_category", "region", "state", "ship_mode",
                        "segment", "order_month"]
NUMERIC_FEATURES = ["discount", "quantity", "unit_price"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Never allowed as features: they define the target, leak the label, or (step 3
# customer tables) summarize future orders.
FORBIDDEN_FEATURES = {
    "profit", "profit_margin", "is_loss", "sales", "is_outlier_sales", "is_outlier_profit",
    "recency_days", "frequency", "tenure_days", "avg_discount", "high_discount_line_share",
    "high_discount_gross_share", "worst_line_profit", "worst_line_share_of_losses",
    "is_high_volume", "is_net_loss", "volume_group", "r_score", "f_score", "m_score",
    "rfm_segment", "cluster", "cluster_contrast",
}

TRAIN_YEARS = [2014, 2015, 2016]
TEST_YEAR = 2017
CV_FOLDS = [([2014], 2015), ([2014, 2015], 2016)]   # expanding windows inside training years
HIGH_DISCOUNT = 0.20
RANDOM_STATE = 42

# Chosen in notebooks/04_loss_risk_model.ipynb with tune() and tune_baseline_b() on the
# expanding-window folds (training years only). The dashboard reuses them so it runs
# exactly the model whose 2017 test results the notebook reports.
TUNED_FOREST_PARAMS = {"max_depth": 16, "min_samples_leaf": 1, "class_weight": "balanced_subsample"}
TUNED_B_THRESHOLD = 0.5

LINES_SQL = f"""
    SELECT row_id, order_id, order_year, sales, profit, {', '.join(FEATURES)}, {TARGET}
    FROM orders_clean
    ORDER BY row_id
"""


def load_lines(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Order lines with model features, target, and sales/profit for evaluation only."""
    lines = con.execute(LINES_SQL).df()
    lines["order_month"] = lines["order_month"].astype(str)
    lines[TARGET] = lines[TARGET].astype(int)
    return lines


def split_by_year(lines: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = 2014-2016, test = 2017."""
    train = lines[lines["order_year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test = lines[lines["order_year"] == TEST_YEAR].reset_index(drop=True)
    return train, test


def in_mid_band(lines: pd.DataFrame) -> pd.Series:
    """Lines discounted 1-20%: the band where Baseline A flags nothing."""
    return (lines["discount"] > 0) & (lines["discount"] <= HIGH_DISCOUNT)


# ---------------------------------------------------------------- baselines

def baseline_a_scores(lines: pd.DataFrame) -> np.ndarray:
    """Baseline A: 1 if discount > 20%, else 0."""
    return (lines["discount"] > HIGH_DISCOUNT).astype(float).to_numpy()


class BaselineB:
    """Baseline A plus sub-category loss rates in the 1-20% band, learned from training data.

    Score: 1 for discount > 20%; the training loss rate of the line's
    sub-category within the 1-20% band for mid-band lines; 0 for undiscounted
    lines. A line is flagged if its score is at least `threshold`.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, train: pd.DataFrame) -> "BaselineB":
        mid = train[in_mid_band(train)]
        self.rates_ = mid.groupby("sub_category")[TARGET].mean()
        self.default_rate_ = mid[TARGET].mean()
        return self

    def scores(self, lines: pd.DataFrame) -> np.ndarray:
        rates = lines["sub_category"].map(self.rates_).fillna(self.default_rate_)
        score = np.where(lines["discount"] > HIGH_DISCOUNT, 1.0,
                         np.where(in_mid_band(lines), rates, 0.0))
        return score.astype(float)

    def flags(self, lines: pd.DataFrame) -> np.ndarray:
        return self.scores(lines) >= self.threshold


def tune_baseline_b(train: pd.DataFrame, thresholds=np.arange(0.1, 0.95, 0.05)) -> float:
    """Pick the Baseline B threshold with the best mean F1 on the expanding-window folds."""
    best, best_f1 = None, -1.0
    for threshold in thresholds:
        f1s = []
        for fit_years, val_year in CV_FOLDS:
            fit = train[train["order_year"].isin(fit_years)]
            val = train[train["order_year"] == val_year]
            flags = BaselineB(threshold).fit(fit).flags(val)
            tp = (flags & (val[TARGET] == 1)).sum()
            precision = tp / max(flags.sum(), 1)
            recall = tp / max((val[TARGET] == 1).sum(), 1)
            f1s.append(0 if tp == 0 else 2 * precision * recall / (precision + recall))
        if np.mean(f1s) > best_f1:
            best, best_f1 = float(threshold), float(np.mean(f1s))
    return best


# ---------------------------------------------------------------- models

def _preprocessor(scale: bool) -> ColumnTransformer:
    """One-hot categoricals; log1p on quantity and unit price; optional scaling."""
    log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    numeric_steps = [("log", ColumnTransformer([("log", log, ["quantity", "unit_price"])],
                                               remainder="passthrough",
                                               verbose_feature_names_out=False))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
    ], verbose_feature_names_out=True)


def make_logistic(c: float = 1.0) -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor(scale=True)),
        ("model", LogisticRegression(C=c, max_iter=5000)),
    ])


def make_forest(max_depth=None, min_samples_leaf: int = 1, class_weight=None,
                n_estimators: int = 300) -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor(scale=False)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            class_weight=class_weight, n_jobs=-1, random_state=RANDOM_STATE)),
    ])


LOGISTIC_GRID = [{"c": c} for c in (0.01, 0.1, 1.0, 10.0)]
FOREST_GRID = [{"max_depth": d, "min_samples_leaf": leaf, "class_weight": w}
               for d in (None, 8, 16) for leaf in (1, 5, 20) for w in (None, "balanced_subsample")]


def cv_score(make_model, params: dict, train: pd.DataFrame) -> float:
    """Mean PR-AUC over the expanding-window folds."""
    scores = []
    for fit_years, val_year in CV_FOLDS:
        fit = train[train["order_year"].isin(fit_years)]
        val = train[train["order_year"] == val_year]
        model = make_model(**params).fit(fit[FEATURES], fit[TARGET])
        scores.append(average_precision_score(val[TARGET], model.predict_proba(val[FEATURES])[:, 1]))
    return float(np.mean(scores))


def tune(make_model, grid: list[dict], train: pd.DataFrame) -> pd.DataFrame:
    """Score every parameter set on the folds; best first.

    The `params` column keeps each original dict, so the best one can be reused
    without pandas turning ints into floats or None into NaN.
    """
    rows = [{**params, "params": params, "cv_pr_auc": cv_score(make_model, params, train)}
            for params in grid]
    return pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)


def best_params(results: pd.DataFrame) -> dict:
    """The parameter dict of the best-scoring row."""
    return dict(results.loc[0, "params"])


# ---------------------------------------------------------------- evaluation

def flag_top(scores: np.ndarray, n: int) -> np.ndarray:
    """Flag exactly the n highest-scoring lines (ties broken by position, stable)."""
    order = np.argsort(-scores, kind="stable")
    flags = np.zeros(len(scores), dtype=bool)
    flags[order[:n]] = True
    return flags


def operating_point(lines: pd.DataFrame, flags: np.ndarray) -> dict:
    """Precision, recall and share of loss dollars caught by a set of flags."""
    y = lines[TARGET].to_numpy() == 1
    loss_dollars = -lines["profit"].clip(upper=0).to_numpy()
    return {
        "flagged": int(flags.sum()),
        "precision": (flags & y).sum() / max(flags.sum(), 1),
        "recall": (flags & y).sum() / max(y.sum(), 1),
        "loss_dollars_caught": loss_dollars[flags].sum() / max(loss_dollars.sum(), 1e-9),
    }


def evaluate(lines: pd.DataFrame, scores: dict[str, np.ndarray], n_flag: int) -> pd.DataFrame:
    """PR-AUC, ROC-AUC and the operating point (top n_flag lines) for every scorer."""
    y = lines[TARGET].to_numpy()
    rows = []
    for name, s in scores.items():
        row = {"model": name,
               "pr_auc": average_precision_score(y, s),
               "roc_auc": roc_auc_score(y, s) if len(set(y)) > 1 else np.nan}
        row.update(operating_point(lines, flag_top(s, n_flag)))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_pr_auc(lines: pd.DataFrame, scores: dict[str, np.ndarray],
                     n_boot: int = 1000) -> pd.DataFrame:
    """95% intervals for PR-AUC, resampling whole orders (lines of an order stay together)."""
    rng = np.random.default_rng(RANDOM_STATE)
    orders = lines["order_id"].to_numpy()
    unique = np.unique(orders)
    index_by_order = pd.Series(np.arange(len(lines))).groupby(orders).apply(np.array)
    y = lines[TARGET].to_numpy()
    samples = {name: [] for name in scores}
    for _ in range(n_boot):
        picked = rng.choice(unique, len(unique))
        idx = np.concatenate(index_by_order.loc[picked].to_numpy())
        for name, s in scores.items():
            samples[name].append(average_precision_score(y[idx], s[idx]))
    return pd.DataFrame([{"model": name, "ci_low": np.percentile(v, 2.5),
                          "ci_high": np.percentile(v, 97.5)} for name, v in samples.items()])


def bootstrap_pr_auc_difference(lines: pd.DataFrame, scores_a: np.ndarray, scores_b: np.ndarray,
                                n_boot: int = 1000) -> dict:
    """Paired bootstrap of PR-AUC(a) - PR-AUC(b): both scored on the same resampled orders."""
    rng = np.random.default_rng(RANDOM_STATE)
    orders = lines["order_id"].to_numpy()
    unique = np.unique(orders)
    index_by_order = pd.Series(np.arange(len(lines))).groupby(orders).apply(np.array)
    y = lines[TARGET].to_numpy()
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate(index_by_order.loc[rng.choice(unique, len(unique))].to_numpy())
        diffs.append(average_precision_score(y[idx], scores_a[idx])
                     - average_precision_score(y[idx], scores_b[idx]))
    return {"difference": average_precision_score(y, scores_a) - average_precision_score(y, scores_b),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "share_of_resamples_above_0": float(np.mean(np.array(diffs) > 0))}


def brier(lines: pd.DataFrame, probabilities: np.ndarray) -> float:
    return brier_score_loss(lines[TARGET], probabilities)


# ---------------------------------------------------------------- importance

def original_feature(column: str) -> str:
    """Map a transformed column name (e.g. 'cat__state_Texas') to its original feature."""
    name = column.split("__", 1)[1]
    for feature in sorted(FEATURES, key=len, reverse=True):
        if name == feature or name.startswith(feature + "_"):
            return feature
    raise ValueError(f"unknown column {column}")


def make_explainer(forest: Pipeline):
    """SHAP TreeExplainer for the forest inside the pipeline."""
    import shap

    return shap.TreeExplainer(forest.named_steps["model"])


def explain(forest: Pipeline, lines: pd.DataFrame, explainer=None) -> tuple[float, pd.DataFrame]:
    """Base value and per-feature SHAP values of the loss class for each line.

    base + the row sum of the per-feature values = the predicted loss probability.
    One-hot columns are summed back into their original feature.
    """
    explainer = explainer or make_explainer(forest)
    prep = forest.named_steps["prep"]
    x = pd.DataFrame(prep.transform(lines[FEATURES]), columns=prep.get_feature_names_out())
    values = explainer.shap_values(x, check_additivity=False)
    values = values[:, :, 1] if np.ndim(values) == 3 else values[1]
    base = explainer.expected_value
    base = float(base[1] if np.ndim(base) else base)
    per_column = pd.DataFrame(values, columns=x.columns, index=lines.index)
    return base, per_column.T.groupby(original_feature).sum().T


def grouped_shap(forest: Pipeline, lines: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP values of the loss class, summed per original feature.

    Returns (per-line SHAP by feature, ranking by mean |SHAP|). Summing one-hot
    columns first means a feature like state is judged as one feature, not 49.
    """
    _, per_feature = explain(forest, lines)
    ranking = (per_feature.abs().mean().sort_values(ascending=False)
               .rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}))
    return per_feature, ranking


def grouped_permutation_importance(model: Pipeline, lines: pd.DataFrame,
                                   n_repeats: int = 10) -> pd.DataFrame:
    """Drop in PR-AUC when each original feature is shuffled (one-hot columns move together)."""
    result = permutation_importance(model, lines[FEATURES], lines[TARGET],
                                    scoring="average_precision", n_repeats=n_repeats,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"feature": FEATURES, "pr_auc_drop": result.importances_mean,
                          "std": result.importances_std})
            .sort_values("pr_auc_drop", ascending=False).reset_index(drop=True))


def logistic_coefficients(logistic: Pipeline) -> pd.DataFrame:
    """Standardized coefficients of the Logistic Regression, largest effect first."""
    names = logistic.named_steps["prep"].get_feature_names_out()
    coef = logistic.named_steps["model"].coef_[0]
    out = pd.DataFrame({"column": names, "feature": [original_feature(n) for n in names],
                        "coefficient": coef})
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
