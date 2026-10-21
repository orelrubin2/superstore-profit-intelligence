"""Load, validate and clean the raw Sample Superstore dataset.

The raw file in data/raw/ is never modified. Every change made here is
recorded in a data-quality log so the cleaning is auditable.

Typical use:
    from src.cleaning import clean_superstore
    df, log = clean_superstore()
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "superstore.csv"

RAW_ENCODING = "cp1252"
EXPECTED_RAW_ROWS = 9_994
DATE_FORMAT = "%m/%d/%Y"
TEXT_COLUMNS = [
    "order_id", "ship_mode", "customer_id", "customer_name", "segment",
    "country", "city", "state", "postal_code", "region", "product_id",
    "category", "sub_category", "product_name",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV with the correct encoding and postal codes as text.

    The file is Windows-1252 (cp1252), not latin-1: it contains cp1252 smart
    quotes (bytes 0x93/0x94) that latin-1 decodes into invisible control
    characters.

    Postal codes are identifiers, not numbers: reading them as integers drops
    the leading zero of New England codes (e.g. 02138 -> 2138).
    """
    df = pd.read_csv(path, encoding=RAW_ENCODING, dtype={"Postal Code": str})
    df["Postal Code"] = df["Postal Code"].str.zfill(5)
    return df


def to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case (e.g. 'Sub-Category' -> 'sub_category')."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def log_cp1252_characters(df: pd.DataFrame, log: list[dict]) -> None:
    """Log rows containing characters that exist in cp1252 but not in latin-1.

    These rows (smart quotes such as “ and ”) are the evidence that the file
    is Windows-1252. They are decoded correctly on read and not changed.
    """
    for col in TEXT_COLUMNS:
        # Characters above U+00FF are the cp1252-only ones (bytes 0x80-0x9F).
        mask = df[col].str.contains(r"[^\x00-\xff]", regex=True)
        if mask.any():
            log.append({
                "check": "encoding_cp1252_characters",
                "column": col,
                "rows_affected": int(mask.sum()),
                "action": "none needed; decoded correctly by reading as cp1252",
            })


def clean_text(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """Replace non-breaking spaces with normal spaces and trim whitespace.

    Without this, the same product name can appear as two different strings
    and split group-by results. Each fix is logged separately per column, plus
    a total of rows changed.
    """
    df = df.copy()
    for col in TEXT_COLUMNS:
        original = df[col]
        fixes = {
            "text_non_breaking_space": (
                original.str.contains("\xa0", regex=False),
                "replaced non-breaking spaces with normal spaces",
            ),
            "text_edge_whitespace": (
                original != original.str.strip(),
                "trimmed leading/trailing whitespace",
            ),
        }
        cleaned = original.str.replace("\xa0", " ", regex=False).str.strip()
        changed = cleaned != original

        for check, (mask, action) in fixes.items():
            if mask.any():
                log.append({
                    "check": check,
                    "column": col,
                    "rows_affected": int(mask.sum()),
                    "action": action,
                })
        if changed.any():
            log.append({
                "check": "text_total_changed",
                "column": col,
                "rows_affected": int(changed.sum()),
                "action": "rows changed by any text fix (a row can match several)",
            })
        df[col] = cleaned
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse order and ship dates; the M/D/YYYY strings have mixed zero-padding."""
    df = df.copy()
    for col in ["order_date", "ship_date"]:
        df[col] = pd.to_datetime(df[col], format=DATE_FORMAT)
    return df


def drop_near_duplicates(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """Drop rows identical to an earlier row in every column except row_id.

    Such rows are almost certainly double entries; the first occurrence is kept
    and every dropped row_id is logged.
    """
    is_dup = df.drop(columns=["row_id"]).duplicated(keep="first")
    dropped_ids = df.loc[is_dup, "row_id"].tolist()
    log.append({
        "check": "near_duplicate_rows",
        "column": "all except row_id",
        "rows_affected": len(dropped_ids),
        "action": f"dropped row_id(s) {dropped_ids}, kept first occurrence",
    })
    return df.loc[~is_dup].reset_index(drop=True)


def log_inconsistent_product_names(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """Find product_ids that map to more than one product_name.

    This is a known quirk of the source data. Names are left unchanged in the
    order lines; dim_product resolves each id to its most frequent name.
    Returns one row per affected (product_id, product_name) pair.
    """
    name_counts = df.groupby("product_id")["product_name"].nunique()
    multi_ids = name_counts[name_counts > 1].index
    cases = (
        df[df["product_id"].isin(multi_ids)]
        .groupby(["product_id", "product_name"], as_index=False)
        .size()
        .rename(columns={"size": "line_count"})
        .sort_values(["product_id", "line_count"], ascending=[True, False])
    )
    log.append({
        "check": "product_id_multiple_names",
        "column": "product_name",
        "rows_affected": len(multi_ids),
        "action": "no change in order lines; dim_product uses most frequent name",
    })
    return cases


def iqr_outlier_flag(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Return True where a value lies outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add date parts, shipping time, margin, loss flag, discount features and outlier flags.

    ASSUMPTION (not a fact from the data): `sales` is recorded AFTER the
    discount, so gross_sales = sales / (1 - discount). The dataset has no
    list-price column to confirm this, so gross_sales, discount_amount and
    unit_price are estimates that inherit this assumption.

    WARNING: is_outlier_sales and is_outlier_profit are for analysis only.
    They are computed from the full dataset's quartiles and from sales/profit
    themselves (profit is the ML target), so they must NOT be used as ML
    features - doing so would leak target information into the model.
    """
    df = df.copy()

    df["order_year"] = df["order_date"].dt.year
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_month"] = df["order_date"].dt.month
    df["order_ym"] = df["order_date"].dt.to_period("M").astype(str)
    df["ship_days"] = (df["ship_date"] - df["order_date"]).dt.days

    df["profit_margin"] = df["profit"] / df["sales"]
    df["is_loss"] = df["profit"] < 0

    df["discount_band"] = pd.cut(
        df["discount"],
        bins=[-0.001, 0.0, 0.20, 0.40, 1.0],
        labels=["0%", "1-20%", "21-40%", ">40%"],
    ).astype(str)

    # Based on the after-discount assumption documented above.
    df["gross_sales"] = df["sales"] / (1 - df["discount"])
    df["discount_amount"] = df["gross_sales"] - df["sales"]
    df["unit_price"] = df["gross_sales"] / df["quantity"]

    # Analysis-only flags; never use as ML features (target leakage).
    df["is_outlier_sales"] = iqr_outlier_flag(df["sales"])
    df["is_outlier_profit"] = iqr_outlier_flag(df["profit"])

    return df


def validate(df: pd.DataFrame) -> None:
    """Raise AssertionError if the cleaned data breaks a basic business rule."""
    assert df.notna().all().all(), "cleaned data contains missing values"
    assert df["row_id"].is_unique, "row_id is not unique"
    assert (df["ship_date"] >= df["order_date"]).all(), "ship_date before order_date"
    assert df["discount"].between(0, 0.8).all(), "discount outside [0, 0.8]"
    assert (df["quantity"] > 0).all(), "non-positive quantity"
    assert (df["sales"] > 0).all(), "non-positive sales"
    assert df["postal_code"].str.fullmatch(r"\d{5}").all(), "postal_code is not 5 digits"
    assert (df["order_year"].between(2014, 2017)).all(), "order date outside 2014-2017"


def clean_superstore(path: Path = RAW_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline.

    Returns the cleaned order-line table (orders_clean) and a data-quality
    log with one row per check.
    """
    log: list[dict] = []

    raw = load_raw(path)
    assert len(raw) == EXPECTED_RAW_ROWS, f"expected {EXPECTED_RAW_ROWS} rows, got {len(raw)}"
    log.append({
        "check": "raw_row_count",
        "column": "all",
        "rows_affected": len(raw),
        "action": "loaded raw file",
    })
    log.append({
        "check": "missing_values",
        "column": "all",
        "rows_affected": int(raw.isna().any(axis=1).sum()),
        "action": "none needed",
    })
    log.append({
        "check": "postal_code_leading_zero",
        "column": "postal_code",
        "rows_affected": int((raw["Postal Code"].str.startswith("0")).sum()),
        "action": "read as text and left-padded to 5 digits",
    })

    df = to_snake_case(raw)
    log_cp1252_characters(df, log)
    df = clean_text(df, log)
    df = parse_dates(df)
    df = drop_near_duplicates(df, log)
    log_inconsistent_product_names(df, log)

    log.append({
        "check": "constant_column",
        "column": "country",
        "rows_affected": len(df),
        "action": "dropped (single value 'United States')",
    })
    df = df.drop(columns=["country"])

    df = add_derived_columns(df)
    validate(df)

    log.append({
        "check": "final_row_count",
        "column": "all",
        "rows_affected": len(df),
        "action": "validation passed",
    })
    return df, pd.DataFrame(log)


if __name__ == "__main__":
    orders_clean, quality_log = clean_superstore()
    print(quality_log.to_string(index=False))
    print(f"\norders_clean: {orders_clean.shape[0]} rows x {orders_clean.shape[1]} columns")
