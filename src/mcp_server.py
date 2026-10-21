"""MCP server that lets Claude query the project's data (step 6).

Runs locally over stdio; Claude Code starts it from .mcp.json. Tools:
    list_tables, describe_table      the data model, with descriptions from sql/SCHEMA.md
    run_sql                          one read-only SELECT, capped at MAX_ROWS rows
    list_saved_queries,              the user's sql/01-07 queries: business question,
    run_saved_query                  insight from the header, and the live result
    predict_loss_risk                the step 4 model for one order line, with SHAP factors
Resource:
    schema://superstore              the full sql/SCHEMA.md

Safety: every connection is read-only with external access disabled and the
configuration locked, so SQL cannot write, read or write files, attach databases or
install extensions. On top of that, run_sql accepts exactly one SELECT statement.

Run manually (it then waits for an MCP client on stdin):
    python -m src.mcp_server
"""

import contextlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd
from mcp.server.mcpserver import MCPServer

import src.loss_model as lm
from src.build_database import DB_PATH, PROJECT_ROOT, build_database
from src.run_sql import execute_sql_file, read_header

SQL_DIR = PROJECT_ROOT / "sql"
SCHEMA_PATH = SQL_DIR / "SCHEMA.md"
MAX_ROWS = 200

mcp = MCPServer(
    name="superstore",
    instructions=(
        "Query the Superstore Profit Intelligence data (US retail order lines, 2014-2017). "
        "Read schema://superstore or call describe_table before writing SQL. Use only numbers "
        "returned by these tools; label anything else as an inference."
    ),
)


# ---------------------------------------------------------------- database access

def ensure_database() -> Path:
    """Build the database from the raw CSV if missing.

    The build prints a summary; stdout carries the MCP protocol, so it is sent
    to stderr instead.
    """
    if not DB_PATH.exists():
        with contextlib.redirect_stdout(sys.stderr):
            build_database()
    return DB_PATH


def connect() -> duckdb.DuckDBPyConnection:
    """Read-only connection that cannot touch files, attach databases or change settings."""
    return duckdb.connect(str(ensure_database()), read_only=True,
                          config={"enable_external_access": False, "lock_configuration": True})


def to_records(df: pd.DataFrame, limit: int = MAX_ROWS) -> dict:
    """JSON-safe result: columns, up to `limit` rows, and whether rows were cut off."""
    shown = df.head(limit)
    payload = json.loads(shown.to_json(orient="split", date_format="iso", index=False))
    return {"columns": payload["columns"], "rows": payload["data"],
            "row_count": len(df), "truncated": len(df) > limit}


def validate_select(query: str) -> None:
    """Raise ValueError unless the query is exactly one read-only SELECT statement."""
    try:
        statements = duckdb.extract_statements(query)
    except duckdb.Error as error:
        raise ValueError(f"SQL syntax error: {error}") from error
    if len(statements) != 1:
        raise ValueError(f"Send exactly one statement (got {len(statements)}).")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise ValueError(f"Only SELECT queries are allowed (got {statements[0].type.name}).")


# ---------------------------------------------------------------- schema descriptions

def _markdown_rows(block: str) -> list[list[str]]:
    """Cells of every data row in the markdown tables of a text block."""
    rows = []
    for line in block.splitlines():
        if line.startswith("|") and not re.match(r"^\|\s*-", line):
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
            rows.append([c.replace("\\|", "|") for c in cells])
    return rows


def _name(cell: str) -> str:
    """A table or column name from a markdown cell: backticks removed."""
    return cell.replace("`", "")


def _text(cell: str) -> str:
    """A description from a markdown cell: `code` becomes 'code', so names with commas stay clear."""
    return cell.replace("`", "'")


@lru_cache(maxsize=1)
def schema_descriptions() -> tuple[dict, dict]:
    """(table -> description, table -> {column -> description}) parsed from SCHEMA.md."""
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    tables, columns = {}, {}
    quick_guide = text.split("## Quick guide", 1)[1].split("**Joins:**", 1)[0]
    for cells in _markdown_rows(quick_guide)[1:]:
        name = _name(cells[0].split()[0])
        tables[name] = f"one row = {_text(cells[1])}; use for: {_text(cells[3])}"
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        title, _, body = section.partition("\n")
        name = title.strip().strip("`").split()[0].strip("`")
        rows = [r for r in _markdown_rows(body) if len(r) == 3 and r[0] != "Column"]
        if rows:
            columns[name] = {_name(r[0]): _text(r[2]) for r in rows}
    return tables, columns


# ---------------------------------------------------------------- tools

@mcp.tool()
def list_tables() -> list[dict]:
    """List every table and view in the database with its row count and purpose."""
    descriptions, _ = schema_descriptions()
    with connect() as con:
        objects = con.execute("""
            SELECT table_name, table_type FROM information_schema.tables
            WHERE table_schema = 'main' ORDER BY table_type, table_name
        """).fetchall()
        return [{"name": name, "type": "view" if kind == "VIEW" else "table",
                 "rows": con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0],
                 "description": descriptions.get(name, "")}
                for name, kind in objects]


@mcp.tool()
def describe_table(table: str) -> dict:
    """Columns, types and descriptions of one table or view (e.g. 'orders_clean')."""
    _, column_docs = schema_descriptions()
    with connect() as con:
        columns = con.execute("""
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position
        """, [table]).fetchall()
    if not columns:
        raise ValueError(f"Unknown table '{table}'. Call list_tables for the available names.")
    docs = column_docs.get(table, {})
    return {"table": table,
            "columns": [{"name": n, "type": t, "description": docs.get(n, "")} for n, t in columns]}


@mcp.tool()
def run_sql(query: str) -> dict:
    """Run one read-only SELECT query (DuckDB SQL) and return up to 200 rows.

    Write margins as SUM(profit) / SUM(sales) and count orders with
    COUNT(DISTINCT order_id). Dates are TIMESTAMPs.
    """
    validate_select(query)
    with connect() as con:
        df = con.execute(query).df()
    return to_records(df)


@mcp.tool()
def list_saved_queries() -> list[dict]:
    """The project's saved business queries (sql/01-07) with their business questions."""
    out = []
    for path in sorted(SQL_DIR.glob("[0-9][0-9]_*.sql")):
        header = read_header(path)
        question = " ".join(l.strip() for l in header.get("Business question", []))
        out.append({"number": path.stem.split("_")[0], "file": f"sql/{path.name}",
                    "business_question": question})
    return out


@mcp.tool()
def run_saved_query(number: str) -> dict:
    """Run saved query `number` (e.g. '03'): its header (question, insight, caveats) and results."""
    if not str(number).strip().isdigit():
        raise ValueError(f"Query number must be digits, e.g. '03' (got '{number}').")
    matches = sorted(SQL_DIR.glob(f"{int(number):02d}_*.sql"))
    if not matches:
        raise ValueError(f"No saved query {number}. Call list_saved_queries.")
    path = matches[0]
    header = {label: " ".join(l.strip() for l in lines) for label, lines in read_header(path).items()}
    with connect() as con:
        results = [to_records(df) for _, df in execute_sql_file(con, path)]
    return {"file": f"sql/{path.name}", "header": header, "results": results}


@lru_cache(maxsize=1)
def get_model() -> dict:
    """Train the step 4 model once per server process (a few seconds)."""
    with connect() as con:
        return lm.train_final_model(con)


@mcp.tool()
def predict_loss_risk(sub_category: str, state: str, discount: float, quantity: int,
                      unit_price: float | None = None, ship_mode: str | None = None,
                      segment: str | None = None, order_month: int | None = None) -> dict:
    """Predicted probability that one order line loses money, with the two discount rules
    and the top SHAP factors. Discount is a fraction (0.3 = 30%). Optional inputs default
    to the sub-category's median list price and the most common ship mode, segment and month.
    The first call trains the model (a few seconds).
    """
    model = get_model()
    train = model["train"]
    for name, value, column in [("sub_category", sub_category, "sub_category"),
                                ("state", state, "state")]:
        if value not in set(train[column]):
            raise ValueError(f"Unknown {name} '{value}'. Options: {sorted(train[column].unique())}")
    if not 0 <= discount <= 0.8:
        raise ValueError("discount must be a fraction between 0 and 0.8 (e.g. 0.3 for 30%).")

    line = lm.build_line(model, sub_category, state, discount, quantity, unit_price,
                         ship_mode, segment, order_month)
    base, contributions = lm.explain(model["forest"], line, model["explainer"])
    row = contributions.iloc[0]
    risk = base + float(row.sum())
    top = row.reindex(row.abs().sort_values(ascending=False).index).head(4)

    with connect() as con:
        history = con.execute("""
            SELECT COUNT(*) AS lines, AVG(is_loss::INT) AS loss_share,
                   (SELECT COUNT(*) FROM orders_clean WHERE state = ? AND discount = ?) AS state_lines
            FROM orders_clean WHERE sub_category = ? AND state = ? AND discount = ?
        """, [state, discount, sub_category, state, discount]).fetchone()
    warnings = []
    if history[2] == 0:
        warnings.append(f"{state} never sold anything at a {discount:.0%} discount in 2014-2017; "
                        "the model has no direct evidence for this combination.")
    observed = sorted(train["discount"].unique())
    if not any(abs(discount - d) < 1e-9 for d in observed):
        warnings.append(f"{discount:.0%} is not one of the discount levels in the data: {observed}.")

    return {
        "input": line.iloc[0].to_dict(),
        "predicted_loss_risk": round(risk, 4),
        "average_line_risk": round(model["base_rate"], 4),
        "rule_a_discount_above_20pct": bool(discount > lm.HIGH_DISCOUNT),
        "rule_b_flag": bool(model["baseline_b"].flags(line)[0]),
        "top_factors": [{"feature": f, "value": str(line.iloc[0][f]),
                         "contribution": round(float(v), 4)} for f, v in top.items()],
        "history": {"similar_lines_2014_2017": int(history[0]),
                    "share_that_lost_money": None if history[0] == 0 else round(float(history[1]), 4)},
        "warnings": warnings,
        "model": "Random Forest trained on 2014-2016, tested on 2017 (PR-AUC 0.96); "
                 "association, not cause.",
    }


# ---------------------------------------------------------------- resources

@mcp.resource("schema://superstore", name="schema", mime_type="text/markdown",
              description="Every table and column in the database (sql/SCHEMA.md).")
def schema() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run("stdio")
