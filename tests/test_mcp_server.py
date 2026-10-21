"""Tests for the MCP server (src/mcp_server.py).

Skipped when `mcp` is not installed (it lives in requirements-mcp.txt, not in the
dashboard's requirements.txt).

Run from the project root (after `python -m src.build_database`):
    python -m pytest
"""

import asyncio
import json
import sys

import duckdb
import numpy as np
import pytest

pytest.importorskip("mcp")

import src.loss_model as lm  # noqa: E402
import src.mcp_server as server  # noqa: E402
from src.build_database import DB_PATH, PROJECT_ROOT  # noqa: E402
from src.run_sql import execute_sql_file  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def database():
    if not DB_PATH.exists():
        pytest.skip("database not built; run `python -m src.build_database` first")


# ---------- safety ----------

@pytest.mark.parametrize("query", [
    "DROP TABLE orders_clean",
    "DELETE FROM orders_clean",
    "CREATE TABLE t AS SELECT 1",
    "SELECT 1; SELECT 2",
    "COPY orders_clean TO 'leak.csv'",
    "ATTACH 'other.duckdb' AS other",
    "INSTALL httpfs",
    "SET enable_external_access = true",
    "SELECT * FROM read_csv('data/raw/superstore.csv')",
    "SELECT * FROM 'data/raw/superstore.csv'",
    "SELECT * FROM read_text('README.md')",
])
def test_unsafe_sql_is_rejected(query):
    """Writes, file access, extensions, settings and multiple statements all fail."""
    with pytest.raises((ValueError, duckdb.Error)):
        server.run_sql(query)


def test_connection_is_locked():
    """Even bypassing validate_select, the connection itself refuses files and writes."""
    with server.connect() as con:
        with pytest.raises(duckdb.Error):
            con.execute("SELECT * FROM read_csv('data/raw/superstore.csv')")
        with pytest.raises(duckdb.Error):
            con.execute("CREATE TABLE t AS SELECT 1")


# ---------- tools ----------

def test_run_sql_matches_database():
    result = server.run_sql("SELECT COUNT(*) AS n, SUM(profit) AS profit FROM orders_clean")
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        n, profit = con.execute("SELECT COUNT(*), SUM(profit) FROM orders_clean").fetchone()
    assert result["columns"] == ["n", "profit"]
    assert result["rows"][0][0] == n and np.isclose(result["rows"][0][1], profit)


def test_run_sql_caps_rows_and_is_json_safe():
    result = server.run_sql("SELECT row_id, order_date, discount FROM orders_clean")
    assert len(result["rows"]) == server.MAX_ROWS
    assert result["truncated"] and result["row_count"] == 9993
    json.dumps(result)                                  # timestamps etc. serialize
    assert isinstance(result["rows"][0][1], str)       # ISO date string


def test_list_and_describe_tables_cover_everything():
    tables = server.list_tables()
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        names = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()}
    assert {t["name"] for t in tables} == names
    assert all(t["description"] for t in tables)
    for name in names:
        assert all(c["description"] for c in server.describe_table(name)["columns"]), name
    with pytest.raises(ValueError):
        server.describe_table("no_such_table")


def test_saved_queries():
    saved = server.list_saved_queries()
    assert [q["number"] for q in saved] == ["01", "02", "03", "04", "05", "06", "07"]
    assert all(q["business_question"] for q in saved)
    result = server.run_saved_query("3")
    assert "Insight" in result["header"]
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        expected = execute_sql_file(con, PROJECT_ROOT / result["file"])
    assert [r["row_count"] for r in result["results"]] == [len(df) for _, df in expected]
    with pytest.raises(ValueError):
        server.run_saved_query("99")
    with pytest.raises(ValueError):
        server.run_saved_query("abc")


def test_predict_loss_risk_matches_model():
    result = server.predict_loss_risk("Binders", "Texas", 0.8, 3)
    model = server.get_model()
    line = lm.build_line(model, "Binders", "Texas", 0.8, 3)
    expected = model["forest"].predict_proba(line[lm.FEATURES])[0, 1]
    assert np.isclose(result["predicted_loss_risk"], expected, atol=1e-4)
    assert result["rule_a_discount_above_20pct"] is True
    assert len(result["top_factors"]) == 4
    assert result["history"]["similar_lines_2014_2017"] > 0
    json.dumps(result)


def test_predict_loss_risk_warnings_and_errors():
    unseen = server.predict_loss_risk("Binders", "California", 0.5, 3)
    assert any("never sold" in w for w in unseen["warnings"])
    with pytest.raises(ValueError, match="Unknown sub_category"):
        server.predict_loss_risk("Spaceships", "Texas", 0.2, 1)
    with pytest.raises(ValueError):
        server.predict_loss_risk("Binders", "Texas", 30, 1)    # percent instead of fraction


def test_predict_loss_risk_warns_on_unseen_sub_category_discount():
    """Binders were never sold at 30% anywhere, but were sold at 80% (in Texas, among others)."""
    unseen = server.predict_loss_risk("Binders", "Texas", 0.3, 10)
    assert any("Binders were never sold at a 30% discount" in w for w in unseen["warnings"])
    seen = server.predict_loss_risk("Binders", "Texas", 0.8, 10)
    assert not any("Binders were never sold" in w for w in seen["warnings"])


# ---------- protocol ----------

def test_server_over_stdio():
    """Start the server as Claude Code would and call it through the MCP protocol."""
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server"],
                                   cwd=str(PROJECT_ROOT))

    async def session():
        async with Client(params) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            result = await client.call_tool("run_sql", {"query": "SELECT COUNT(*) AS n FROM orders_clean"})
            resource = await client.read_resource("schema://superstore")
            return tools, result, resource

    tools, result, resource = asyncio.run(session())
    assert tools == {"list_tables", "describe_table", "run_sql", "list_saved_queries",
                     "run_saved_query", "predict_loss_risk"}
    assert not result.is_error
    assert "9993" in json.dumps(result.model_dump(mode="json"))
    assert "customer_features" in resource.contents[0].text
