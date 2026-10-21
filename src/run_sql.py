"""Run a .sql file against the project DuckDB database and print the results.

Usage (from the project root):
    python -m src.run_sql sql/01_category_profitability.sql
    python -m src.run_sql sql/01_category_profitability.sql --max-rows 100

The database is opened read-only, so a query can never change the data.
A file may contain several statements separated by semicolons; every statement
that returns rows is printed as its own table.
"""

import argparse
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

from src.build_database import DB_PATH, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    """Read the SQL file path and display options from the command line."""
    parser = argparse.ArgumentParser(description="Run a .sql file against superstore.duckdb.")
    parser.add_argument("sql_file", type=Path, help="path to a .sql file, e.g. sql/01_example.sql")
    parser.add_argument("--max-rows", type=int, default=50,
                        help="maximum rows to print per result (default 50)")
    parser.add_argument("--decimals", type=int, default=2,
                        help="decimal places for numeric columns (default 2)")
    return parser.parse_args()


def format_result(df: pd.DataFrame, max_rows: int, decimals: int) -> str:
    """Render a result as a plain-text table with rounded numbers and a row count."""
    if df.empty:
        return f"{'  '.join(df.columns)}\n(0 rows)"
    shown = df.head(max_rows)
    text = shown.to_string(
        index=False,
        float_format=lambda x: f"{x:,.{decimals}f}",
        na_rep="NULL",
    )
    footer = f"({len(df):,} rows"
    footer += f", showing first {max_rows})" if len(df) > max_rows else ")"
    return f"{text}\n{footer}"


HEADER_LABELS = ("Business question", "Why we checked", "Insight", "Caveats",
                 "Caveat", "Hypothesis", "Next")
HEADER_LABEL_RE = re.compile(rf"^\s*({'|'.join(HEADER_LABELS)}):\s*(.*)$")


def read_header(sql_file: Path) -> dict[str, list[str]]:
    """Return the labelled sections of a .sql file's leading comment block.

    The header is the run of `--` lines at the top of the file, ending at the
    first blank or non-comment line. Each section ("Business question",
    "Insight", ...) maps to its lines with the `--` prefix removed and the
    indentation kept, so bullets and continuation lines can be told apart.
    """
    sections: dict[str, list[str]] = {}
    current = None
    for line in sql_file.read_text(encoding="utf-8").splitlines():
        if not line.startswith("--"):
            break
        text = line[2:]
        match = HEADER_LABEL_RE.match(text)
        if match:
            current = match.group(1)
            # Keep the first line's text at the column where it starts.
            sections[current] = [" " * match.start(2) + match.group(2)]
        elif current:
            sections[current].append(text)
    return sections


INTEGER_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
                 "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT"}


def execute_sql_file(con: duckdb.DuckDBPyConnection, sql_file: Path) -> list[tuple[int, pd.DataFrame]]:
    """Execute every statement in sql_file and return (statement number, result) pairs.

    Statements that return no rows (e.g. SET) are skipped. Raises duckdb.Error
    with the statement number added if a statement fails.
    """
    statements = duckdb.extract_statements(sql_file.read_text(encoding="utf-8"))
    results = []
    for number, statement in enumerate(statements, start=1):
        try:
            result = con.execute(statement)
        except duckdb.Error as error:
            raise type(error)(f"statement {number}: {error}") from error
        if result.description is None:
            continue
        sql_types = {col[0]: str(col[1]) for col in result.description}
        df = result.df()
        # Settings such as SET return an empty "Success" table; skip it.
        if df.empty and list(df.columns) == ["Success"]:
            continue
        # pandas turns HUGEINT (e.g. SUM of integers) into float; restore integers.
        for column, sql_type in sql_types.items():
            if sql_type in INTEGER_TYPES and df[column].dtype.kind == "f":
                df[column] = df[column].astype("Int64")
        results.append((number, df))
    return results


def run_sql_file(sql_file: Path, max_rows: int = 50, decimals: int = 2) -> None:
    """Execute every statement in sql_file and print each result set."""
    if not DB_PATH.exists():
        sys.exit(f"Database not found: {DB_PATH.relative_to(PROJECT_ROOT)}\n"
                 "Build it first with: python -m src.build_database")
    if not sql_file.is_file():
        sys.exit(f"SQL file not found: {sql_file}")

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        try:
            results = execute_sql_file(con, sql_file)
        except duckdb.Error as error:
            sys.exit(f"Error in {sql_file}:\n{error}")

    if not results:
        sys.exit(f"No statements returning rows in {sql_file}")
    for number, df in results:
        if len(results) > 1:
            print(f"\n-- Result of statement {number} --")
        print(format_result(df, max_rows, decimals))


if __name__ == "__main__":
    # Windows consoles default to a legacy code page; force UTF-8 for names like “Résumé”.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    run_sql_file(args.sql_file, args.max_rows, args.decimals)
