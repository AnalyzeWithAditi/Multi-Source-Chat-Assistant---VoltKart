"""Deterministic table math: tables live in in-memory SQLite; the LLM only writes SQL, SQLite computes."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import pandas as pd

RULES = """Table business rules (apply when the columns exist):
- line_amount = quantity * unit_price_inr * (1 - discount_pct/100). View order_lines = order_items joined to orders, with line_amount and the order columns (customer_id, store_id, order_date, channel, order_status, ...).
- Revenue / sales / spend = SUM(line_amount) over Delivered orders only; Cancelled orders are never revenue. View revenue_lines = order_lines filtered to Delivered: use it for revenue, spend and units-sold questions.
- unit_price_inr is the price actually charged; list_price_inr is only the catalogue price.
- Money is INR; dates are ISO YYYY-MM-DD text (use substr / strftime / julianday). Q1 = Jan-Mar.
- Join keys: store_id, customer_id, product_id, order_id, order_item_id. Always join and aggregate in SQL."""

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex)\b", re.I)
LINE_EXPR = "oi.quantity * oi.unit_price_inr * (1 - oi.discount_pct / 100.0)"


@dataclass
class QueryResult:
    df: pd.DataFrame
    sql: str
    tables: list
    views: list
    total: int


def _find(tables: dict, need: set):
    return next((n for n, df in tables.items() if need <= set(df.columns)), None)


def _make_views(conn, tables: dict) -> dict:
    views = {}
    oi = _find(tables, {"order_id", "quantity", "unit_price_inr", "discount_pct"})
    if not oi or "order_lines" in tables:
        return views
    o = _find({n: d for n, d in tables.items() if n != oi}, {"order_id", "order_status"})
    extra = join = ""
    if o:
        extra = "".join(f', o."{c}"' for c in tables[o].columns if c not in tables[oi].columns)
        join = f'LEFT JOIN "{o}" o ON o.order_id = oi.order_id'
    conn.execute(f'CREATE VIEW order_lines AS SELECT oi.*, {LINE_EXPR} AS line_amount{extra} FROM "{oi}" oi {join}')
    views["order_lines"] = [oi] + ([o] if o else [])
    if o:
        conn.execute("CREATE VIEW revenue_lines AS SELECT * FROM order_lines WHERE lower(trim(order_status)) = 'delivered'")
        views["revenue_lines"] = list(views["order_lines"])
    return views


def _t(s: pd.Series) -> str:
    return "REAL" if pd.api.types.is_float_dtype(s) else "INTEGER" if pd.api.types.is_integer_dtype(s) else "TEXT"


def df_to_md(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "(no rows)"

    def f(v):
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            return ("%.2f" % v).rstrip("0").rstrip(".")
        return str(v).replace("|", "/")

    d = df.head(max_rows)
    out = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    out += ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False, name=None)]
    return "\n".join(out)


class TableEngine:
    def __init__(self):
        self.tables: dict[str, pd.DataFrame] = {}
        self.views: dict[str, list] = {}
        self.conn = None
        self._rebuild()

    def _rebuild(self):
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        for n, df in self.tables.items():
            df.to_sql(n, conn, index=False)
        self.views = _make_views(conn, self.tables)
        conn.execute("PRAGMA query_only = ON")
        self.conn = conn

    def add(self, tables: dict):
        self.tables.update(tables)
        self._rebuild()

    def remove(self, names):
        for n in names:
            self.tables.pop(n, None)
        self._rebuild()

    def schema_text(self) -> str:
        if not self.tables:
            return "(no tables uploaded)"
        lines = []
        for n, df in self.tables.items():
            cols = ", ".join(f"{c} {_t(df[c])}" for c in df.columns)
            lines.append(f"- {n} ({len(df)} rows): {cols}")
            vals = [f"{c} in {{{', '.join(map(str, df[c].dropna().unique()))}}}" for c in df.columns
                    if df[c].dtype == object and 1 <= df[c].nunique() <= 10]
            if vals:
                lines.append("    categorical values: " + "; ".join(vals))
            lines.append("    example: " + ", ".join(f"{c}={df[c].iloc[0]}" for c in df.columns)[:300])
        for v, src in self.views.items():
            lines.append(f"- VIEW {v} (from {', '.join(src)}): see business rules; columns include line_amount")
        return "\n".join(lines)

    def run_sql(self, sql: str) -> QueryResult:
        s = sql.strip().rstrip(";").strip()
        if not re.match(r"(?is)^(select|with)\b", s) or ";" in s or FORBIDDEN.search(s):
            raise ValueError("Only a single read-only SELECT/WITH statement is allowed.")
        df = pd.read_sql_query(s, self.conn)
        used_views = [v for v in self.views if re.search(rf"\b{re.escape(v)}\b", s, re.I)]
        base = {n for n in self.tables if re.search(rf"\b{re.escape(n)}\b", s, re.I)}
        for v in used_views:
            base.update(self.views[v])
        return QueryResult(df, s, sorted(base), used_views, len(df))


def check_revenue_rule(engine: TableEngine) -> list[str]:
    """Verify the revenue view against an independent pandas computation (Delivered only, discounted)."""
    if "revenue_lines" not in engine.views:
        return []
    oi, o = engine.views["order_lines"]
    m = engine.tables[oi].merge(engine.tables[o][["order_id", "order_status"]], on="order_id", how="left")
    m = m[m["order_status"].str.strip().str.lower() == "delivered"]
    expected = float((m["quantity"] * m["unit_price_inr"] * (1 - m["discount_pct"] / 100)).sum())
    got = float(engine.run_sql("SELECT COALESCE(SUM(line_amount), 0) AS r FROM revenue_lines").df["r"][0])
    problems = [] if abs(expected - got) < 1e-6 else [f"revenue mismatch: sql={got} pandas={expected}"]
    if engine.run_sql("SELECT COUNT(*) AS c FROM revenue_lines WHERE lower(order_status) <> 'delivered'").df["c"][0]:
        problems.append("revenue_lines contains non-delivered orders")
    return problems
