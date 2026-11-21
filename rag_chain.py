"""RAG chain: Claude tool-calling over deterministic SQL (tables) and version-aware document retrieval."""
from __future__ import annotations

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from chain.charts import build_spec
from chain.memory_store import MemoryStore, answer_text, check_citations, citation_loc, extract_tags, web_citations
from rules.domain_rules import RULES, TableEngine, df_to_md
from rules.file_rules import load_file, save_upload
from rules.version_rules import family_versions, latest_versions, order_hits, version_status
from vectorstore.chroma_store import VectorStore, load_registry, upsert_entry

load_dotenv()
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_STEPS, MAX_SHOW = 8, 100
WEB_TOOL = os.getenv("WEB_SEARCH_TOOL_TYPE", "web_search_20250305")  # Anthropic server-side web search
WEB_ON = ("Web search is ON for this question: you may call web_search for facts that are not in the uploaded files "
          "(e.g. another company's policy) and combine web results with uploaded data in one answer. Uploaded files remain "
          "the only source for VoltKart facts. Cite every web fact inline as a markdown link [Source title](URL).")
WEB_OFF = ("Web search is OFF: you have no web access. Never use outside or web knowledge; if the uploaded files do not "
           "contain a fact, say \"This is not in the provided sources.\"")

if os.getenv("LANGSMITH_TRACING", "").lower() == "true" and not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "false"  # tracing is optional: never fail without a key

SYSTEM = """You are the VoltKart data assistant. Answer ONLY from tool results (and web results when web search is on), never from memory or guesses.
Tools: query_tables (SQLite over the uploaded tables), search_documents (text of uploaded files) and make_chart (inline bar/line/pie chart from a SQL query).
{web_rule}

Rules:
- Any number, count, sum, ranking, join or date difference: use query_tables and let SQL compute it. Never do arithmetic yourself.
- Policy / text facts: use search_documents. Results show status=LATEST or status=OLDER. When versions disagree, answer from the LATEST version. Use an OLDER version only if the fact is absent from the latest one, and then say which file and date it came from. Mention when a newer version changed a value.
- Questions that mix data and policy: do both (query the order, then search the policy) and combine.
- When asked for a chart (bar, line, pie), call make_chart with a SQL query whose first column is the label and the other columns numeric; give a clear title and axis labels with units. Never invent chart values. For monthly series include every month (months with no sales as 0). The chart is displayed automatically: add a one or two sentence summary and cite the chart's tag, e.g. [T2].
- Cite every factual claim immediately with the tag from the tool result, e.g. [T1] or [D2]. Only use tags returned in THIS turn.
- If the sources do not contain the answer, say exactly: "This is not in the provided sources." Do not invent anything.
- Use the conversation history to resolve follow-ups ("that store", "the second one"), but re-query for facts.
- Be concise. Markdown tables are fine. Amounts in INR.

Business rules for tables:
{rules}

Uploaded files:
{files}

Table schema:
{schema}
"""


def missing_keys() -> list[str]:
    need = ["ANTHROPIC_API_KEY"] + (["OPENAI_API_KEY"] if os.getenv("EMBEDDINGS", "local").lower() == "openai" else [])
    return [k for k in need if not os.getenv(k)]


# ---------- retrieval & formatting (module-level so they can be tested without an LLM) ----------
def retrieve(store: VectorStore, entries: list[dict], query: str, scope: str = "documents", k: int = 8) -> list:
    where = {"kind": "doc"} if scope == "documents" else {"kind": "table"} if scope == "tables" else None
    hits = list(store.search(query, k, where))
    latest = latest_versions(entries)
    have = {h.metadata["source_file"] for h in hits}
    for fam in {h.metadata.get("doc_family") for h in hits if h.metadata.get("kind") == "doc"}:
        top = latest.get(fam)
        if top and top["name"] not in have:  # make sure the newest version of this family is always seen
            hits += store.search(query, 4, store.where(source_file=top["name"]))
            have.add(top["name"])
    seen, out = set(), []
    for h in hits:
        m = h.metadata
        key = (m["source_file"], m.get("page"), m.get("section"), m.get("row"), h.page_content[:60])
        if key not in seen:
            seen.add(key)
            out.append(h)
    return order_hits(out, latest)[:14]


def format_hits(hits: list, entries: list[dict], start: int) -> tuple[str, list[dict]]:
    latest, cites, blocks = latest_versions(entries), [], []
    for i, h in enumerate(hits, start):
        m = h.metadata
        c = {"id": f"D{i}", "type": "chunk", "source_file": m["source_file"], "doc_date": m.get("doc_date", ""),
             "page": m.get("page", 0), "section": m.get("section", ""), "sheet": m.get("sheet", ""),
             "table": m.get("table", ""), "row": m.get("row", 0), "status": version_status(m, latest),
             "snippet": h.page_content[:300]}
        cites.append(c)
        blocks.append(f"[{c['id']}] file={c['source_file']} | doc_date={c['doc_date'] or 'n/a'} | status={c['status']} | "
                      f"location={citation_loc(c)}\n{h.page_content[:1200]}")
    head = [f"Versions of '{f}': " + ", ".join(f"{n} ({d})" for n, d in family_versions(entries, f)) + " (newest first)"
            for f in sorted({h.metadata.get("doc_family") for h in hits if h.metadata.get("kind") == "doc"})]
    return "\n".join(head + [""] + blocks if head else blocks), cites


def format_query(res, n: int) -> tuple[str, dict]:
    cite = {"id": f"T{n}", "type": "table", "tables": res.tables, "views": res.views, "sql": res.sql,
            "rows": res.total, "preview": df_to_md(res.df, 20)}
    head = f"[T{n}] tables={', '.join(res.tables)} | rows={res.total}" + (f" (showing first {MAX_SHOW})" if res.total > MAX_SHOW else "")
    return f"{head}\n{df_to_md(res.df, MAX_SHOW)}", cite


class Assistant:
    def __init__(self, store=None, engine=None, memory=None, llm=None):
        self.store = store or VectorStore()
        self.memory = memory or MemoryStore()
        self._llm = llm
        self.engine = engine
        if engine is None:
            self.engine = TableEngine()
            for e in load_registry():  # rebuild SQL tables after a restart from saved uploads
                p = os.path.join(os.getenv("DATA_DIR", "data"), "uploads", e["name"])
                if e.get("tables") and os.path.exists(p):
                    self.engine.add(load_file(p, e["name"]).tables)

    @property
    def llm(self):
        if self._llm is None:
            from langchain_anthropic import ChatAnthropic

            self._llm = ChatAnthropic(model=MODEL, max_tokens=2048)
        return self._llm

    # ----- ingestion -----
    def ingest(self, name: str, data: bytes) -> dict:
        path = save_upload(name, data)
        reg = load_registry()
        known = {e["doc_family"] for e in reg if e["name"] != name and e.get("kind") == "doc"}
        f = load_file(path, name, known)
        old = next((e for e in reg if e["name"] == name), None)
        if old:
            self.engine.remove(old.get("tables", []))
            self.store.delete_file(name)
        if f.tables:
            self.engine.add(f.tables)
        if f.chunks:
            self.store.add(f.chunks)
        entry = {"name": name, "file_type": f.file_type, "kind": f.kind, "doc_date": f.doc_date,
                 "doc_family": f.doc_family, "tables": list(f.tables), "chunks": len(f.chunks),
                 "rows": {t: len(d) for t, d in f.tables.items()}}
        if not f.chunks:
            entry["warning"] = "No text could be extracted (scanned PDF?)."
        upsert_entry(entry)
        return entry

    # ----- question answering -----
    def _tools(self, cites: list, entries: list):
        engine, store = self.engine, self.store

        @tool
        def query_tables(sql: str) -> str:
            """Run ONE read-only SQLite SELECT/WITH query over the uploaded tables and views in the schema. Use it for every count, sum, join, ranking or date calculation."""
            try:
                res = engine.run_sql(sql)
            except Exception as e:
                return f"SQL ERROR: {e}. Fix the query and retry."
            text, cite = format_query(res, sum(c["type"] == "table" for c in cites) + 1)
            cites.append(cite)
            return text

        @tool
        def search_documents(query: str, scope: str = "documents") -> str:
            """Semantic search over uploaded files. scope: 'documents' (PDF/DOCX/TXT, default), 'tables' (row records of CSV/XLSX) or 'all'."""
            hits = retrieve(store, entries, query, scope)
            if not hits:
                return "NO RESULTS"
            text, new = format_hits(hits, entries, sum(c["type"] == "chunk" for c in cites) + 1)
            cites.extend(new)
            return text

        @tool
        def make_chart(sql: str, chart_type: str, title: str, x_label: str, y_label: str) -> str:
            """Draw a bar, line or pie chart inline for the user from a SQL query on the uploaded tables. The query's FIRST column is the category/x axis and the remaining columns are numeric series, e.g. SELECT category, SUM(line_amount) AS revenue FROM revenue_lines ... GROUP BY 1. Values are taken exactly from the query result. Give a clear title and axis labels with units (e.g. 'Revenue (INR)'). For time series include every period."""
            try:
                res = engine.run_sql(sql)
                spec = build_spec(res.df, chart_type, title, x_label, y_label)
            except Exception as e:
                return f"CHART ERROR: {e}. Fix the query or arguments and retry."
            text, cite = format_query(res, sum(c["type"] == "table" for c in cites) + 1)
            cite["chart"] = spec
            cites.append(cite)
            return text + f"\n(A {spec['kind']} chart of this data is shown to the user automatically. Cite its tag.)"

        return {"query_tables": query_tables, "search_documents": search_documents, "make_chart": make_chart}

    def _system(self, entries: list[dict], web: bool) -> str:
        files = "\n".join(f"- {e['name']} ({e['file_type']}" + (f", doc date {e['doc_date']}" if e.get("doc_date") else "") + ")"
                          for e in entries) or "(none)"
        return SYSTEM.format(rules=RULES, files=files, schema=self.engine.schema_text(), web_rule=WEB_ON if web else WEB_OFF)

    def ask(self, question: str, chat_id: int, web: bool = False) -> dict:
        """Answer in the context of ONE chat only. web=False means the web tool is never even offered to the model."""
        ts = datetime.now().isoformat(timespec="seconds")
        entries, cites = load_registry(), []
        tools = self._tools(cites, entries)
        history = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
                   for m in self.memory.history(chat_id) if m["content"].strip()]
        msgs = [SystemMessage(self._system(entries, web)), *history, HumanMessage(question)]
        tool_list = list(tools.values()) + ([{"type": WEB_TOOL, "name": "web_search", "max_uses": 5}] if web else [])
        llm, answer, start = self.llm.bind_tools(tool_list), "", len(msgs)
        for _ in range(MAX_STEPS):
            ai = llm.invoke(msgs)
            msgs.append(ai)
            if not ai.tool_calls:
                if (ai.response_metadata or {}).get("stop_reason") == "pause_turn":
                    continue  # long server-side web search: let the model resume
                answer = answer_text(ai.content).strip()
                break
            for tc in ai.tool_calls:
                t = tools.get(tc["name"])
                out = t.invoke(tc["args"]) if t else "UNKNOWN TOOL"
                msgs.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
        answer = answer or "I could not complete this request within the step limit."
        warnings = check_citations(answer, cites)
        by_id = {c["id"]: c for c in cites}
        shown = [by_id[t] for t in extract_tags(answer) if t in by_id]
        if not shown and cites and "not in the provided sources" not in answer.lower():
            shown = list(cites)  # show what was consulted even if the model forgot to tag
        shown += [c for c in cites if c.get("chart") and c not in shown]  # a chart always carries its citation
        if web:
            shown += web_citations(m.content for m in msgs[start:] if isinstance(m, AIMessage))
        self.memory.add(chat_id, "user", question, ts=ts, web=web)
        self.memory.add(chat_id, "assistant", answer, shown, web=web)
        return {"answer": answer, "citations": shown, "warnings": warnings}
