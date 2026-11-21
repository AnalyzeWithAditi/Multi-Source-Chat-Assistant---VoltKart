"""Persistent multi-chat memory (SQLite) + citation helpers shared by the UI, chain and PDF export."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from rules.file_rules import data_dir

TAG_RE = re.compile(r"\[([DT]\d+(?:\s*[,;]\s*[DT]\d+)*)\]")
DEFAULT_TITLE = "New chat"


# ---------- citation helpers ----------
def extract_tags(text: str) -> list[str]:
    seen = []
    for grp in TAG_RE.findall(text or ""):
        for t in re.split(r"\s*[,;]\s*", grp):
            if t not in seen:
                seen.append(t)
    return seen


def strip_tags(text: str) -> str:
    return re.sub(r"\s?" + TAG_RE.pattern, "", text or "")


def citation_loc(c: dict) -> str:
    parts = [f"page {c['page']}" if c.get("page") else "", f"section {c['section']}" if c.get("section") else "",
             f"sheet {c['sheet']}" if c.get("sheet") else "", f"table {c['table']}" if c.get("table") else "",
             f"row {c['row']}" if c.get("row") else ""]
    return ", ".join(p for p in parts if p) or "n/a"


def citation_label(c: dict) -> str:
    if c["type"] == "web":
        return f"[{c['id']}] Web: {c['title']} | {c['url']}"
    if c["type"] == "table":
        via = f" (via {', '.join(c['views'])})" if c.get("views") else ""
        chart = f" | chart: {c['chart']['title']}" if c.get("chart") else ""
        return f"[{c['id']}] Tables: {', '.join(c['tables'])}{via} | {c['rows']} row(s){chart} | query below"
    return (f"[{c['id']}] {c['source_file']} | doc date {c.get('doc_date') or 'n/a'} | "
            f"{citation_loc(c)} | {c.get('status', '')}")


def check_citations(answer: str, cites: list[dict]) -> list[str]:
    """Lightweight check: tags must exist; consulted sources should be cited."""
    ids, tags, warns = {c["id"] for c in cites}, extract_tags(answer), []
    unknown = [t for t in tags if t not in ids]
    if unknown:
        warns.append(f"Answer cites unknown source tags: {unknown}")
    if cites and not tags and "not in the provided sources" not in answer.lower():
        warns.append("Sources were consulted but the answer has no inline citations.")
    return warns


def answer_text(content) -> str:
    """Final answer text from an Anthropic message; text before the last web-search result is search chatter."""
    if isinstance(content, str):
        return content
    blocks = [b for b in content if isinstance(b, dict)]
    last = max((i for i, b in enumerate(blocks) if b.get("type") == "web_search_tool_result"), default=-1)
    text = "".join(b.get("text", "") for b in blocks[last + 1:] if b.get("type") == "text")
    return text or "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def web_citations(contents) -> list[dict]:
    """Web sources (title + URL) from the content blocks of this turn's model messages.
    Prefers sources the model actually cited; falls back to the top search results."""
    cited, found = {}, {}
    for content in contents:
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "web_search_tool_result" and isinstance(b.get("content"), list):
                for r in b["content"]:
                    if isinstance(r, dict) and r.get("url"):
                        found.setdefault(r["url"], r.get("title") or r["url"])
            for c in b.get("citations") or []:
                if isinstance(c, dict) and c.get("url"):
                    cited.setdefault(c["url"], {"title": c.get("title") or c["url"], "snippet": (c.get("cited_text") or "")[:200]})
    pick = cited or {u: {"title": t, "snippet": ""} for u, t in list(found.items())[:3]}
    return [{"id": f"W{i}", "type": "web", "title": v["title"], "url": u, "snippet": v["snippet"]}
            for i, (u, v) in enumerate(pick.items(), 1)]


# ---------- storage ----------
class MemoryStore:
    def __init__(self, path=None):
        self.path = str(path or data_dir() / "chat.db")
        with self._c() as c:
            c.execute("CREATE TABLE IF NOT EXISTS chats(id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, created_at TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, "
                      "ts TEXT, role TEXT, content TEXT, citations TEXT, web INTEGER DEFAULT 0)")
            cols = {r[1] for r in c.execute("PRAGMA table_info(messages)")}  # migrate single-chat databases
            if "chat_id" not in cols:
                c.execute("ALTER TABLE messages ADD COLUMN chat_id INTEGER")
            if "web" not in cols:
                c.execute("ALTER TABLE messages ADD COLUMN web INTEGER DEFAULT 0")
            if c.execute("SELECT 1 FROM messages WHERE chat_id IS NULL LIMIT 1").fetchone():
                cur = c.execute("INSERT INTO chats(title, created_at) VALUES (?, ?)", ("Earlier chat", self._now()))
                c.execute("UPDATE messages SET chat_id = ? WHERE chat_id IS NULL", (cur.lastrowid,))

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @contextmanager
    def _c(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ----- chats -----
    def new_chat(self, title: str = DEFAULT_TITLE) -> int:
        with self._c() as c:
            return c.execute("INSERT INTO chats(title, created_at) VALUES (?, ?)", (title, self._now())).lastrowid

    def list_chats(self) -> list[dict]:
        with self._c() as c:
            rows = c.execute("SELECT id, title, created_at FROM chats ORDER BY id DESC").fetchall()
        return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

    def delete_chat(self, chat_id: int):
        with self._c() as c:
            c.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    # ----- messages (always scoped to one chat: no cross-chat leakage) -----
    def add(self, chat_id: int, role: str, content: str, citations=None, ts: str | None = None, web: bool = False):
        with self._c() as c:
            c.execute("INSERT INTO messages(chat_id, ts, role, content, citations, web) VALUES (?,?,?,?,?,?)",
                      (chat_id, ts or self._now(), role, content, json.dumps(citations or []), int(bool(web))))
            if role == "user":  # first question names the chat
                title = content.strip().replace("\n", " ")
                title = title[:40] + ("..." if len(title) > 40 else "")
                c.execute("UPDATE chats SET title = ? WHERE id = ? AND title = ?", (title, chat_id, DEFAULT_TITLE))

    def all(self, chat_id: int) -> list[dict]:
        with self._c() as c:
            rows = c.execute("SELECT ts, role, content, citations, web FROM messages WHERE chat_id = ? ORDER BY id",
                             (chat_id,)).fetchall()
        return [{"ts": r[0], "role": r[1], "content": r[2], "citations": json.loads(r[3]), "web": bool(r[4])} for r in rows]

    def history(self, chat_id: int, n: int = 12) -> list[dict]:
        """Last n messages of THIS chat for the LLM. Citation tags are stripped (they only mean something in
        their own turn); a compact note of the sources behind each answer keeps follow-ups resolvable."""
        out = []
        for m in self.all(chat_id)[-n:]:
            text = strip_tags(m["content"])
            notes = []
            for c in m["citations"][:5]:
                if c["type"] == "table":
                    notes.append("SQL: " + c["sql"][:300])
                elif c["type"] == "web":
                    notes.append(f"web: {c['title']}")
                else:
                    notes.append(f"{c['source_file']} ({c.get('doc_date') or 'n/a'})")
            if notes:
                text += "\n\n[Sources behind this answer: " + " ; ".join(notes) + "]"
            out.append({"role": m["role"], "content": text})
        return out

    def clear(self, chat_id: int):
        with self._c() as c:
            c.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
