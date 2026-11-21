"""Dynamic file readers: CSV/XLSX/PDF/DOCX/TXT -> chunks with source metadata (+ DataFrames for tables)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rules.version_rules import doc_family, extract_doc_date, match_family

SUPPORTED = {".csv": "csv", ".xlsx": "xlsx", ".pdf": "pdf", ".docx": "docx", ".txt": "txt"}
MAX_CHARS = 1000
LOC_DEFAULTS = {"page": 0, "section": "", "sheet": "", "table": "", "row": 0}
REQUIRED_META = ("source_file", "file_type", "doc_date", "doc_family", "kind", *LOC_DEFAULTS)


def data_dir() -> Path:
    p = Path(os.getenv("DATA_DIR", "data"))
    (p / "uploads").mkdir(parents=True, exist_ok=True)
    return p


def save_upload(name: str, data: bytes) -> Path:
    p = data_dir() / "uploads" / Path(name).name
    p.write_bytes(data)
    return p


@dataclass
class Loaded:
    name: str
    file_type: str
    kind: str  # "doc" | "table"
    doc_date: str
    doc_family: str
    chunks: list = field(default_factory=list)  # [{"text", "meta"}]
    tables: dict = field(default_factory=dict)  # table_name -> DataFrame


# ---------- text helpers ----------
def _is_heading(line: str) -> bool:
    words = line.split()
    if len(line) > 80 or line[0] in "-*•":
        return False
    if line.startswith("#") or line.endswith(":") or (line.isupper() and len(words) <= 10):
        return True
    if re.match(r"^(section|article|part)?\s*\d+(\.\d+)*[.)]?\s+\S", line, re.I) and len(words) <= 10:
        return True
    return len(words) <= 6 and ":" not in line and not line.endswith((".", ",", ";", "?", "!"))


def _sections(text: str) -> list[dict]:
    """Blank-line / heading separated blocks with line ranges. Heading lines stay in the text."""
    out, buf, heading = [], [], ""

    def flush():
        nonlocal buf
        if buf:
            out.append({"heading": heading, "text": "\n".join(t for _, t in buf), "l0": buf[0][0], "l1": buf[-1][0]})
            buf = []

    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            flush()
            continue
        if _is_heading(line):
            flush()
            heading = line.lstrip("#").strip()
        buf.append((i, line))
    flush()
    return out


def _split(text: str, n: int = MAX_CHARS) -> list[str]:
    if len(text) <= n:
        return [text]
    out, cur = [], ""
    for ln in text.split("\n"):
        while len(ln) > n:
            if cur:
                out.append(cur)
                cur = ""
            out.append(ln[:n])
            ln = ln[n:]
        if cur and len(cur) + len(ln) + 1 > n:
            out.append(cur)
            cur = ln
        else:
            cur = f"{cur}\n{ln}" if cur else ln
    if cur:
        out.append(cur)
    return out


def _text_units(text: str, loc: dict, show_lines: bool = False) -> list[tuple]:
    units = []
    for s in _sections(text):
        sec = s["heading"]
        if show_lines:
            rng = f"lines {s['l0']}-{s['l1']}"
            sec = f"{sec} ({rng})" if sec else rng
        for piece in _split(s["text"]):
            body = piece if (not s["heading"] or piece.startswith(s["heading"])) else f"[{s['heading']}]\n{piece}"
            units.append((body, {**loc, "section": sec}))
    return units


# ---------- readers (return (full_text_for_dating, units)) ----------
def _read_txt(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, _text_units(text, {}, show_lines=True)


def _read_pdf(path: Path):
    from pypdf import PdfReader

    full, units = [], []
    for p, page in enumerate(PdfReader(str(path)).pages, 1):
        t = page.extract_text() or ""
        full.append(t)
        units += _text_units(t, {"page": p})
    return "\n".join(full), units


def _table_units(rows: list[list[str]], heading: str, n: int) -> list[tuple]:
    if not rows:
        return []
    header, data = (rows[0], rows[1:]) if len(rows) > 1 else ([], rows)
    lines = []
    for r in data:
        lines.append(" | ".join(f"{h}: {c}" if h else c for h, c in zip(header or [""] * len(r), r)))
    units = []
    for i in range(0, len(lines), 10):
        first = i + (2 if header else 1)
        label = f"[Table {n}" + (f" under {heading}" if heading else "") + "]"
        units.append((label + "\n" + "\n".join(lines[i:i + 10]), {"section": heading, "table": str(n), "row": first}))
    return units


def _read_docx(path: Path):
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = Document(str(path))
    units, full, buf, heading, tcount = [], [], [], "", 0

    def flush():
        nonlocal buf
        if buf:
            text = "\n".join(buf)
            if heading and not text.startswith(heading):
                text = f"[{heading}]\n{text}"
            units.append((text, {"section": heading}))
            buf = []

    try:
        for s in d.sections:
            full += [p.text for p in s.header.paragraphs] + [p.text for p in s.footer.paragraphs]
    except Exception:
        pass
    for el in d.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(el, d)
            t = p.text.strip()
            if not t:
                continue
            full.append(t)
            try:
                style = (p.style.name or "").lower()
            except Exception:
                style = ""
            if style.startswith(("heading", "title")):
                flush()
                heading = t
            buf.append(t)
            if sum(len(x) for x in buf) > MAX_CHARS:
                flush()
        elif tag == "tbl":
            flush()
            tcount += 1
            rows = [[c.text.strip() for c in r.cells] for r in Table(el, d).rows]
            full += [" ".join(r) for r in rows]
            units += _table_units(rows, heading, tcount)
    flush()
    return "\n".join(full), units


# ---------- tabular ----------
def _sanitize(s: str) -> str:
    n = re.sub(r"\W+", "_", str(s).strip()).strip("_").lower() or "col"
    return "t_" + n if n[0].isdigit() else n


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all").copy()
    cols, seen = [], {}
    for c in df.columns:
        n = _sanitize(c)
        k = seen.get(n, 0)
        seen[n] = k + 1
        cols.append(n if k == 0 else f"{n}_{k}")
    df.columns = cols
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            s = df[c].dropna()
            fmt = "%Y-%m-%d" if (s.dt.normalize() == s).all() else "%Y-%m-%d %H:%M:%S"
            df[c] = df[c].dt.strftime(fmt)
    return df.reset_index(drop=True)


def _dtype(s: pd.Series) -> str:
    return "REAL" if pd.api.types.is_float_dtype(s) else "INTEGER" if pd.api.types.is_integer_dtype(s) else "TEXT"


def _read_tabular(path: Path, name: str, ft: str):
    frames = {"": pd.read_csv(path, encoding_errors="replace")} if ft == "csv" else pd.read_excel(path, sheet_name=None)
    stem = _sanitize(Path(name).stem)
    tables, units = {}, []
    for sheet, raw in frames.items():
        df = _normalize(raw)
        if df.empty:
            continue
        tname = stem if len(frames) == 1 else f"{stem}_{_sanitize(sheet)}"
        tables[tname] = df
        sample = ", ".join(f"{c}={df[c].iloc[0]}" for c in df.columns[:8])
        schema = (f"Table {tname} (file {name}" + (f", sheet {sheet}" if sheet else "") + f"): {len(df)} rows. Columns: "
                  + ", ".join(f"{c} ({_dtype(df[c])})" for c in df.columns) + f". Example row: {sample}")
        units.append((schema, {"sheet": sheet, "table": tname, "row": 0, "section": "schema"}))
        cols = list(df.columns)
        for i, vals in enumerate(df.itertuples(index=False, name=None)):
            rec = " | ".join(f"{c}: {v}" for c, v in zip(cols, vals) if not pd.isna(v))
            units.append((f"{tname} row {i + 2}: {rec}", {"sheet": sheet, "table": tname, "row": i + 2}))
    return tables, units, ""


# ---------- public API ----------
def load_file(path, name: str | None = None, known_families=()) -> Loaded:
    path = Path(path)
    name = name or path.name
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext or name}")
    ft = SUPPORTED[ext]
    tables: dict = {}
    if ft in ("csv", "xlsx"):
        kind = "table"
        tables, units, full = _read_tabular(path, name, ft)
    else:
        kind = "doc"
        full, units = {"pdf": _read_pdf, "docx": _read_docx, "txt": _read_txt}[ft](path)
    base = {
        "source_file": name,
        "file_type": ft,
        "kind": kind,
        "doc_date": extract_doc_date(name, full),
        "doc_family": match_family(doc_family(name), known_families),
    }
    chunks = [{"text": t, "meta": {**base, **LOC_DEFAULTS, **loc}} for t, loc in units if t.strip()]
    return Loaded(name, ft, kind, base["doc_date"], base["doc_family"], chunks, tables)


def check_chunk_metadata(chunks) -> list[str]:
    """Lightweight check: every chunk has all source/location keys with Chroma-safe scalar values."""
    problems = []
    for i, c in enumerate(chunks):
        m = c["meta"]
        miss = [k for k in REQUIRED_META if k not in m]
        if miss:
            problems.append(f"chunk {i} missing {miss}")
        bad = [k for k, v in m.items() if not isinstance(v, (str, int, float, bool))]
        if bad:
            problems.append(f"chunk {i} non-scalar metadata {bad}")
        if not miss and m["kind"] == "doc" and not (m["page"] or m["section"] or m["table"]):
            problems.append(f"chunk {i} has no location")
    return problems
