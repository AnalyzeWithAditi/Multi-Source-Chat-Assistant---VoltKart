"""Document dating, family grouping and latest-version resolution (no heavy deps)."""
import re
from datetime import date, datetime
from difflib import get_close_matches
from pathlib import Path

LABEL = re.compile(r"(?:document\s+date|effective\s+date)\s*[:\-\u2013\u2014]\s*([^\n\r|]{4,40})", re.I)
FILE_DATE = re.compile(r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})(?!\d)|(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)")
FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%d/%m/%Y", "%d-%m-%Y")


def _parse(s: str) -> str:
    s = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", s.strip(), flags=re.I)
    words = s.replace(",", " ").split()
    for n in range(min(len(words), 4), 0, -1):
        cand = " ".join(words[:n]).rstrip(".")
        for f in FORMATS:
            try:
                return datetime.strptime(cand, f).date().isoformat()
            except ValueError:
                pass
    return ""


def _filename_date(name: str) -> str:
    m = FILE_DATE.search(Path(name).stem)
    if not m:
        return ""
    y, mo, d = (m.group(1), m.group(2), m.group(3)) if m.group(1) else (m.group(4), m.group(5), m.group(6))
    try:
        return date(int(y), int(mo), int(d)).isoformat()
    except ValueError:
        return ""


def extract_doc_date(filename: str, text: str = "") -> str:
    """'Document Date' / 'Effective Date' in the body wins; falls back to the file name; '' if unknown."""
    for m in LABEL.finditer(text or ""):
        d = _parse(m.group(1))
        if d:
            return d
    return _filename_date(filename)


def doc_family(filename: str) -> str:
    """Normalised file name without date/version/extension, so versions of one document share a family."""
    stem = FILE_DATE.sub(" ", Path(filename).stem.lower())
    stem = re.sub(r"[_\-.]+", " ", stem)
    stem = re.sub(r"\b(v\d+|ver\d*|final|draft|copy|old|new|latest)\b|\(\d+\)", " ", stem)
    return re.sub(r"[^a-z0-9]+", " ", stem).strip() or Path(filename).stem.lower()


def match_family(fam: str, known) -> str:
    """Snap a family name to a near-identical known one (e.g. small file-name differences between versions)."""
    known = list(known)
    if fam in known:
        return fam
    m = get_close_matches(fam, known, n=1, cutoff=0.85)
    return m[0] if m else fam


def latest_versions(entries) -> dict:
    """family -> registry entry with the most recent doc_date (upload time breaks ties; undated is oldest)."""
    best = {}
    for e in entries:
        fam = e.get("doc_family")
        if not fam or e.get("kind") != "doc":
            continue
        key = (e.get("doc_date") or "", e.get("uploaded_at") or "")
        if fam not in best or key > best[fam][0]:
            best[fam] = (key, e)
    return {f: v[1] for f, v in best.items()}


def version_status(meta: dict, latest: dict) -> str:
    if meta.get("kind") != "doc":
        return "N/A"
    d = meta.get("doc_date") or ""
    if not d:
        return "UNDATED"
    top = latest.get(meta.get("doc_family"))
    if not top or top["name"] == meta.get("source_file") or (top.get("doc_date") or "") <= d:
        return "LATEST"
    return f"OLDER (superseded by {top['name']} dated {top['doc_date']})"


def order_hits(hits, latest: dict):
    """Stable sort: latest/non-versioned chunks first, superseded ones after (relevance order kept inside)."""
    return sorted(hits, key=lambda h: 0 if version_status(h.metadata, latest).startswith(("LATEST", "N/A")) else 1)


def family_versions(entries, fam: str):
    v = [(e["name"], e.get("doc_date") or "undated") for e in entries if e.get("doc_family") == fam and e.get("kind") == "doc"]
    return sorted(v, key=lambda x: x[1], reverse=True)


def summarize_families(entries) -> dict:
    """Check helper / UI summary: family -> {'latest': name, 'versions': [(name, date)]} for multi-version families."""
    latest = latest_versions(entries)
    out = {}
    for fam, top in latest.items():
        vs = family_versions(entries, fam)
        if len(vs) > 1:
            out[fam] = {"latest": top["name"], "versions": vs}
    return out
