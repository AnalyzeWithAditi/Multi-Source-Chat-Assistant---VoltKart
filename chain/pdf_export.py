"""Export the chat (messages, timestamps, citations) to PDF with reportlab."""
from __future__ import annotations

import re
import textwrap
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

from chain.charts import render_png
from chain.memory_store import citation_label


def _clean(s) -> str:  # built-in PDF fonts are cp1252 only
    return str(s).replace("\u20b9", "Rs. ").encode("cp1252", "replace").decode("cp1252")


def _inline(s: str) -> str:
    s = escape(_clean(s))
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', s)


def _styles():
    base = getSampleStyleSheet()["BodyText"]
    return {
        "body": ParagraphStyle("b", parent=base, fontSize=9.5, leading=12.5),
        "cell": ParagraphStyle("c", parent=base, fontSize=8, leading=10),
        "meta": ParagraphStyle("m", parent=base, fontSize=8, leading=10, textColor=colors.grey),
        "head": ParagraphStyle("h", parent=base, fontSize=10.5, leading=14, spaceBefore=8),
        "cite": ParagraphStyle("ci", parent=base, fontSize=8, leading=10, leftIndent=8, textColor=colors.HexColor("#333333")),
        "code": ParagraphStyle("co", parent=base, fontName="Courier", fontSize=7, leading=8.5, leftIndent=12),
    }


def _table(block: list[str], st, width):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in block
            if not re.match(r"^\|?[\s:\-|]+\|?$", ln.strip())]
    n = max(len(r) for r in rows)
    data = [[Paragraph(_inline(c), st["cell"]) for c in r + [""] * (n - len(r))] for r in rows]
    t = Table(data, colWidths=[width / n] * n, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def _markdown(text: str, st, width) -> list:
    flow, lines, i = [], str(text).split("\n"), 0
    while i < len(lines):
        if lines[i].strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            flow.append(_table(block, st, width))
            flow.append(Spacer(1, 3))
            continue
        ln = lines[i]
        m = re.match(r"\s*#+\s*(.*)", ln)
        flow.append(Paragraph(_inline(f"**{m.group(1)}**") if m else (_inline(ln) or "&nbsp;"), st["body"]))
        i += 1
    return flow


def export_chat_pdf(messages: list[dict], title: str = "VoltKart chat export") -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=title)
    width = A4[0] - 36 * mm
    st = _styles()
    story = [Paragraph(f"<b>{escape(title)}</b>", getSampleStyleSheet()["Title"])]
    if not messages:
        story.append(Paragraph("No messages.", st["body"]))
    for m in messages:
        who = "User" if m["role"] == "user" else "Assistant"
        web = f" &nbsp; | &nbsp; Web search: {'ON' if m.get('web') else 'OFF'}"
        story.append(Paragraph(f"<b>{who}</b> &nbsp; <font size=8 color='grey'>{escape(m['ts'])}{web}</font>", st["head"]))
        story += _markdown(m["content"], st, width)
        cites = m.get("citations") or []
        for c in cites:  # charts are drawn from the stored spec (same values as the query result)
            if c.get("chart"):
                w = min(width, 150 * mm)
                story.append(Image(BytesIO(render_png(c["chart"])), width=w, height=w * 4 / 7))
        if cites:
            story.append(Paragraph("<b>Citations</b>", st["meta"]))
            for c in cites:
                story.append(Paragraph(_inline(citation_label(c)), st["cite"]))
                if c["type"] == "table":
                    story.append(Preformatted(_clean("\n".join(textwrap.wrap(c["sql"], 110, replace_whitespace=False))), st["code"]))
        story.append(Spacer(1, 6))
    doc.build(story)
    return buf.getvalue()
