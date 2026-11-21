"""Offline pipeline tests: no API keys needed (the vector test uses fake embeddings)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chain.memory_store import MemoryStore, check_citations, extract_tags, strip_tags  # noqa: E402
from chain.pdf_export import export_chat_pdf  # noqa: E402
from rules import domain_rules as dr  # noqa: E402
from rules import file_rules as fr  # noqa: E402
from rules import version_rules as vr  # noqa: E402


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))


def frames():
    return {
        "stores": pd.DataFrame({"store_id": ["S1", "S2"], "store_name": ["A", "B"], "store_manager": ["Mia", "Raj"]}),
        "orders": pd.DataFrame({"order_id": ["O1", "O2", "O3"], "customer_id": ["C1", "C2", "C1"],
                                "store_id": ["S1", "S2", "S1"], "order_date": ["2025-01-05", "2025-02-01", "2025-03-01"],
                                "order_status": ["Delivered", "Delivered", "Cancelled"]}),
        "order_items": pd.DataFrame({"order_item_id": ["I1", "I2", "I3"], "order_id": ["O1", "O2", "O3"],
                                     "product_id": ["P1", "P1", "P2"], "quantity": [2, 1, 5],
                                     "unit_price_inr": [100.0, 200.0, 50.0], "discount_pct": [10, 0, 0]}),
    }


def test_dates_and_family():
    assert vr.extract_doc_date("x.txt", "Title\nEffective Date: 1 April 2024 Version 3") == "2024-04-01"
    assert vr.extract_doc_date("Return_Policy_2025-01-15.docx", "no label") == "2025-01-15"
    assert vr.extract_doc_date("a.txt", "Document Date: 2025-09-01") == "2025-09-01"
    assert vr.doc_family("Return_Policy_2024-04-01.pdf") == vr.doc_family("return-policy_v2_2025-09-01.txt") == "return policy"


def test_txt_loader_metadata(tmp_path):
    p = tmp_path / "policy_2025-09-01.txt"
    p.write_text("Document Date: 2025-09-01\n\nRETURN WINDOW\nGold members: 14 days.\n\nREFUNDS\nRefunds take 5 business days.\n")
    f = fr.load_file(p)
    assert f.doc_date == "2025-09-01" and f.kind == "doc" and f.doc_family == "policy"
    assert fr.check_chunk_metadata(f.chunks) == []
    gold = next(c for c in f.chunks if "Gold" in c["text"])
    assert gold["meta"]["section"].startswith("RETURN WINDOW") and "lines" in gold["meta"]["section"]


def test_docx_loader(tmp_path):
    from docx import Document

    d = Document()
    d.add_heading("Returns", 1)
    d.add_paragraph("Effective Date: 2025-01-15")
    d.add_paragraph("Gold members: 14 days.")
    t = d.add_table(rows=2, cols=2)
    for i, v in enumerate(["Tier", "Days", "Gold", "14"]):
        t.cell(i // 2, i % 2).text = v
    p = tmp_path / "policy.docx"
    d.save(p)
    f = fr.load_file(p)
    assert f.doc_date == "2025-01-15" and fr.check_chunk_metadata(f.chunks) == []
    assert any(c["meta"]["table"] == "1" and "Tier: Gold | Days: 14" in c["text"] for c in f.chunks)
    assert any(c["meta"]["section"] == "Returns" and "Gold members" in c["text"] for c in f.chunks)


def test_pdf_loader_pages(tmp_path):
    from reportlab.pdfgen import canvas

    p = tmp_path / "notice.pdf"
    c = canvas.Canvas(str(p))
    c.drawString(72, 800, "Document Date: 2025-11-05")
    c.showPage()
    c.drawString(72, 800, "Store closed in December.")
    c.save()
    f = fr.load_file(p)
    assert f.doc_date == "2025-11-05"
    assert {c["meta"]["page"] for c in f.chunks} == {1, 2}
    assert fr.check_chunk_metadata(f.chunks) == []


def test_csv_xlsx_loaders(tmp_path):
    df = pd.DataFrame({"Store ID": ["S1"], "Opened Date": pd.to_datetime(["2024-03-01"])})
    df.to_excel(tmp_path / "stores.xlsx", index=False)
    df.assign(**{"Opened Date": "2024-03-01"}).to_csv(tmp_path / "stores2.csv", index=False)
    x, c = fr.load_file(tmp_path / "stores.xlsx"), fr.load_file(tmp_path / "stores2.csv")
    assert list(x.tables["stores"].columns) == ["store_id", "opened_date"]
    assert x.tables["stores"]["opened_date"][0] == "2024-03-01"
    assert fr.check_chunk_metadata(x.chunks + c.chunks) == []
    assert any(ch["meta"]["row"] == 2 and "store_id: S1" in ch["text"] for ch in c.chunks)
    assert any(ch["meta"]["section"] == "schema" for ch in x.chunks)


def test_joins_and_revenue_rule():
    e = dr.TableEngine()
    e.add(frames())
    r = e.run_sql("SELECT s.store_name, s.store_manager, SUM(r.line_amount) AS rev FROM revenue_lines r "
                  "JOIN stores s ON s.store_id = r.store_id GROUP BY 1, 2 ORDER BY rev DESC")
    assert list(r.df["store_name"]) == ["B", "A"]
    assert list(r.df["rev"]) == [200.0, 180.0]  # A: 2*100*0.9; cancelled O3 excluded
    assert r.tables == ["order_items", "orders", "stores"] and r.views == ["revenue_lines"]
    assert dr.check_revenue_rule(e) == []


def test_sql_guard():
    e = dr.TableEngine()
    e.add(frames())
    for bad in ("DROP TABLE stores", "SELECT 1; SELECT 2", "DELETE FROM orders", "PRAGMA table_info(orders)"):
        with pytest.raises(ValueError):
            e.run_sql(bad)


def test_version_resolution(tmp_path):
    entries = []
    for n, d, t in [("pol_2024-04-01.txt", "2024-04-01", "Gold: 7 days"), ("pol_2025-09-01.txt", "2025-09-01", "Gold: 14 days")]:
        (tmp_path / n).write_text(f"Document Date: {d}\n{t}")
        f = fr.load_file(tmp_path / n)
        entries.append({"name": n, "kind": "doc", "doc_date": f.doc_date, "doc_family": f.doc_family})
    assert vr.latest_versions(entries)["pol"]["name"] == "pol_2025-09-01.txt"
    assert vr.summarize_families(entries)["pol"]["latest"] == "pol_2025-09-01.txt"
    old = {"kind": "doc", "doc_date": "2024-04-01", "doc_family": "pol", "source_file": "pol_2024-04-01.txt"}
    assert vr.version_status(old, vr.latest_versions(entries)).startswith("OLDER")


def test_vector_store_latest_and_persistence(tmp_path):
    pytest.importorskip("chromadb")
    pytest.importorskip("langchain_chroma")
    pytest.importorskip("langchain_core")
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from chain.rag_chain import format_hits, retrieve
    from vectorstore.chroma_store import VectorStore

    emb, entries = DeterministicFakeEmbedding(size=16), []
    st = VectorStore(emb, tmp_path / "chroma")
    for n, d in [("pol_2024-04-01.txt", "2024-04-01"), ("pol_2025-09-01.txt", "2025-09-01")]:
        (tmp_path / n).write_text(f"Document Date: {d}\nGold: window of {d}")
        f = fr.load_file(tmp_path / n)
        st.add(f.chunks)
        entries.append({"name": n, "kind": "doc", "doc_date": f.doc_date, "doc_family": f.doc_family})
    st2 = VectorStore(emb, tmp_path / "chroma")  # new instance = restart; data must survive
    hits = retrieve(st2, entries, "gold return window", k=1)
    assert hits[0].metadata["doc_date"] == "2025-09-01"  # latest forced in and ordered first
    text, cites = format_hits(hits, entries, 1)
    assert cites[0]["status"] == "LATEST" and "[D1]" in text


def test_memory_persists_and_history(tmp_path):
    p = tmp_path / "c.db"
    m = MemoryStore(p)
    chat = m.new_chat()
    m.add(chat, "user", "top store?")
    m.add(chat, "assistant", "Store B [T1]", [{"id": "T1", "type": "table", "sql": "SELECT 1", "tables": ["stores"], "rows": 1}])
    h = MemoryStore(p).history(chat)  # fresh instance = restart
    assert len(h) == 2 and "[T1]" not in h[1]["content"] and "SELECT 1" in h[1]["content"]


def test_chats_are_isolated_and_titled(tmp_path):
    m = MemoryStore(tmp_path / "c.db")
    a, b = m.new_chat(), m.new_chat()
    m.add(a, "user", "Which store had the highest revenue in 2025, and who manages it?", web=True)
    m.add(b, "user", "What is the return window for Gold members?")
    assert [x["content"] for x in m.history(a)] == ["Which store had the highest revenue in 2025, and who manages it?"]
    assert [x["content"] for x in m.history(b)] == ["What is the return window for Gold members?"]
    assert m.all(a)[0]["web"] is True and m.all(b)[0]["web"] is False
    titles = {c["id"]: c["title"] for c in m.list_chats()}
    assert titles[a].startswith("Which store") and titles[b].startswith("What is the return")
    m.clear(a)
    assert m.all(a) == [] and len(m.all(b)) == 1
    m.delete_chat(b)
    assert [c["id"] for c in m.list_chats()] == [a]


def test_old_single_chat_database_is_migrated(tmp_path):
    import sqlite3

    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, role TEXT, content TEXT, citations TEXT)")
    con.execute("INSERT INTO messages(ts, role, content, citations) VALUES ('2026-01-01T00:00:00','user','hello','[]')")
    con.commit()
    con.close()
    m = MemoryStore(p)
    chats = m.list_chats()
    assert len(chats) == 1 and m.all(chats[0]["id"])[0]["content"] == "hello"


def test_web_answer_text_and_citations():
    from chain.memory_store import answer_text, web_citations

    blocks = [{"type": "text", "text": "Let me search."},
              {"type": "server_tool_use", "name": "web_search", "input": {"query": "x"}},
              {"type": "web_search_tool_result", "content": [{"type": "web_search_result", "url": "https://a.example/1", "title": "A"},
                                                             {"type": "web_search_result", "url": "https://b.example/2", "title": "B"}]},
              {"type": "text", "text": "Amazon allows 10 days.",
               "citations": [{"type": "web_search_result_location", "url": "https://a.example/1", "title": "A", "cited_text": "10 days"}]}]
    assert answer_text(blocks) == "Amazon allows 10 days."
    w = web_citations([blocks, "plain string content"])
    assert w == [{"id": "W1", "type": "web", "title": "A", "url": "https://a.example/1", "snippet": "10 days"}]
    assert web_citations([[blocks[2]]])[0]["url"] == "https://a.example/1"  # falls back to results if nothing cited
    assert web_citations([]) == []


def test_chart_spec_matches_query_values():
    from chain.charts import build_spec

    e = dr.TableEngine()
    e.add(frames())
    df = e.run_sql("SELECT s.store_name, SUM(r.line_amount) AS revenue FROM revenue_lines r "
                   "JOIN stores s ON s.store_id = r.store_id GROUP BY 1 ORDER BY 2 DESC").df
    spec = build_spec(df, "Bar", "Revenue by store", "Store", "Revenue (INR)")
    assert spec["labels"] == ["B", "A"] and spec["series"] == [{"name": "revenue", "values": [200.0, 180.0]}]
    assert spec["title"] == "Revenue by store" and spec["x_label"] == "Store" and spec["y_label"] == "Revenue (INR)"
    with pytest.raises(ValueError):
        build_spec(df, "scatter", "t", "x", "y")
    with pytest.raises(ValueError):
        build_spec(df[["store_name"]], "bar", "t", "x", "y")
    with pytest.raises(ValueError):
        build_spec(pd.DataFrame({"a": list("abc"), "b": [1, -1, 2]}), "pie", "t", "x", "y")


def test_chart_png_renders():
    pytest.importorskip("matplotlib")
    from chain.charts import build_spec, render_png

    df = pd.DataFrame({"month": ["2025-01", "2025-02", "2025-03"], "units": [3, 0, 5]})
    for kind in ("bar", "line", "pie"):
        assert render_png(build_spec(df, kind, "Units", "Month", "Units sold")).startswith(b"\x89PNG")


def test_pdf_export_and_citation_checks():
    pytest.importorskip("matplotlib")
    from chain.charts import build_spec

    spec = build_spec(pd.DataFrame({"c": ["x", "y"], "v": [1, 2]}), "bar", "T", "C", "V")
    msgs = [{"ts": "2026-01-01T10:00:00", "role": "user", "content": "Total in ₹?", "citations": [], "web": True},
            {"ts": "2026-01-01T10:00:05", "role": "assistant", "content": "| a | b |\n|---|---|\n| 1 | 2 |\n**Done** [T1]", "web": True,
             "citations": [{"id": "T1", "type": "table", "tables": ["orders"], "views": [], "sql": "SELECT 1", "rows": 1, "chart": spec},
                           {"id": "W1", "type": "web", "title": "Amazon returns", "url": "https://a.example/1", "snippet": ""}]}]
    assert export_chat_pdf(msgs, "Chat").startswith(b"%PDF")
    assert extract_tags("x [T1, D2] y [D2]") == ["T1", "D2"] and strip_tags("x [T1]") == "x"
    assert check_citations("fact [T9]", [{"id": "T1"}]) and not check_citations("fact [T1]", [{"id": "T1"}])
    assert not check_citations("This is not in the provided sources.", [{"id": "T1"}])
