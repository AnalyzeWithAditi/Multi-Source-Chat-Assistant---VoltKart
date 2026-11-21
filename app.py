"""Streamlit UI: multiple chats, web-search toggle, file upload, inline charts, citations, PDF export."""
import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from chain.charts import render_png  # noqa: E402
from chain.memory_store import citation_label  # noqa: E402
from chain.pdf_export import export_chat_pdf  # noqa: E402
from chain.rag_chain import Assistant, missing_keys  # noqa: E402
from rules.version_rules import summarize_families  # noqa: E402
from vectorstore.chroma_store import load_registry  # noqa: E402

st.set_page_config(page_title="VoltKart Assistant", layout="wide")
st.title("VoltKart multi-source assistant")

if missing_keys():
    st.error(f"Set {', '.join(missing_keys())} in your .env file (see .env.example), then restart.")
    st.stop()


@st.cache_resource
def get_assistant() -> Assistant:
    return Assistant()


@st.cache_data(show_spinner=False)
def chart_png(spec_json: str) -> bytes:
    return render_png(json.loads(spec_json))


a = get_assistant()
st.session_state.setdefault("up", 0)
st.session_state.setdefault("notes", [])

# ---- which chat is open ----
chats = a.memory.list_chats() or [{"id": a.memory.new_chat(), "title": "New chat"}]
if st.session_state.get("chat_id") not in {c["id"] for c in chats}:
    st.session_state.chat_id = chats[0]["id"]
chat_id = st.session_state.chat_id
msgs = a.memory.all(chat_id)
chat_title = next(c["title"] for c in a.memory.list_chats() or chats if c["id"] == chat_id)


def show_message(role, content, ts, web, cites):
    with st.chat_message(role):
        st.markdown(content)
        st.caption(f"{ts}  |  web search {'on' if web else 'off'}")
        for c in cites:
            if c.get("chart"):
                st.image(chart_png(json.dumps(c["chart"])))
        if cites:
            with st.expander(f"Citations ({len(cites)})"):
                for c in cites:
                    if c["type"] == "web":
                        st.markdown(f"**[{c['id']}] [{c['title']}]({c['url']})**")
                        if c.get("snippet"):
                            st.caption(c["snippet"])
                        continue
                    st.markdown(f"**{citation_label(c)}**")
                    if c["type"] == "table":
                        st.code(c["sql"], language="sql")
                        st.markdown(c["preview"])
                    else:
                        st.caption(c.get("snippet", ""))


# ---- sidebar: chats, web toggle, files, export ----
with st.sidebar:
    st.header("Chats")
    if st.button("New chat", use_container_width=True, disabled=not msgs):  # an empty chat is reused
        st.session_state.chat_id = a.memory.new_chat()
        st.rerun()
    for c in a.memory.list_chats():
        if st.button(c["title"], key=f"chat{c['id']}", use_container_width=True,
                     type="primary" if c["id"] == chat_id else "secondary"):
            st.session_state.chat_id = c["id"]
            st.rerun()

    st.header("Web search")
    web = st.toggle("Use web search for my next message", value=False, key="web")
    st.caption("Off: the assistant never touches the web. On: it can combine web results with your files.")

    st.header("Files")
    files = st.file_uploader("Upload CSV, XLSX, PDF, DOCX, TXT", type=["csv", "xlsx", "pdf", "docx", "txt"],
                             accept_multiple_files=True, key=f"up{st.session_state.up}")
    if files:
        notes = []
        for f in files:
            with st.spinner(f"Reading {f.name}..."):
                try:
                    e = a.ingest(f.name, f.getvalue())
                    notes.append(("ok", f"{f.name}: {e['chunks']} chunks" + (f" - {e['warning']}" if e.get("warning") else "")))
                except Exception as ex:
                    notes.append(("err", f"{f.name}: {ex}"))
        st.session_state.notes = notes
        st.session_state.up += 1
        st.rerun()
    for kind, msg in st.session_state.notes:
        (st.success if kind == "ok" else st.error)(msg)

    reg = load_registry()
    if reg:
        st.dataframe(pd.DataFrame([{"file": e["name"], "type": e["file_type"], "doc date": e.get("doc_date") or "",
                                    "family": e.get("doc_family", "") if e["kind"] == "doc" else "",
                                    "tables": ", ".join(e.get("tables", []))} for e in reg]),
                     hide_index=True, use_container_width=True)
        for fam, info in summarize_families(reg).items():
            st.caption(f"'{fam}': latest = {info['latest']}")
    else:
        st.info("No files uploaded yet. Files are shared by all chats.")

    st.header("This chat")
    st.download_button("Export chat as PDF", data=export_chat_pdf(msgs, f"VoltKart chat: {chat_title}"),
                       file_name="chat.pdf", mime="application/pdf", disabled=not msgs)
    if st.button("Delete this chat"):
        a.memory.delete_chat(chat_id)
        st.rerun()

# ---- chat ----
st.subheader(chat_title)
for m in msgs:
    show_message(m["role"], m["content"], m["ts"], m.get("web"), m["citations"])

if prompt := st.chat_input("Ask about your tables and documents"):
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        with st.spinner("Thinking..."):
            res = a.ask(prompt, chat_id, web)
        show_message("assistant", res["answer"], "just now", web, res["citations"])
        for w in res["warnings"]:
            st.caption(f"Note: {w}")
    except Exception as ex:
        hint = " Web search may not be enabled for your Anthropic organization (Console settings)." if "web_search" in str(ex).lower() else ""
        st.error(f"Request failed: {ex}.{hint}")
