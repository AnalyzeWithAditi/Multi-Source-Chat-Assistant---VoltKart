"""Streamlit UI: multiple chats, web-search toggle, file upload, inline charts, citations, PDF export."""
import html
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

st.set_page_config(page_title="VoltKart Assistant", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------- styling (UI only)
THEMES = {
    "Dark": dict(
        bg="#0B2E33", text="#E6F4F6", muted="#9CC3C8",
        sb_bg="#93B1B5", sb_text="#0B2E33",
        hl1="#4F7C82", hl2="#B8E3E9",
        card="#123F46", border="rgba(184,227,233,.25)", input_bg="#123F46",
        user_msg="rgba(79,124,130,.45)",
        panel_bg="#4F7C82", panel_text="#F2FBFC", status_off="rgba(255,255,255,.15)",
        err_bg="rgba(179,38,30,.22)",
    ),
    "Light": dict(
        bg="#EEF7F8", text="#0B2E33", muted="#4F7C82",
        sb_bg="#93B1B5", sb_text="#0B2E33",
        hl1="#4F7C82", hl2="#B8E3E9",
        card="#FFFFFF", border="rgba(79,124,130,.35)", input_bg="#FFFFFF",
        user_msg="rgba(184,227,233,.7)",
        panel_bg="#B8E3E9", panel_text="#0B2E33", status_off="rgba(11,46,51,.10)",
        err_bg="#FDECEA",
    ),
}
st.session_state.setdefault("theme", "Dark")
T = THEMES.get(st.session_state.theme, THEMES["Dark"])
VARS = (
    "<style>:root{--primary:#F4A7BE;--green:#93C572;--ink:#0B2E33;--files-bg:#4F7C82;--files-text:#F2FBFC;"
    + "".join(f"--{k.replace('_', '-')}:{v};" for k, v in T.items())
    + "}</style>"
)

STATIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&family=Source+Sans+3:wght@400;600;700&display=swap');

html, body, .stApp,
.stApp :is(p, label, h1, h2, h3, h4, li, button, input, textarea, caption),
[data-testid="stMarkdownContainer"] {
    font-family: 'Source Sans Pro', 'Source Sans 3', 'Source Sans', sans-serif;
}

/* page */
.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: var(--bg); }
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--bg); }
.block-container { padding-top: 2.2rem; }

/* main-area text */
.stApp [data-testid="stAppViewContainer"] :is(p, li, label, h1, h2, h3, h4, [data-testid="stMarkdownContainer"]) { color: var(--text); }
.stApp [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] :is(p, span) { color: var(--muted); }

/* sidebar */
.stApp [data-testid="stSidebar"], .stApp [data-testid="stSidebarContent"] { background: var(--sb-bg); }
.stApp [data-testid="stSidebar"] :is(p, li, label, h1, h2, h3, h4, small, [data-testid="stMarkdownContainer"]) { color: var(--sb-text); }
.stApp [data-testid="stSidebar"] [data-testid="stCaptionContainer"] :is(p, span) { color: var(--sb-text); opacity: .8; }
.stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: var(--files-bg); }
.stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] :is(span, small, div, p) { color: var(--files-text); }
.stApp [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg { fill: var(--files-text); color: var(--files-text); }
.stApp [data-testid="stSidebar"] :is([data-testid="stBaseButton-primary"], button[kind="primary"], [data-testid="stDownloadButton"] button) :is(p, span) { color: var(--ink); }

/* chat avatars (user and assistant): white background */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { background: #FFFFFF !important; }
[data-testid^="stChatMessageAvatar"] :is(span, svg, div) { color: var(--ink) !important; fill: var(--ink); background: transparent !important; }

/* uploader area / file list */
.stApp [data-testid="stSidebar"] .st-key-file_list { background: var(--files-bg); border-radius: 10px; padding: .4rem; margin-bottom: .5rem; }

/* header */
.vk-header { display: flex; align-items: center; gap: .75rem; margin-bottom: .5rem; }
.vk-bolt { width: 40px; height: 40px; border-radius: 10px; background: var(--primary);
           display: flex; align-items: center; justify-content: center; flex: none; }
.vk-title { font-size: 1.7rem; font-weight: 700; line-height: 1.1; color: var(--text); }
.vk-sub { font-size: .95rem; color: var(--muted); margin-top: 2px; }

/* messages */
[data-testid="stChatMessage"] { background: transparent; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--user-msg); border-radius: 12px;
}
.stApp [data-testid="stExpander"] details { background: var(--card); border: 1px solid var(--border); border-radius: 10px; }
.stApp [data-testid="stExpander"] summary :is(p, span) { color: var(--text); }

/* chat input */
[data-testid="stChatInput"] { background: var(--input-bg); border-radius: .75rem; }
[data-testid="stChatInput"] textarea { color: var(--text); background: transparent; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted); }

/* sidebar chat list: left aligned, truncated, subtle active highlight */
[class*="st-key-chat"] button {
    justify-content: flex-start; text-align: left; background: transparent; border: none;
    box-shadow: none; padding-left: .75rem;
}
[class*="st-key-chat"] button p {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; text-align: left;
}
[class*="st-key-chat"] button:hover { background: rgba(255, 255, 255, .35); }
[class*="st-key-chatactive"] button {
    background: var(--hl2); border-left: 4px solid var(--hl1); font-weight: 700;
}

/* web-search panel: its own background, different from the sidebar */
.stApp [data-testid="stSidebar"] .st-key-web_panel {
    background: var(--panel-bg); border-radius: 12px; padding: .9rem 1rem; margin: .5rem 0 1rem 0;
}
.stApp [data-testid="stSidebar"] .st-key-web_panel :is(p, label, small, [data-testid="stMarkdownContainer"]) { color: var(--panel-text); }
.stApp [data-testid="stSidebar"] .st-key-web_panel [data-testid="stCaptionContainer"] :is(p, span) { color: var(--panel-text); opacity: .85; }
.vk-panel-title { font-weight: 700; font-size: 1.15rem; margin-bottom: .35rem; }
.vk-status { font-size: .85rem; font-weight: 600; padding: 6px 12px; border-radius: 8px; margin: .25rem 0 .5rem 0; }
.vk-status.on { background: var(--green); color: var(--ink); }
.vk-status.off { background: var(--status-off); font-weight: 400; }

/* de-emphasised delete */
.st-key-delete_chat button { background: transparent; border: none; box-shadow: none; font-size: .85rem; }
.st-key-delete_chat button p { color: #7A1710 !important; }
.st-key-delete_chat button:hover { background: rgba(122, 23, 16, .12); }

/* friendly error card */
.vk-error { background: var(--err-bg); border-left: 5px solid #B3261E; border-radius: 10px;
            padding: .8rem 1rem; margin-top: .5rem; color: var(--text); }
.vk-error .t { font-weight: 700; margin-bottom: .25rem; }
.vk-error .d { font-size: .9rem; color: var(--muted); word-break: break-word; }
.vk-error .h { font-size: .9rem; margin-top: .4rem; }
</style>
"""

WEB_ON_CSS = """
<style>
[data-testid="stChatInput"] {
    box-shadow: 0 0 0 2px var(--green); background: color-mix(in srgb, var(--green) 18%, var(--input-bg));
}
</style>
"""

BOLT_SVG = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="#0B2E33" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M13 2 4 14h6l-1 8 9-12h-6z"/></svg>')

st.markdown(VARS + STATIC_CSS, unsafe_allow_html=True)
st.markdown(
    f'<div class="vk-header"><div class="vk-bolt">{BOLT_SVG}</div>'
    '<div><div class="vk-title">VoltKart Assistant</div>'
    '<div class="vk-sub">Ask across your tables, documents and the web</div></div></div>',
    unsafe_allow_html=True,
)

if missing_keys():
    st.error(f"Set {', '.join(missing_keys())} in your .env file (see .env.example), then restart.")
    st.stop()


@st.cache_resource
def get_assistant() -> Assistant:
    return Assistant()


@st.cache_data(show_spinner=False)
def chart_png(spec_json: str) -> bytes:
    return render_png(json.loads(spec_json))


def short(text: str, n: int = 28) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


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
        st.caption(f"{ts}  ·  Web search {'on' if web else 'off'}")
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


# ---- sidebar: theme, chats, web toggle, files, chat options ----
with st.sidebar:
    st.radio("Theme", ["Dark", "Light"], key="theme", horizontal=True)

    st.header("Chats")
    if st.button("New chat", use_container_width=True, type="primary", disabled=not msgs):  # an empty chat is reused
        st.session_state.chat_id = a.memory.new_chat()
        st.rerun()
    for c in a.memory.list_chats():
        is_active = c["id"] == chat_id
        key = f"chatactive{c['id']}" if is_active else f"chat{c['id']}"
        if st.button(short(c["title"]), key=key, help=c["title"], use_container_width=True):
            st.session_state.chat_id = c["id"]
            st.rerun()

    with st.container(key="web_panel"):
        st.markdown('<div class="vk-panel-title">Web search</div>', unsafe_allow_html=True)
        web = st.toggle("Use web search for my next message", value=False, key="web")
        if web:
            st.markdown('<div class="vk-status on">● Web search on for the next message</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="vk-status off">○ Web search off · files only</div>', unsafe_allow_html=True)
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
        files_df = pd.DataFrame([{"file": e["name"], "type": e["file_type"], "doc date": e.get("doc_date") or "",
                                  "family": e.get("doc_family", "") if e["kind"] == "doc" else "",
                                  "tables": ", ".join(e.get("tables", []))} for e in reg])
        with st.container(key="file_list"):
            st.dataframe(files_df.style.set_properties(**{"background-color": "#4F7C82", "color": "#F2FBFC"}),
                         hide_index=True, use_container_width=True)
        for fam, info in summarize_families(reg).items():
            st.caption(f"'{fam}': latest = {info['latest']}")
    else:
        st.info("No files uploaded yet. Files are shared by all chats.")

    st.header("Chat options")
    st.download_button("Export chat as PDF", data=export_chat_pdf(msgs, f"VoltKart chat: {chat_title}"),
                       file_name="chat.pdf", mime="application/pdf", disabled=not msgs,
                       type="primary", use_container_width=True)
    if st.button("Delete this chat", key="delete_chat"):
        a.memory.delete_chat(chat_id)
        st.rerun()

# accent the chat input while web search is on
if web:
    st.markdown(WEB_ON_CSS, unsafe_allow_html=True)

# ---- chat ----
st.subheader(chat_title)
for m in msgs:
    show_message(m["role"], m["content"], m["ts"], m.get("web"), m["citations"])

if prompt := st.chat_input("Ask about your tables and documents"):
    with st.chat_message("user"):
        st.markdown(prompt)
    res, err = None, None
    with st.status("Working on your question…", expanded=True) as status:
        st.write("📁 Searching your files")
        if web:
            st.write("🌐 Searching the web")
        st.write("✍️ Writing answer")
        try:
            res = a.ask(prompt, chat_id, web)
            status.update(label="Answer ready", state="complete", expanded=False)
        except Exception as ex:
            err = ex
            status.update(label="Something went wrong", state="error", expanded=False)
    if err is None:
        show_message("assistant", res["answer"], "just now", web, res["citations"])
        for w in res["warnings"]:
            st.caption(f"Note: {w}")
    else:
        hint = ("Web search may not be enabled for your Anthropic organization (Console settings)."
                if "web_search" in str(err).lower() else "")
        st.markdown(
            '<div class="vk-error"><div class="t">We couldn\'t get an answer this time</div>'
            f'<div class="d">{html.escape(str(err))}</div>'
            + (f'<div class="h">💡 {html.escape(hint)}</div>' if hint else "")
            + "</div>",
            unsafe_allow_html=True,
        )
