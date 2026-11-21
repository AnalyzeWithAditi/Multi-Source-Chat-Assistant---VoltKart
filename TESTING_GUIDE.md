# Testing guide

## 1. Automated (offline, no API keys)
```bash
python -m pytest tests/test_pipeline_steps.py -v
```
Covers: date/family extraction, TXT/DOCX/PDF/CSV/XLSX loaders and metadata, SQL joins and the revenue rule
(Delivered only, discounted `line_amount`), SQL read-only guard, latest-version resolution, Chroma persistence
(fake embeddings), multi-chat isolation and migration, web-citation parsing, chart specs/PNG rendering, PDF export (with chart and web citation), citation-tag checks.

Built-in check helpers you can also call from a shell:
`file_rules.check_chunk_metadata`, `domain_rules.check_revenue_rule`, `version_rules.summarize_families`,
`memory_store.check_citations`.

## 2. Manual end-to-end (needs both API keys)
1. `streamlit run app.py`, upload all files of the sample bundle (tables and documents; uploading more later also works).
2. Ask the evaluation questions in ONE chat, exact wording:
   - Tables: Q1-Q5 (Q2 is a follow-up and must use memory).
   - Documents: Q6-Q12. Check each answer cites the newest version (file, date, page/section).
   - Combined: Q13, Q14.
3. Open each answer's **Citations** expander: table answers show tables + SQL; document answers show file, doc date, page/section, status.
4. Ask something not in the files (e.g. "What is the CEO's phone number?") - expect "This is not in the provided sources."
5. Stop and restart the app, reload the page: files, history and answers are still there. Ask Q20 ("Summarize what we discussed so far").
6. Click **Export chat as PDF**, save it as `evaluation/chat1.pdf` (core-only build: one chat, so everything goes in chat1.pdf).

## 3. Bonus features (manual)
- **Web toggle:** with it OFF ask Q17 - expect "This is not in the provided sources." and no web links. Turn it ON, ask Q17 again, then Q18: web facts show as `[W#] title | URL` in Citations and the answer compares them with the VoltKart policy.
- **Multiple chats:** New chat, ask something, switch back and forth: each chat keeps its own history. Ask "what about the second one?" in a fresh chat - it must NOT know the other chat's context. Restart the app: all chats are still listed.
- **Charts:** ask Q15 and Q16 (and "show it as a pie chart"). Each chart has a title and axis labels, a `[T#]` citation with the SQL, and the numbers match `Citations -> preview`. Q19: in Chat 1 ask for the first question as a bar chart of the top 5 stores.
- Export each chat to PDF: charts, web on/off state per message and web sources are included.

## 4. Spot-check numbers independently
```python
import pandas as pd
o, i = pd.read_csv("orders.csv"), pd.read_csv("order_items.csv")
m = i.merge(o, on="order_id"); m = m[m.order_status == "Delivered"]
m["amt"] = m.quantity * m.unit_price_inr * (1 - m.discount_pct / 100)
print(m[m.order_date.str.startswith("2025")].groupby("store_id").amt.sum().sort_values(ascending=False))
```
