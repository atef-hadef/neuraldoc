"""NeuralDoc Streamlit dashboard — upload, search, ask, anomalies."""

import os

import requests
import streamlit as st

API_URL = os.getenv("NEURALDOC_API_URL", "http://localhost:8000")

st.set_page_config(page_title="NeuralDoc", page_icon="🧠", layout="wide")
st.title("🧠 NeuralDoc — Intelligent Document Intelligence")

tab_upload, tab_search, tab_ask, tab_anomalies = st.tabs(
    ["📤 Upload", "🔍 Search", "💬 Ask", "🚨 Anomalies"]
)

with tab_upload:
    st.subheader("Ingest documents (PDF / DOCX / TXT)")
    files = st.file_uploader("Drop files", accept_multiple_files=True, type=["pdf", "docx", "txt"])
    if st.button("Ingest", disabled=not files):
        for f in files:
            resp = requests.post(f"{API_URL}/documents", files={"file": (f.name, f, f.type)})
            if resp.ok:
                st.success(f"Ingested {f.name}")
            else:
                st.error(f"{f.name}: {resp.text}")
    resp = requests.get(f"{API_URL}/documents")
    if resp.ok:
        docs = resp.json()
        st.write(f"**{len(docs)} documents indexed**")
        st.dataframe(docs, use_container_width=True)

with tab_search:
    st.subheader("Semantic search")
    query = st.text_input("Query", placeholder="e.g. What is the refund policy?")
    top_k = st.slider("Top K", 1, 20, 5)
    if st.button("Search") and query:
        resp = requests.get(f"{API_URL}/search", params={"q": query, "top_k": top_k})
        if resp.ok:
            for hit in resp.json()["hits"]:
                st.markdown(f"**{hit['citation']}** — score `{hit['score']:.3f}`")
                st.write(hit["text"][:800])
                st.divider()
        else:
            st.error(resp.text)

with tab_ask:
    st.subheader("Ask with citations")
    question = st.text_input("Question", placeholder="e.g. Summarize the key risks in these contracts")
    if st.button("Ask") and question:
        resp = requests.post(f"{API_URL}/ask", json={"question": question})
        if resp.ok:
            data = resp.json()
            st.markdown(data["answer"])
            with st.expander("Citations"):
                st.write(data["citations"])
        else:
            st.error(resp.text)

with tab_anomalies:
    st.subheader("Anomaly detection (Isolation Forest)")
    if st.button("Run scan"):
        resp = requests.get(f"{API_URL}/anomalies")
        if resp.ok:
            rows = resp.json()
            flagged = [r for r in rows if r["is_anomaly"]]
            st.metric("Documents scanned", len(rows))
            st.metric("Anomalies flagged", len(flagged))
            st.dataframe(rows, use_container_width=True)
        else:
            st.error(resp.text)
