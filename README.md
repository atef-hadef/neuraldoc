<div align="center">

# 🧠 NeuralDoc — Document Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-4338ca?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square)
[![CI](https://github.com/atef-hadef/neuraldoc/actions/workflows/ci.yml/badge.svg)](https://github.com/atef-hadef/neuraldoc/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-11-22c55e?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

An end-to-end AI system that ingests documents (**PDF / DOCX / TXT**), extracts structured
knowledge, enables **semantic search** (FAISS + local sentence-transformers), answers questions
**with citations** via a **LangGraph** agent, and flags off-topic documents with
**Isolation Forest** anomaly detection — served through a **FastAPI** REST API and an
interactive **Streamlit** dashboard.

Fully local by default (zero API cost). Optional OpenAI refinement when `OPENAI_API_KEY` is set.

**Contents:** [Features](#-features) · [Architecture](#️-architecture) · [Quickstart](#-quickstart) · [API](#-api-reference) · [Testing](#-testing) · [Configuration](#️-configuration-env) · [Design decisions](#-key-design-decisions) · [Limitations](#-limitations-and-roadmap)

---

## ✨ Features

| Capability | Implementation |
|---|---|
| Document parsing | PyMuPDF (PDF), python-docx (DOCX incl. tables), UTF-8/latin-1 TXT |
| Chunking | Sliding character windows with overlap + tiny-tail merging |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, deterministic hashed fallback offline |
| Vector search | FAISS `IndexFlatIP` over L2-normalized vectors (cosine), persisted to disk |
| Question answering | LangGraph pipeline `retrieve → compose`, extractive + cited, optional LangChain LLM refine |
| Anomaly detection | Per-document mean embeddings + scikit-learn IsolationForest |
| Persistence | SQLite + SQLAlchemy 2.0 (documents, chunks) |
| API | FastAPI, Pydantic v2 validation, async-ready endpoints |
| Dashboard | Streamlit: upload / search / ask / anomalies tabs |
| Quality gates | pytest (unit + integration), GitHub Actions CI, Docker + Compose, loguru logging |

## 🏗️ Architecture

```text
                    ┌─────────────┐
                    │ PDF/DOCX/TXT│
                    └──────┬──────┘
                           ▼
              ┌────────────────────────┐
              │ Ingestion: parse+chunk │
              └──────┬─────────────────┘
                     ▼
        ┌───────────────────────┐     ┌──────────────┐
        │ Embedder (ST / hash)  │────▶│ FAISS index  │
        └───────────────────────┘     └──────┬───────┘
                                             ▼
   ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐
   │ FastAPI  │───▶│ SearchService│───▶│ QAService   │───▶│ Cited     │
   │ routers  │    │ cosine top-k │    │ (LangGraph) │    │ answer    │
   └──────────┘    └──────────────┘    └─────────────┘    └───────────┘
        │                 │                    │
        ▼                 ▼                    ▼
   ┌──────────┐    ┌──────────────┐    ┌─────────────────┐
   │ SQLAlchemy│    │ payloads.json│    │ IsolationForest │
   │ metadata  │    │ index.faiss  │    │ anomaly scan    │
   └──────────┘    └──────────────┘    └─────────────────┘
```

## 🚀 Quickstart

```bash
# 1. Install (Python 3.11+)
pip install -e ".[dev]"
cp .env.example .env

# 2. Seed demo documents
python scripts/seed_samples.py

# 3. Run the API
make api  # uvicorn neuraldoc.api.main:app --reload --port 8000

# 4. Run the dashboard (second terminal)
make dashboard  # streamlit run dashboard/app.py
```

Open **http://localhost:8000/docs** (API) and **http://localhost:8501** (dashboard).

### Docker

```bash
docker compose up --build
# API → :8000, dashboard → :8501
```

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status + index stats |
| `POST` | `/documents` | Upload PDF/DOCX/TXT (multipart `file`) → `201` |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{id}` | Delete + rebuild index → `204` |
| `GET` | `/search?q=...&top_k=5` | Ranked semantic hits with citations |
| `POST` | `/ask` | `{"question": "..."}` → cited answer |
| `GET` | `/anomalies` | IsolationForest scan, sorted by anomaly score |

Example:

```bash
curl -F "file=@data/samples/refund_policy.txt" http://localhost:8000/documents
curl "http://localhost:8000/search?q=refund+period&top_k=3"
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
curl http://localhost:8000/anomalies
```

## 🧪 Testing

```bash
pytest -q
```

11 tests: chunker/parser units, ingest→search→ask flow, anomaly detector,
and full API integration (upload/search/ask/anomalies/delete + 415 rejection).
Embeddings run through the hashed fallback in CI so the suite is offline-safe.

## ⚙️ Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `NEURALDOC_DATABASE_URL` | `sqlite:///./data/neuraldoc.db` | Metadata DB |
| `NEURALDOC_INDEX_DIR` | `./data/indices` | FAISS persistence |
| `NEURALDOC_UPLOAD_DIR` | `./data/uploads` | Temp upload staging |
| `NEURALDOC_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | ST model |
| `NEURALDOC_CHUNK_SIZE` / `NEURALDOC_CHUNK_OVERLAP` | `800` / `120` | Chunker tuning |
| `NEURALDOC_TOP_K` | `5` | Default retrieval depth |
| `OPENAI_API_KEY` | _(empty)_ | Enables LangChain LLM refinement |

## 📁 Project Layout

```text
src/neuraldoc/
  config.py  logging.py  exceptions.py  schemas.py  services.py
  ingestion/  retrieval/  anomaly/  db/  api/
dashboard/app.py  tests/  scripts/seed_samples.py
docker/  .github/workflows/ci.yml  docker-compose.yml
```

## 🧠 Key Design Decisions

1. **Local-first embeddings** with hashed fallback — the service boots and tests
   pass with a cold model cache or no network; the cosine contract is preserved.
2. **Extractive QA by default** — reproducible, zero-cost, always cited; the
   LangGraph agent cleanly adds LLM refinement only when credentials exist.
3. **Mean-pool document vectors** for anomaly detection — robust to variable
   chunk counts and sensitive to off-topic documents.
4. **Index rebuild on delete** — simple and correct at portfolio scale; a
   production upgrade path would use FAISS ID maps + soft deletes.

## 🚧 Limitations and roadmap

- Question answering is **extractive** by default (`retrieve → compose`): it returns the best cited passages. An LLM refinement step is optional.
- Retrieval is dense (cosine) only; a hybrid lexical + dense search and a re-ranker are the next step.
- Anomaly detection works at document level (mean embedding + IsolationForest).
- FAISS index is rebuilt on delete; for large collections, move to ID maps with soft deletes.
- Next: retrieval evaluation set (hit rate, MRR), authentication, and streaming answers.

## 📄 License

MIT — see [LICENSE](LICENSE).
