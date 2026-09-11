"""Cited question-answering orchestrated with LangGraph.

Graph: retrieve -> compose -> (optional LLM refine) -> answer-with-citations.

- Fully-local default: extractive answer built from the top-ranked chunks,
  always returned with explicit citations (document + chunk + score).
- Optional LangChain LLM refinement: if OPENAI_API_KEY (or another chat model
  factory) is configured, the draft is refined; citations are preserved.
This keeps the demo reproducible with zero API cost while showing real
agentic orchestration (LangChain + LangGraph per the stack requirements).
"""

import os
import re
from dataclasses import dataclass, field
from typing import TypedDict

from loguru import logger

from neuraldoc.retrieval.search import SearchHit, SearchService

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CitedAnswer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)


def _extractive_draft(question: str, hits: list[SearchHit]) -> str:
    if not hits:
        return "I couldn't find relevant information in the indexed documents."
    sentences: list[str] = []
    for hit in hits[:3]:
        sentences.extend(_SENT_SPLIT.split(hit.chunk.text.strip()))
    top = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
    if not top:
        top = [hits[0].chunk.text[:400]]
    return " ".join(f"• {s}" for s in top)


def _try_llm_refine(question: str, draft: str, context: str) -> str | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # type: ignore[import-not-found]
        prompt = (
            "Answer the question using ONLY the context below. "
            "Keep it concise and factual.\n\n"
            f"Question: {question}\nContext: {context}\nDraft: {draft}"
        )
        return llm.invoke(prompt).content  # type: ignore[no-any-return]
    except Exception as exc:
        logger.warning(f"LLM refinement skipped: {exc}")
        return None


class _GraphState(TypedDict, total=False):
    question: str
    hits: list[SearchHit]
    draft: str
    answer: str


def build_qa_graph(search_service: SearchService):
    """Build the LangGraph QA pipeline bound to a SearchService."""
    from langgraph.graph import END, StateGraph

    def retrieve(state: _GraphState) -> _GraphState:
        hits = search_service.search(state["question"], top_k=5)
        return {"hits": hits}

    def compose(state: _GraphState) -> _GraphState:
        hits = state.get("hits", [])
        draft = _extractive_draft(state["question"], hits)
        context = "\n".join(h.chunk.text for h, _ in [(x.chunk, x.score) for x in hits][:3])
        refined = _try_llm_refine(state["question"], draft, context)
        citations = [h.citation for h in hits]
        answer = refined or draft
        if citations:
            answer += "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)
        return {"draft": draft, "answer": answer}

    graph = StateGraph(_GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("compose", compose)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "compose")
    graph.add_edge("compose", END)
    return graph.compile()


class QAService:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self._graph = build_qa_graph(search_service)

    def ask(self, question: str) -> CitedAnswer:
        if not question.strip():
            raise ValueError("Question must not be empty")
        state = self._graph.invoke({"question": question})
        hits: list[SearchHit] = state.get("hits", [])
        return CitedAnswer(
            question=question,
            answer=state.get("answer", ""),
            citations=[h.citation for h in hits],
            hits=hits,
        )
