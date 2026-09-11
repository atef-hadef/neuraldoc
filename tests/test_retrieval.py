"""Retrieval, QA and anomaly tests (offline-safe via hashed embeddings)."""

from neuraldoc.anomaly.detector import AnomalyDetector


def test_ingest_search_ask_flow(service, sample_txt):
    doc = service.ingest_file(sample_txt, original_filename="policy.txt")
    assert doc.chunk_count >= 1

    hits = service.search_service.search("refund within 30 days", top_k=3)
    assert hits, "expected at least one search hit"
    assert any("refund" in h.chunk.text.lower() for h in hits)

    answer = service.qa_service.ask("What is the refund policy?")
    assert "refund" in answer.answer.lower()
    assert answer.citations, "QA must return citations"


def test_anomaly_detector_needs_minimum():
    detector = AnomalyDetector()
    import numpy as np

    few = {1: ("a.txt", np.ones(8)), 2: ("b.txt", np.ones(8))}
    assert all(not r.is_anomaly for r in detector.detect(few))


def test_anomaly_detector_flags_outlier(service, tmp_path):
    import numpy as np

    vecs = {i: (f"doc{i}.txt", np.full(8, float(i))) for i in range(1, 9)}
    vecs[99] = ("outlier.txt", np.full(8, 1000.0))
    results = AnomalyDetector(contamination=0.1).detect(vecs)
    assert any(r.document_id == 99 and r.is_anomaly for r in results)
