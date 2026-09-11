"""Anomaly detection over document embeddings (Isolation Forest)."""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyResult:
    document_id: int
    filename: str
    score: float  # lower = more anomalous (decision_function)
    is_anomaly: bool


class AnomalyDetector:
    """Fits IsolationForest on per-document mean embeddings.

    Each document is represented by the mean of its chunk vectors, which is
    robust to variable chunk counts and highlights off-topic documents.
    """

    def __init__(self, contamination: float = 0.08):
        self.contamination = contamination

    def detect(self, doc_vectors: dict[int, tuple[str, np.ndarray]]) -> list[AnomalyResult]:
        """doc_vectors: {document_id: (filename, mean_vector)}."""
        if len(doc_vectors) < 3:
            logger.info("Not enough documents for anomaly detection (need >= 3)")
            return [
                AnomalyResult(document_id=doc_id, filename=name, score=0.0, is_anomaly=False)
                for doc_id, (name, _) in doc_vectors.items()
            ]
        ids = list(doc_vectors.keys())
        names = [doc_vectors[i][0] for i in ids]
        matrix = np.stack([doc_vectors[i][1] for i in ids]).astype(np.float64)
        model = IsolationForest(contamination=self.contamination, random_state=42)
        preds = model.fit_predict(matrix)
        scores = model.decision_function(matrix)
        results = [
            AnomalyResult(
                document_id=doc_id,
                filename=name,
                score=float(score),
                is_anomaly=bool(pred == -1),
            )
            for doc_id, name, pred, score in zip(ids, names, preds, scores)
        ]
        logger.info(f"Anomaly scan: {sum(r.is_anomaly for r in results)}/{len(results)} flagged")
        return sorted(results, key=lambda r: r.score)
