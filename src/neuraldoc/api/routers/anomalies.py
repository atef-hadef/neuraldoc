"""Anomaly detection endpoint."""

from fastapi import APIRouter, Depends

from neuraldoc.api.deps import get_service
from neuraldoc.schemas import AnomalyOut
from neuraldoc.services import DocumentService

router = APIRouter(tags=["anomalies"])


@router.get("/anomalies", response_model=list[AnomalyOut])
def anomalies(service: DocumentService = Depends(get_service)) -> list[AnomalyOut]:
    return [AnomalyOut(**vars(r)) for r in service.detect_anomalies()]
