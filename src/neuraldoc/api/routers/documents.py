"""Document upload / list / delete endpoints."""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from neuraldoc.api.deps import get_service
from neuraldoc.exceptions import EmptyDocumentError, NeuralDocError, UnsupportedFileTypeError
from neuraldoc.schemas import DocumentOut
from neuraldoc.services import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED = {".pdf", ".docx", ".txt"}


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile = File(...), service: DocumentService = Depends(get_service)
) -> DocumentOut:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{suffix}'")
    tmp = service.settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        doc = service.ingest_file(tmp, original_filename=file.filename)
    except (UnsupportedFileTypeError, EmptyDocumentError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except NeuralDocError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp.unlink(missing_ok=True)
    return DocumentOut.model_validate(doc)


@router.get("", response_model=list[DocumentOut])
def list_documents(service: DocumentService = Depends(get_service)) -> list[DocumentOut]:
    return [DocumentOut.model_validate(d) for d in service.list_documents()]


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, service: DocumentService = Depends(get_service)) -> None:
    if not service.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
