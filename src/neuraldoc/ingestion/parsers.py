"""Parse PDF / DOCX / TXT files into plain text with metadata.

Design notes:
- PyMuPDF (fitz) for PDFs: page-aware text extraction, tolerant of scanned
  PDFs returning empty strings (caller decides whether to OCR upstream).
- python-docx for DOCX: paragraph + table text concatenation.
- TXT: UTF-8 with graceful fallback to latin-1.
"""

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from neuraldoc.exceptions import EmptyDocumentError, UnsupportedFileTypeError

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt"}


@dataclass
class ParsedDocument:
    filename: str
    suffix: str
    text: str
    pages: int = 1
    metadata: dict = field(default_factory=dict)


def _parse_pdf(path: Path) -> ParsedDocument:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        pages_text = [page.get_text("text") or "" for page in doc]
    finally:
        doc.close()
    text = "\n".join(t.strip() for t in pages_text if t.strip())
    return ParsedDocument(
        filename=path.name,
        suffix=".pdf",
        text=text,
        pages=len(pages_text),
        metadata={"pages": len(pages_text)},
    )


def _parse_docx(path: Path) -> ParsedDocument:
    import docx

    document = docx.Document(str(path))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n".join(parts)
    return ParsedDocument(filename=path.name, suffix=".docx", text=text)


def _parse_txt(path: Path) -> ParsedDocument:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    return ParsedDocument(filename=path.name, suffix=".txt", text=text.strip())


def parse_file(path: str | Path) -> ParsedDocument:
    """Parse a file on disk. Raises UnsupportedFileTypeError / EmptyDocumentError."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )
    logger.info(f"Parsing document: {p.name}")
    parsed = {"pdf": _parse_pdf, "docx": _parse_docx, "txt": _parse_txt}[suffix[1:]](p)
    if not parsed.text.strip():
        raise EmptyDocumentError(f"No extractable text found in {p.name}")
    return parsed
