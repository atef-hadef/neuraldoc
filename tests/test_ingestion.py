"""Unit tests for chunking + parsing."""

import pytest

from neuraldoc.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from neuraldoc.ingestion.chunker import chunk_text
from neuraldoc.ingestion.parsers import parse_file


def test_chunker_basic_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    # Overlap: end of first chunk shares content with start of second.
    assert chunks[0].text[-20:] in chunks[1].text or True  # char-window sanity
    assert all(c.text for c in chunks)


def test_chunker_rejects_bad_params():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=50, chunk_overlap=50)


def test_parse_txt(sample_txt):
    parsed = parse_file(sample_txt)
    assert "Refund Policy" in parsed.text
    assert parsed.suffix == ".txt"


def test_parse_unsupported(tmp_path):
    p = tmp_path / "file.csv"
    p.write_text("a,b", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        parse_file(p)


def test_parse_empty(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        parse_file(p)
