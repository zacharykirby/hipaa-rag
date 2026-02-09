"""Tests for document loader."""
import os
import tempfile
from pathlib import Path

import pytest

from hipaa_rag.loader import detect_document_type, get_page_count, get_pages


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "test_data"


@pytest.fixture
def sample_image_path(test_data_dir):
    path = test_data_dir / "bronchitis_chart.png"
    if not path.exists():
        pytest.skip("test_data/bronchitis_chart.png not found")
    return path


def test_detect_document_type_image(sample_image_path):
    assert detect_document_type(sample_image_path) == "image"


def test_detect_document_type_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        detect_document_type(Path("/nonexistent/file.png"))


def test_get_pages_single_image(sample_image_path):
    pages = list(get_pages(sample_image_path))
    assert len(pages) == 1
    page_idx, png_bytes = pages[0]
    assert page_idx == 0
    assert isinstance(png_bytes, bytes)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_get_page_count_single_image(sample_image_path):
    assert get_page_count(sample_image_path) == 1


def test_get_pages_respects_max_pages(sample_image_path):
    pages = list(get_pages(sample_image_path, max_pages=1))
    assert len(pages) == 1
    pages = list(get_pages(sample_image_path, max_pages=0))
    assert len(pages) == 0


@pytest.fixture
def sample_pdf_path(test_data_dir):
    """Create a 2-page PDF from test images."""
    import fitz

    img1 = test_data_dir / "bronchitis_chart.png"
    if not img1.exists():
        pytest.skip("test_data/bronchitis_chart.png not found")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, filename=str(img1))
    page2 = doc.new_page()
    page2.insert_image(page2.rect, filename=str(img1))
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    yield path
    Path(path).unlink(missing_ok=True)


def test_detect_document_type_pdf(sample_pdf_path):
    assert detect_document_type(sample_pdf_path) == "pdf"


def test_get_pages_pdf(sample_pdf_path):
    pages = list(get_pages(sample_pdf_path))
    assert len(pages) == 2
    for i, (idx, png_bytes) in enumerate(pages):
        assert idx == i
        assert isinstance(png_bytes, bytes)
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_get_pages_pdf_max_pages(sample_pdf_path):
    pages = list(get_pages(sample_pdf_path, max_pages=1))
    assert len(pages) == 1


def test_get_page_count_pdf(sample_pdf_path):
    assert get_page_count(sample_pdf_path) == 2


def test_unsupported_type_raises(test_data_dir):
    # Create a file with unknown magic bytes
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(b"NOT_A_VALID_FORMAT")
        path = f.name
    try:
        with pytest.raises(ValueError, match="Unsupported document type"):
            detect_document_type(path)
    finally:
        Path(path).unlink(missing_ok=True)
