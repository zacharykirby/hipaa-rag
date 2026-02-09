"""
Document loader: detect type and yield pages as images for PDF, TIFF, and image files.
"""
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Magic bytes for file type detection
PDF_SIGNATURE = b"%PDF"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8"
TIFF_II = b"II*\x00"
TIFF_MM = b"MM\x00*"
WEBP_SIGNATURE = b"RIFF"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
PDF_EXTENSIONS = {".pdf"}
TIFF_EXTENSIONS = {".tif", ".tiff"}


def _read_magic(path: str | Path, size: int = 12) -> bytes:
    """Read first bytes of file for magic number detection."""
    with open(path, "rb") as f:
        return f.read(size)


def detect_document_type(path: str | Path) -> str:
    """
    Detect document type from path (extension + magic bytes).
    Returns one of: "pdf", "tiff", "image".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    magic = _read_magic(path)

    if magic.startswith(PDF_SIGNATURE) or suffix in PDF_EXTENSIONS:
        return "pdf"
    if magic.startswith(TIFF_II) or magic.startswith(TIFF_MM) or suffix in TIFF_EXTENSIONS:
        return "tiff"
    if (
        magic.startswith(PNG_SIGNATURE)
        or magic.startswith(JPEG_SIGNATURE)
        or (magic.startswith(WEBP_SIGNATURE) and b"WEBP" in magic[:20])
        or suffix in IMAGE_EXTENSIONS
    ):
        return "image"

    raise ValueError(
        f"Unsupported document type: {path}. "
        "Supported: PDF, TIFF, PNG, JPEG, WebP."
    )


def get_pages(
    document_path: str | Path,
    max_pages: int | None = None,
) -> Iterator[tuple[int, bytes]]:
    """
    Yield (page_index, png_bytes) for each page. 0-based page index.
    Single-page formats (PNG, JPEG, etc.) yield one item.
    Memory-efficient: one page in memory at a time.
    """
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    doc_type = detect_document_type(path)
    count = 0

    if doc_type == "pdf":
        import fitz

        doc = fitz.open(path)
        try:
            for i in range(len(doc)):
                if max_pages is not None and count >= max_pages:
                    break
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=150, alpha=False)
                png_bytes = pix.tobytes("png")
                yield (i, png_bytes)
                count += 1
        finally:
            doc.close()

    elif doc_type == "tiff":
        from io import BytesIO

        from PIL import Image

        img = Image.open(path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        n = 0
        while True:
            try:
                img.seek(n)
            except EOFError:
                break
            if max_pages is not None and count >= max_pages:
                break
            buf = BytesIO()
            frame = img.copy()
            if frame.mode in ("RGBA", "P"):
                frame = frame.convert("RGB")
            frame.save(buf, format="PNG")
            yield (n, buf.getvalue())
            count += 1
            n += 1
        if count == 0:
            raise ValueError(f"No pages found in TIFF: {path}")

    else:
        # Single image
        if max_pages is not None and max_pages <= 0:
            return
        with open(path, "rb") as f:
            data = f.read()
        # Normalize to PNG for API consistency (JPEG/WebP often accepted; we can pass through)
        if path.suffix.lower() in {".png"}:
            yield (0, data)
        else:
            from io import BytesIO

            from PIL import Image

            img = Image.open(BytesIO(data))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            yield (0, buf.getvalue())


def get_page_count(document_path: str | Path) -> int:
    """Return number of pages without loading page data."""
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    doc_type = detect_document_type(path)

    if doc_type == "pdf":
        import fitz

        doc = fitz.open(path)
        try:
            return len(doc)
        finally:
            doc.close()

    if doc_type == "tiff":
        from PIL import Image

        img = Image.open(path)
        n = 0
        while True:
            try:
                img.seek(n)
                n += 1
            except EOFError:
                break
        return n

    return 1
