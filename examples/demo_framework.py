"""
Demo of the SecureRAG framework: single image, multi-page PDF, structured extraction.
"""
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from hipaa_rag import SecureRAG

load_dotenv()


def make_sample_pdf(image_paths: list[Path], out_path: Path) -> None:
    """Create a multi-page PDF from image files (for demo)."""
    import fitz

    doc = fitz.open()
    for img_path in image_paths:
        if not img_path.exists():
            continue
        page = doc.new_page()
        page.insert_image(page.rect, filename=str(img_path))
    doc.save(str(out_path))
    doc.close()


def main():
    print("=== SecureRAG Framework Demo ===\n")

    rag = SecureRAG(model="qwen2-vl", enable_audit_log=True)
    test_data = Path(__file__).parent.parent / "test_data"
    bronchitis_chart = test_data / "bronchitis_chart.png"

    # --- 1. Single image (same API as before) ---
    print("1. Simple query (single image)\n")
    result = rag.query(
        document=str(bronchitis_chart),
        question="What is the patient's primary diagnosis with ICD-10 code?",
    )
    print(f"Answer: {result.answer}\n")
    print(f"Tokens used: {result.tokens_used}, Pages: {result.page_count}\n")

    # --- 2. Structured extraction (single image) ---
    print("-" * 50)
    print("\n2. Extract structured data (single image)\n")
    fields = [
        "patient_name",
        "date_of_birth",
        "primary_diagnosis",
        "prescribed_medications",
    ]
    extracted = rag.extract_structured_data(
        document=str(bronchitis_chart),
        fields=fields,
    )
    print("Extracted fields:")
    for key, value in extracted.items():
        print(f"  {key}: {value}")

    # --- 3. Multi-page PDF (same API) ---
    images = [test_data / "chest_pain_er_1.png", test_data / "chest_pain_er_2.png"]
    if all(p.exists() for p in images):
        print("\n" + "-" * 50)
        print("\n3. Multi-page PDF\n")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)
        try:
            make_sample_pdf(images, pdf_path)
            result = rag.query(
                document=str(pdf_path),
                question="What is the chief complaint or reason for visit?",
                max_tokens=400,
            )
            print(f"Answer (combined from {result.page_count} pages):\n{result.answer}\n")
            print(f"Tokens used: {result.tokens_used}\n")

            extracted_pdf = rag.extract_structured_data(
                document=str(pdf_path),
                fields=fields,
            )
            print("Extracted from PDF (merged across pages):")
            for key, value in extracted_pdf.items():
                print(f"  {key}: {value}")
        finally:
            pdf_path.unlink(missing_ok=True)
    else:
        print("\n(Skip multi-page PDF demo: need chest_pain_er_1.png and chest_pain_er_2.png)")

    print("\n" + "=" * 50)
    print("Check logs/audit.log for audit trail")


if __name__ == "__main__":
    main()
