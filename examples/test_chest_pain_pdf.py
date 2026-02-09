"""
Test run using test_data/chest_pain_er.pdf — query and structured extraction.
Run from repo root with API available: python examples/test_chest_pain_pdf.py
"""
from pathlib import Path

from dotenv import load_dotenv

from hipaa_rag import SecureRAG
from hipaa_rag.loader import detect_document_type, get_page_count, get_pages

load_dotenv()

PDF_PATH = Path(__file__).parent.parent / "test_data" / "chest_pain_er.pdf"


def main():
    # 1. Loader sanity check (no RAG yet — clean output order)
    print("=== Test: chest_pain_er.pdf ===\n")
    print("1. Loader check")
    print(f"   Path: {PDF_PATH}")
    print(f"   Exists: {PDF_PATH.exists()}")
    if not PDF_PATH.exists():
        print("   ERROR: PDF not found.")
        return
    doc_type = detect_document_type(PDF_PATH)
    page_count = get_page_count(PDF_PATH)
    print(f"   Type: {doc_type}")
    print(f"   Page count: {page_count}")
    pages_list = list(get_pages(PDF_PATH))
    print(f"   get_pages() yielded: {len(pages_list)} page(s)\n")

    # 2. Initialize RAG and run query
    print("2. Query (all pages)")
    rag = SecureRAG(model="qwen2-vl", enable_audit_log=True)
    result = rag.query(
        document=str(PDF_PATH),
        question="What is the chief complaint or reason for visit? Summarize key findings.",
        max_tokens=500,
    )
    print(f"   Pages processed: {result.page_count}")
    print(f"   Tokens used: {result.tokens_used}")
    print("\n   Answer:")
    print("   " + "\n   ".join(result.answer.strip().split("\n")))

    # 3. Structured extraction
    print("\n3. Structured extraction (merged across pages)")
    fields = [
        "patient_name",
        "date_of_birth",
        "chief_complaint",
        "primary_diagnosis",
        "prescribed_medications",
    ]
    extracted = rag.extract_structured_data(
        document=str(PDF_PATH),
        fields=fields,
    )
    for key, value in extracted.items():
        print(f"   {key}: {value}")

    print("\n=== Done. Check logs/audit.log for audit trail. ===")


if __name__ == "__main__":
    main()
