"""
Demo of the SecureRAG framework
"""
from pathlib import Path
from hipaa_rag.core import SecureRAG
from dotenv import load_dotenv
load_dotenv()

def main():
    print("=== SecureRAG Framework Demo ===\n")
    
    # Initialize framework
    rag = SecureRAG(
        model="qwen2-vl",
        enable_audit_log=True
    )
    
    # Path to test data
    test_data = Path(__file__).parent.parent / "test_data"
    bronchitis_chart = test_data / "bronchitis_chart.png"
    
    print("1. Simple Query\n")
    result = rag.query(
        document=str(bronchitis_chart),
        question="What is the patient's primary diagnosis with ICD-10 code?"
    )
    print(f"Answer: {result.answer}\n")
    print(f"Tokens used: {result.tokens_used}\n")
    
    print("-" * 50)
    print("\n2. Extract Structured Data\n")
    
    fields = [
        "patient_name",
        "date_of_birth",
        "primary_diagnosis",
        "prescribed_medications"
    ]
    
    extracted = rag.extract_structured_data(
        document=str(bronchitis_chart),
        fields=fields
    )
    
    print("Extracted fields:")
    for key, value in extracted.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Check logs/audit.log for audit trail")

if __name__ == "__main__":
    main()