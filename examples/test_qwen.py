"""
Spike: Test vision models via OpenAI-compatible API
Works with: LMStudio, Ollama (with openai compatibility), Azure OpenAI, OpenAI
"""
import os
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_vision_model(image_path, base_url=None, api_key=None):
    """
    Test a vision model via OpenAI-compatible API
    
    Args:
        image_path: Path to medical chart image
        base_url: API base URL (e.g., "http://localhost:1234/v1" for LMStudio)
        api_key: API key (use "lm-studio" for LMStudio, or your actual key)
    """
    # Default to LMStudio local
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
    
    print(f"\n=== Testing Vision Model ===")
    print(f"Base URL: {base_url}")
    print(f"Image: {image_path}")
    
    # Initialize client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    # Encode image
    base64_image = encode_image(image_path)
    
    # Test with a simple question
    prompt = "What type of medical document is this? What is the patient's primary diagnosis?"
    
    try:
        response = client.chat.completions.create(
            model="qwen2-vl",  # Or whatever model you have loaded
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        print(f"\n✓ Model Response:")
        print(f"{answer}\n")
        
        # Show token usage if available
        if hasattr(response, 'usage'):
            print(f"Tokens used: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Is LMStudio running with a vision model loaded?")
        print("2. Is the API server started? (Check LMStudio settings)")
        print("3. Is the model compatible with vision? (Qwen2-VL, LLaVA, etc.)")
        return False

def test_multiple_questions(image_path):
    """Test extracting specific information"""
    questions = [
        "What is the patient's name and date of birth?",
        "What is the primary diagnosis with ICD-10 code?",
        "What medications were prescribed?",
        "When should the patient follow up?",
    ]
    
    print("\n=== Testing Information Extraction ===")
    
    base64_image = encode_image(image_path)
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "lm-studio")
    )
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. {question}")
        
        try:
            response = client.chat.completions.create(
                model="qwen2-vl",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": f"Look at this medical chart and answer: {question}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            print(f"   → {answer}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            break

if __name__ == "__main__":
    print("=== HIPAA RAG Vision Model Test ===\n")
    
    # Check for test images
    test_data_dir = Path(__file__).parent.parent / "test_data"
    
    # Test with bronchitis chart first (simpler)
    bronchitis_chart = test_data_dir / "bronchitis_chart.png"
    diabetes_chart = test_data_dir / "diabetes_chart.png"
    
    if not bronchitis_chart.exists():
        print(f"✗ Test data not found: {bronchitis_chart}")
        print("Add your test charts to test_data/ directory")
        exit(1)
    
    # Basic test
    success = test_vision_model(bronchitis_chart)
    
    if success:
        # Try extracting specific information
        test_multiple_questions(bronchitis_chart)
        
        # If we have the diabetes chart, test that too
        if diabetes_chart.exists():
            print("\n" + "="*50)
            print("Testing with more complex chart (diabetes)...")
            print("="*50)
            test_vision_model(diabetes_chart)
    
    print("\n" + "="*50)
    print("Configuration for different backends:")
    print("="*50)
    print("\nLMStudio (local):")
    print("  OPENAI_BASE_URL=http://localhost:1234/v1")
    print("  OPENAI_API_KEY=lm-studio")
    print("\nOllama (with OpenAI compatibility):")
    print("  OPENAI_BASE_URL=http://localhost:11434/v1")
    print("  OPENAI_API_KEY=ollama")
    print("\nAzure OpenAI:")
    print("  OPENAI_BASE_URL=https://your-resource.openai.azure.com/")
    print("  OPENAI_API_KEY=your-azure-key")
    print("\nOpenAI:")
    print("  OPENAI_BASE_URL=https://api.openai.com/v1")
    print("  OPENAI_API_KEY=your-openai-key")