# HIPAA Compliant RAG Framework
> Build AI applications with Protected Health Information (PHI) without sending data to third-party APIs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Early Development](https://img.shields.io/badge/status-early%20development-orange)](https://github.com/zacharykirby/hipaa-rag)

## The Problem
Healthcare organizations want to use LLMs for medical chart analysis, clinical decision support, and documentation. But sending Protected Health Information (PHI) to third-party API providers creates compliance risk and liability.

**This framework aims to solve that.**

---

## Key Features

- **HIPAA-first design** - Privacy and compliance built first
- **Local or cloud** - Run models on-prem OR use HIPAA-compliant cloud providers (Azure/AWS/GCP with BAAs)
- **Hybrid Mode** - Route PHI to local models, non-PHI to cloud for optimal cost/performance
- **Audit Logging** - Track all data access for compliance
- **Modular Architecture** - Swap models and backends without rewriting applications

---

## Use Cases

- Medical chart analysis and summarization
- Clinical evidence extraction for quality measures
- Prior authorization document processing
- Patient communication automation
- Research data analysis with sensitive health information

---

## Architecture

```
User Input
    ↓
Framework (Data Router)
    ↓
    ├─→ PHI? → Local Model (Qwen, Llama, etc.)
    └─→ Non-PHI? → Cloud API (Azure OpenAI, AWS Bedrock)
    ↓
Audit Log
    ↓
Response
```

**Deployment options:**
- **Local-only**: Maximum privacy, all processing on-premises
- **Cloud-only**: Leverage HIPAA-compliant cloud services with BAAs
- **Hybrid**: Intelligent routing based on data sensitivity

---

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **LMStudio** (for local models)
   - Download from [https://lmstudio.ai](https://lmstudio.ai)
   - Load a vision model (recommended: Qwen3-VL 7B)
   - Start the local API server (Settings → Developer → Start Server)

### Installation

```bash
# Clone the repository
git clone https://github.com/zacharykirby/hipaa-rag.git
cd hipaa-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```bash
# For LMStudio (local)
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio

# Or for a remote LMStudio instance
# OPENAI_BASE_URL=http://192.168.0.X:1234/v1
```

---

## Usage

### Simple Query

```python
from hipaa_rag import SecureRAG

# Initialize framework
rag = SecureRAG(model="qwen3-vl")

# Query a medical chart
result = rag.query(
    document="path/to/medical_chart.png",
    question="What is the patient's primary diagnosis?"
)

print(result.answer)
print(f"Tokens used: {result.tokens_used}")
```

### Extract Structured Data

```python
from hipaa_rag import SecureRAG

rag = SecureRAG(model="qwen3-vl")

# Extract specific fields
data = rag.extract_structured_data(
    document="path/to/medical_chart.png",
    fields=[
        "patient_name",
        "date_of_birth",
        "primary_diagnosis",
        "prescribed_medications"
    ]
)

# Returns dictionary:
# {
#     "patient_name": "Jane Doe",
#     "date_of_birth": "01/15/1980",
#     "primary_diagnosis": "Acute Bronchitis (J20.9)",
#     "prescribed_medications": "Tessalon Perles 100mg TID"
# }
```

### Audit Logging

All queries are automatically logged for HIPAA compliance:

```python
rag = SecureRAG(
    model="qwen3-vl",
    enable_audit_log=True,
    audit_log_path="logs/audit.log"
)

# Queries are logged with:
# - Timestamp
# - Document accessed
# - Question asked
# - Model used
# - Tokens consumed
```

Check `logs/audit.log` for the audit trail.

---

## Run the Demo

```bash
python examples/demo_framework.py
```

This will:
1. Query a test medical chart
2. Extract structured data
3. Generate an audit log

---

## Current Status

🚧 **Phase 1: MVP Development** (In Progress)

**Working:**
- ✅ Local model integration (Qwen2-VL via LMStudio)
- ✅ Medical chart Q&A
- ✅ Structured data extraction (JSON)
- ✅ Audit logging
- ✅ OpenAI-compatible API abstraction

**Roadmap:**
- **Phase 1 (Current)**: Local-only deployment, basic functionality
- **Phase 2**: Cloud integration (Azure OpenAI, AWS Bedrock with BAAs)
- **Phase 3**: Hybrid mode, production hardening, comprehensive docs

---

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (coming soon)
pytest

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

---

## Project Structure

```
hipaa-rag/
├── src/hipaa_rag/       # Core framework
│   ├── core.py          # SecureRAG class
│   └── __init__.py
├── examples/            # Usage examples
│   └── demo_framework.py
├── test_data/          # Sample medical charts
├── tests/              # Test suite
├── logs/               # Audit logs (generated)
└── docs/               # Documentation
```

---

# Test Data

This directory contains AI-generated medical charts for testing and demonstration purposes.

**These are NOT real patient records.**

- All patient names, dates, and medical information are fictional
- Generated using AI image generation tools (Google Gemini nano-banana)
- Used to demonstrate the framework's capabilities

If you're testing with real medical data, create a separate directory (e.g., `data/`) which is excluded from version control.

---

## Disclaimer

**This is infrastructure software, not medical advice or a medical device.** 

Organizations using this framework are responsible for:
- Ensuring HIPAA compliance in their specific deployment
- Obtaining necessary Business Associate Agreements (BAAs)
- Proper access controls and security measures
- Validation for their specific use cases

This framework provides tools to help maintain compliance, but does not guarantee it.

---

## License

MIT License - see [LICENSE](./LICENSE) file for details.

---

## Contact

Built by Zach - Healthcare AI Engineer

Questions? Open an issue or reach out directly.