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

## Use Cses

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

## Disclaimer

**This is infrastructure software, not medical advice or a medical device.** 

Organizations using this framework are responsible for:
- Ensuring HIPAA compliance in their specific deployment
- Obtaining necessary Business Associate Agreements (BAAs)
- Proper access controls and security measures
- Validation for their specific use cases

This framework provides tools to help maintain compliance, but does not guarantee it.

---
