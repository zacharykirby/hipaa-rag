"""HIPAA-compliant RAG framework for medical documents"""

from .core import SecureRAG, QueryResult

__version__ = "0.1.0"
__all__ = ["SecureRAG", "QueryResult"]