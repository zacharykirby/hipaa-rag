"""
Core framework for HIPAA-compliant RAG on medical documents
"""
import os
import json
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from openai import OpenAI

from .loader import get_pages, get_page_count

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Field names treated as list-like (concat + dedupe when merging multi-page extraction)
LIST_LIKE_FIELDS = frozenset(
    k.lower()
    for k in (
        "medications",
        "prescribed_medications",
        "diagnoses",
        "allergies",
        "conditions",
        "problems",
    )
)


@dataclass
class QueryResult:
    """Result from a medical chart query"""

    question: str
    answer: str
    timestamp: datetime
    tokens_used: Optional[int] = None
    model: Optional[str] = None
    document_path: Optional[str] = None
    page_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
            "tokens_used": self.tokens_used,
            "model": self.model,
            "document_path": self.document_path,
            "page_count": self.page_count,
        }


class SecureRAG:
    """
    HIPAA-compliant RAG framework for medical documents
    
    Supports multiple backends via OpenAI-compatible API:
    - LMStudio (local)
    - Ollama (local)
    - Azure OpenAI (cloud with BAA)
    - OpenAI (cloud)
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "qwen2-vl",
        enable_audit_log: bool = True,
        audit_log_path: Optional[str] = None
    ):
        """
        Initialize SecureRAG
        
        Args:
            base_url: API endpoint (defaults to env OPENAI_BASE_URL)
            api_key: API key (defaults to env OPENAI_API_KEY)
            model: Model name to use
            enable_audit_log: Whether to log all queries for compliance
            audit_log_path: Path to audit log file
        """
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "lm-studio")
        self.model = model
        self.enable_audit_log = enable_audit_log
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        # Setup audit logging
        if self.enable_audit_log:
            if audit_log_path is None:
                audit_log_path = "logs/audit.log"
            
            self.audit_log_path = Path(audit_log_path)
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create audit logger
            self.audit_logger = logging.getLogger("hipaa_rag.audit")
            self.audit_logger.setLevel(logging.INFO)
            
            # File handler for audit log
            handler = logging.FileHandler(self.audit_log_path)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.audit_logger.addHandler(handler)
            
            logger.info(f"Audit logging enabled: {self.audit_log_path}")
        
        logger.info(f"SecureRAG initialized with base_url={self.base_url}, model={self.model}")
    
    def _query_single_image(
        self,
        base64_image: str,
        question: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> tuple[str, Optional[int]]:
        """Call vision API with one image. Returns (answer, tokens_used)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = response.choices[0].message.content
        tokens_used = (
            response.usage.total_tokens if hasattr(response, "usage") else None
        )
        return (answer or "", tokens_used)

    def query(
        self,
        document: str,
        question: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
        max_pages: Optional[int] = None,
    ) -> QueryResult:
        """
        Query a medical document (image, PDF, or multi-page TIFF).

        Args:
            document: Path to document (PNG, JPEG, PDF, TIFF, etc.)
            question: Question to ask about the document
            temperature: Model temperature (lower = more focused)
            max_tokens: Maximum tokens per page response
            max_pages: Optional cap on pages processed (e.g. for very large PDFs)

        Returns:
            QueryResult with combined answer and metadata
        """
        document_path = str(Path(document).resolve())
        logger.info(f"Querying document: {document_path}")
        logger.info(f"Question: {question}")

        page_answers: list[str] = []
        total_tokens = 0

        for page_idx, png_bytes in get_pages(document_path, max_pages=max_pages):
            base64_image = base64.b64encode(png_bytes).decode("utf-8")
            try:
                answer, tokens_used = self._query_single_image(
                    base64_image,
                    question,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                page_answers.append(answer)
                if tokens_used is not None:
                    total_tokens += tokens_used
                logger.debug(f"Page {page_idx + 1} done. Tokens: {tokens_used}")
            except Exception as e:
                logger.error(f"Query failed on page {page_idx + 1}: {e}")
                raise

        if not page_answers:
            raise ValueError(f"No pages processed for document: {document_path}")

        # Combine answers
        if len(page_answers) == 1:
            combined = page_answers[0]
        else:
            parts = [
                f"--- Page {i + 1} ---\n{a}"
                for i, a in enumerate(page_answers)
                if (a and a.strip())
            ]
            combined = "\n\n".join(parts)

        page_count = len(page_answers)
        logger.info(f"Query successful. Pages: {page_count}, Tokens: {total_tokens}")

        result = QueryResult(
            question=question,
            answer=combined,
            timestamp=datetime.now(),
            tokens_used=total_tokens or None,
            model=self.model,
            document_path=document_path,
            page_count=page_count,
        )

        if self.enable_audit_log:
            self._log_query(result)

        return result
    
    def _log_query(self, result: QueryResult):
        """Log query to audit trail (one entry per document)."""
        doc_part = result.document_path or ""
        if result.page_count is not None and result.page_count > 1:
            doc_part = f"{doc_part} ({result.page_count} pages)"
        log_entry = (
            f"QUERY | Document: {doc_part} | "
            f"Question: {result.question} | "
            f"Tokens: {result.tokens_used} | "
            f"Model: {result.model}"
        )
        self.audit_logger.info(log_entry)

    def _parse_extraction_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from model response (strip markdown code fence if present)."""
        text = response_text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
        text = text.strip()
        return json.loads(text)

    def _merge_extracted_pages(
        self, page_dicts: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge per-page extraction dicts: first non-empty for scalars, concat+dedupe for list-like."""
        merged: Dict[str, Any] = {}
        for d in page_dicts:
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                key_lower = k.lower()
                if key_lower in LIST_LIKE_FIELDS:
                    # List-like: collect all values, dedupe
                    if not isinstance(v, list):
                        v = [v] if v not in (None, "") else []
                    existing = merged.get(k)
                    if existing is None:
                        merged[k] = list(dict.fromkeys(str(x).strip() for x in v if x))
                    else:
                        if not isinstance(existing, list):
                            existing = [existing]
                        merged[k] = list(
                            dict.fromkeys(
                                existing + [str(x).strip() for x in v if x]
                            )
                        )
                else:
                    # First non-empty wins
                    if k not in merged or not merged[k]:
                        if v is not None and str(v).strip():
                            merged[k] = v
        return merged

    def extract_structured_data(
        self,
        document: str,
        fields: list[str],
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract specific fields from a medical document (image, PDF, or TIFF).
        Multi-page: merges per-page JSON (first non-empty for scalars, concat+dedupe for list-like fields).
        """
        fields_str = ", ".join([f'"{f}"' for f in fields])
        example = ",\n  ".join([f'"{f}": "value"' for f in fields])
        prompt = (
            f"Extract the following information from this medical chart: {fields_str}.\n\n"
            f"Return the answer as valid JSON only, with no other text:\n"
            f"{{\n  {example}\n}}"
        )

        document_path = str(Path(document).resolve())
        page_dicts: list[Dict[str, Any]] = []

        for page_idx, png_bytes in get_pages(document_path, max_pages=max_pages):
            base64_image = base64.b64encode(png_bytes).decode("utf-8")
            answer, _ = self._query_single_image(
                base64_image,
                prompt,
                max_tokens=300,
                temperature=0.0,
            )
            try:
                parsed = self._parse_extraction_response(answer)
                page_dicts.append(parsed)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse JSON from page {page_idx + 1}: {e}. Raw: {answer[:200]}"
                )

        if not page_dicts:
            return {}

        merged = self._merge_extracted_pages(page_dicts)
        logger.info(
            f"Extracted {len(merged)} fields from {len(page_dicts)} page(s): {list(merged.keys())}"
        )

        # Audit: one entry for the whole document
        if self.enable_audit_log:
            page_count = len(page_dicts)
            doc_part = f"{document_path} ({page_count} pages)" if page_count > 1 else document_path
            self.audit_logger.info(
                f"EXTRACT | Document: {doc_part} | Fields: {list(merged.keys())} | Model: {self.model}"
            )

        return merged