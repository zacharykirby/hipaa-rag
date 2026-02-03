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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from a medical chart query"""
    question: str
    answer: str
    timestamp: datetime
    tokens_used: Optional[int] = None
    model: Optional[str] = None
    document_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
            "tokens_used": self.tokens_used,
            "model": self.model,
            "document_path": self.document_path
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
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def query(
        self,
        document: str,
        question: str,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> QueryResult:
        """
        Query a medical document
        
        Args:
            document: Path to medical document image
            question: Question to ask about the document
            temperature: Model temperature (lower = more focused)
            max_tokens: Maximum tokens in response
            
        Returns:
            QueryResult with answer and metadata
        """
        logger.info(f"Querying document: {document}")
        logger.info(f"Question: {question}")
        
        # Encode image
        try:
            base64_image = self._encode_image(document)
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise
        
        # Make API call
        try:
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
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            logger.info(f"Query successful. Tokens used: {tokens_used}")
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
        
        # Create result
        result = QueryResult(
            question=question,
            answer=answer,
            timestamp=datetime.now(),
            tokens_used=tokens_used,
            model=self.model,
            document_path=document
        )
        
        # Audit log
        if self.enable_audit_log:
            self._log_query(result)
        
        return result
    
    def _log_query(self, result: QueryResult):
        """Log query to audit trail"""
        log_entry = (
            f"QUERY | Document: {result.document_path} | "
            f"Question: {result.question} | "
            f"Tokens: {result.tokens_used} | "
            f"Model: {result.model}"
        )
        self.audit_logger.info(log_entry)
    
    def extract_structured_data(self,
                                document: str,
                                fields: list[str]) -> Dict[str, str]:
        """Extract specific fields from a medical document as JSON"""
    
        fields_str = ", ".join([f'"{field}"' for field in fields])
        prompt = (
            f"Extract the following information from this medical chart: {fields_str}.\n\n"
            f"Return the answer as valid JSON only, with no other text:\n"
            f'{{\n  "patient_name": "value",\n  "date_of_birth": "value",\n  '
            f'"primary_diagnosis": "value",\n  "prescribed_medications": "value"\n}}'
        )
        
        result = self.query(document, prompt, max_tokens=300, temperature=0.0)
        
        # Try to parse JSON from response
        try:
            # Sometimes model wraps in ```json, clean that
            response_text = result.answer.strip()
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            extracted = json.loads(response_text)
            logger.info(f"Extracted {len(extracted)} fields: {list(extracted.keys())}")
            return extracted
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.warning(f"Raw response: {result.answer}")
            return {}