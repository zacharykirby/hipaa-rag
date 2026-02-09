"""Tests for core SecureRAG logic (merge, parse) without API calls."""
import json

import pytest

from hipaa_rag.core import SecureRAG


def test_parse_extraction_response_plain_json():
    rag = SecureRAG(enable_audit_log=False)
    text = '{"patient_name": "Jane", "dob": "01/01/1980"}'
    out = rag._parse_extraction_response(text)
    assert out == {"patient_name": "Jane", "dob": "01/01/1980"}


def test_parse_extraction_response_markdown_fence():
    rag = SecureRAG(enable_audit_log=False)
    text = '```json\n{"a": "b"}\n```'
    out = rag._parse_extraction_response(text)
    assert out == {"a": "b"}


def test_parse_extraction_response_json_block_with_lang():
    rag = SecureRAG(enable_audit_log=False)
    text = '```json\n{"x": 1}\n```'
    out = rag._parse_extraction_response(text)
    assert out == {"x": 1}


def test_merge_extracted_pages_first_non_empty():
    rag = SecureRAG(enable_audit_log=False)
    page_dicts = [
        {"patient_name": "", "date_of_birth": "01/15/1980"},
        {"patient_name": "Jane Doe", "date_of_birth": ""},
    ]
    merged = rag._merge_extracted_pages(page_dicts)
    assert merged["patient_name"] == "Jane Doe"
    assert merged["date_of_birth"] == "01/15/1980"


def test_merge_extracted_pages_list_like_concat_dedupe():
    rag = SecureRAG(enable_audit_log=False)
    page_dicts = [
        {"prescribed_medications": ["Aspirin", "Lisinopril"]},
        {"prescribed_medications": ["Aspirin", "Metformin"]},
    ]
    merged = rag._merge_extracted_pages(page_dicts)
    assert merged["prescribed_medications"] == [
        "Aspirin",
        "Lisinopril",
        "Metformin",
    ]


def test_merge_extracted_pages_list_like_single_value():
    rag = SecureRAG(enable_audit_log=False)
    page_dicts = [
        {"prescribed_medications": "Aspirin"},
        {"prescribed_medications": ["Metformin"]},
    ]
    merged = rag._merge_extracted_pages(page_dicts)
    assert "Aspirin" in merged["prescribed_medications"]
    assert "Metformin" in merged["prescribed_medications"]


def test_merge_extracted_pages_empty_list():
    rag = SecureRAG(enable_audit_log=False)
    page_dicts = [{"patient_name": "Jane"}, {"patient_name": "John"}]
    merged = rag._merge_extracted_pages(page_dicts)
    assert merged["patient_name"] == "Jane"  # first non-empty


def test_merge_extracted_pages_skips_non_dict():
    rag = SecureRAG(enable_audit_log=False)
    page_dicts = [{"a": "1"}, "not a dict", {"b": "2"}]
    merged = rag._merge_extracted_pages(page_dicts)
    assert merged == {"a": "1", "b": "2"}
