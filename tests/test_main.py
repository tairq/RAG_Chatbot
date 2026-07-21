"""Smoke tests for utility functions in app/main.py.

Tests here mock the Supabase-backed ``list_documents`` call so they
can run without a live database connection.
"""

from unittest.mock import patch

import pytest

from app.main import resolve_document_references


class TestResolveDocumentReferencesNoMentionedDocs:
    """resolve_document_references called without an explicit mentioned_docs list."""

    @patch("app.main.list_documents")
    def test_query_contains_doc_name(self, mock_list_docs):
        """When query contains a document title, it should be extracted."""
        mock_list_docs.return_value = [
            {"title": "annual_report_2025.pdf"},
            {"title": "budget.xlsx"},
        ]

        titles, query = resolve_document_references(
            "What does the annual_report_2025.pdf say about revenue?",
            None,
        )

        assert titles == ["annual_report_2025.pdf"]
        # The title should be stripped from the retrieval query
        assert "annual_report_2025.pdf" not in query
        assert "revenue" in query.lower()

    @patch("app.main.list_documents")
    def test_no_doc_match(self, mock_list_docs):
        """When query does not name any document, titles should be empty."""
        mock_list_docs.return_value = [
            {"title": "report.pdf"},
        ]

        titles, query = resolve_document_references(
            "What is the capital of France?",
            None,
        )

        assert titles is None
        assert "capital of France" in query

    @patch("app.main.list_documents")
    def test_case_insensitive_match(self, mock_list_docs):
        """Document title matching should be case-insensitive."""
        mock_list_docs.return_value = [
            {"title": "MyDoc.PDF"},
        ]

        titles, query = resolve_document_references(
            "Summarize mydoc.pdf",
            None,
        )

        assert titles == ["MyDoc.PDF"]
        assert "mydoc.pdf" not in query.lower()

    @patch("app.main.list_documents")
    def test_multiple_docs_mentioned(self, mock_list_docs):
        """Multiple document titles in one query should all be captured."""
        mock_list_docs.return_value = [
            {"title": "doc1.pdf"},
            {"title": "doc2.pdf"},
            {"title": "doc3.pdf"},
        ]

        titles, query = resolve_document_references(
            "Compare doc1.pdf and doc2.pdf",
            None,
        )

        assert len(titles) == 2
        assert "doc1.pdf" in titles
        assert "doc2.pdf" in titles
        assert "doc3.pdf" not in titles


class TestResolveDocumentReferencesWithMentionedDocs:
    """resolve_document_references called with an explicit mentioned_docs list."""

    @patch("app.main.list_documents")
    def test_mentioned_docs_merged_with_query_match(self, mock_list_docs):
        """Explicit mentioned_docs + doc name in query should be merged."""
        mock_list_docs.return_value = [
            {"title": "report.pdf"},
        ]

        titles, query = resolve_document_references(
            "Tell me about report.pdf",
            mentioned_docs=["budget.xlsx"],
        )

        assert "report.pdf" in titles
        assert "budget.xlsx" in titles
        assert len(titles) == 2

    @patch("app.main.list_documents")
    def test_mentioned_docs_no_dedup(self, mock_list_docs):
        """If a title matches both mentioned_docs and the query, it should appear once."""
        mock_list_docs.return_value = [
            {"title": "shared.pdf"},
        ]

        titles, query = resolve_document_references(
            "Analyze shared.pdf",
            mentioned_docs=["shared.pdf"],
        )

        assert titles == ["shared.pdf"]
