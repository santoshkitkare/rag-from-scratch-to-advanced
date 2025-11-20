"""Ingestion package for document loading and text splitting."""
from .document_loader import DocumentLoader
from .text_splitter import TextSplitter

__all__ = ["DocumentLoader", "TextSplitter"]
