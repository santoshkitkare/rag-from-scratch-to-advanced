"""
Document loading utilities for various file formats.
"""
import os
from typing import List
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)


class DocumentLoader:
    """Load documents from various file formats."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.txt': 'text',
        '.md': 'markdown',
    }
    
    @classmethod
    def load_document(cls, file_path: str) -> List[Document]:
        """
        Load a document from a file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file format is not supported
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(cls.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        file_type = cls.SUPPORTED_EXTENSIONS[extension]
        
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type == 'text':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_type == 'markdown':
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"No loader configured for file type: {file_type}")
        
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata['source'] = str(path.name)
            doc.metadata['file_type'] = file_type
        
        return documents
    
    @classmethod
    def load_documents_from_directory(cls, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of Document objects from all files
        """
        all_documents = []
        directory = Path(directory_path)
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                try:
                    documents = cls.load_document(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return all_documents
