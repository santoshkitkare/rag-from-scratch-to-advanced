"""
Text splitting utilities for chunking documents.
"""
from typing import List

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_core.documents import Document


class TextSplitter:
    """Split text into chunks using various strategies."""
    
    @staticmethod
    def split_documents(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive"
    ) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            strategy: Splitting strategy ("recursive" or "character")
            
        Returns:
            List of chunked documents
        """
        if strategy == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        elif strategy == "character":
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
        else:
            raise ValueError(f"Unknown splitting strategy: {strategy}")
        
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        return chunks
    
    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive"
    ) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            strategy: Splitting strategy ("recursive" or "character")
            
        Returns:
            List of text chunks
        """
        if strategy == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        elif strategy == "character":
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
        else:
            raise ValueError(f"Unknown splitting strategy: {strategy}")
        
        return text_splitter.split_text(text)
