"""
Embedding generation utilities.
"""
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings


class EmbeddingModel:
    """Wrapper for embedding models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embeddings
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Query embedding
        """
        return self.embeddings.embed_query(text)
    
    def get_embeddings(self) -> Embeddings:
        """
        Get the underlying embeddings object.
        
        Returns:
            LangChain Embeddings object
        """
        return self.embeddings
