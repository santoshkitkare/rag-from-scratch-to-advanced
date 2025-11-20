"""
Retrieval strategies for RAG.
"""
from typing import List, Optional
from enum import Enum

from langchain.schema import Document
from langchain.retrievers import (
    ContextualCompressionRetriever,
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

from .vector_store import VectorStoreManager


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    CONTEXTUAL_COMPRESSION = "contextual_compression"


class Retriever:
    """Retrieval strategies for RAG system."""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC,
        top_k: int = 4
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store_manager: Vector store manager
            strategy: Retrieval strategy to use
            top_k: Number of documents to retrieve
        """
        self.vector_store_manager = vector_store_manager
        self.strategy = strategy
        self.top_k = top_k
        self._documents_cache: Optional[List[Document]] = None
    
    def set_documents_cache(self, documents: List[Document]) -> None:
        """
        Cache documents for retrieval strategies that need them.
        
        Args:
            documents: List of documents to cache
        """
        self._documents_cache = documents
    
    def retrieve(
        self,
        query: str,
        llm=None
    ) -> List[Document]:
        """
        Retrieve documents based on the selected strategy.
        
        Args:
            query: Query string
            llm: Language model (required for some strategies)
            
        Returns:
            List of retrieved documents
        """
        if self.strategy == RetrievalStrategy.SEMANTIC:
            return self._semantic_search(query)
        
        elif self.strategy == RetrievalStrategy.HYBRID:
            return self._hybrid_search(query)
        
        elif self.strategy == RetrievalStrategy.CONTEXTUAL_COMPRESSION:
            if llm is None:
                raise ValueError("LLM required for contextual compression")
            return self._contextual_compression(query, llm)
        
        else:
            raise ValueError(f"Unknown retrieval strategy: {self.strategy}")
    
    def _semantic_search(self, query: str) -> List[Document]:
        """
        Perform semantic similarity search.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
        """
        return self.vector_store_manager.similarity_search(
            query,
            k=self.top_k
        )
    
    def _hybrid_search(self, query: str) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword-based search.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
        """
        if self._documents_cache is None:
            # Fallback to semantic search if no documents cached
            return self._semantic_search(query)
        
        # Semantic retriever
        semantic_retriever = self.vector_store_manager.get_retriever(
            search_kwargs={"k": self.top_k}
        )
        
        # BM25 retriever for keyword search
        bm25_retriever = BM25Retriever.from_documents(self._documents_cache)
        bm25_retriever.k = self.top_k
        
        # Ensemble retriever combining both
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.5, 0.5]  # Equal weights
        )
        
        return ensemble_retriever.get_relevant_documents(query)
    
    def _contextual_compression(self, query: str, llm) -> List[Document]:
        """
        Perform retrieval with contextual compression.
        
        Args:
            query: Query string
            llm: Language model for compression
            
        Returns:
            List of compressed, relevant documents
        """
        base_retriever = self.vector_store_manager.get_retriever(
            search_kwargs={"k": self.top_k * 2}  # Retrieve more, then compress
        )
        
        compressor = LLMChainExtractor.from_llm(llm)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever.get_relevant_documents(query)
