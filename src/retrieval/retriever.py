"""
Retrieval strategies for RAG.
"""
from typing import List, Optional
from enum import Enum

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from .vector_store import VectorStoreManager


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


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
            llm: Language model (optional, for future use)
            
        Returns:
            List of retrieved documents
        """
        if self.strategy == RetrievalStrategy.SEMANTIC:
            return self._semantic_search(query)
        
        elif self.strategy == RetrievalStrategy.HYBRID:
            return self._hybrid_search(query)
        
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
        
        # Get semantic results
        semantic_docs = self._semantic_search(query)
        
        # Get BM25 results
        bm25_retriever = BM25Retriever.from_documents(self._documents_cache)
        bm25_retriever.k = self.top_k
        bm25_docs = bm25_retriever.get_relevant_documents(query)
        
        # Combine results (simple merge, remove duplicates)
        seen_content = set()
        combined_docs = []
        
        for doc in semantic_docs + bm25_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                combined_docs.append(doc)
                if len(combined_docs) >= self.top_k:
                    break
        
        return combined_docs[:self.top_k]
