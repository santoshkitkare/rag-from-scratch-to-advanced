"""
Vector store management for document storage and retrieval.
"""
import os
from typing import List, Optional
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain.embeddings.base import Embeddings


class VectorStoreManager:
    """Manage vector stores for document retrieval."""
    
    def __init__(
        self,
        embeddings: Embeddings,
        store_type: str = "faiss",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the vector store manager.
        
        Args:
            embeddings: Embeddings model to use
            store_type: Type of vector store ("faiss" or "chroma")
            persist_directory: Directory to persist the vector store
        """
        self.embeddings = embeddings
        self.store_type = store_type.lower()
        self.persist_directory = persist_directory
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of documents to index
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")
        
        if self.store_type == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Save if persist directory is specified
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
                save_path = os.path.join(self.persist_directory, "faiss_index")
                self.vector_store.save_local(save_path)
                
        elif self.store_type == "chroma":
            if not self.persist_directory:
                self.persist_directory = "./chroma_db"
            
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
    
    def load_vector_store(self) -> None:
        """Load an existing vector store from disk."""
        if not self.persist_directory:
            raise ValueError("No persist directory specified")
        
        if self.store_type == "faiss":
            load_path = os.path.join(self.persist_directory, "faiss_index")
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"FAISS index not found at {load_path}")
            
            self.vector_store = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
        elif self.store_type == "chroma":
            if not os.path.exists(self.persist_directory):
                raise FileNotFoundError(f"Chroma DB not found at {self.persist_directory}")
            
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of documents to add
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create or load one first.")
        
        self.vector_store.add_documents(documents)
        
        # Persist if using FAISS
        if self.store_type == "faiss" and self.persist_directory:
            save_path = os.path.join(self.persist_directory, "faiss_index")
            self.vector_store.save_local(save_path)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Query string
            k: Number of documents to return
            **kwargs: Additional arguments for the search
            
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search(query, k=k, **kwargs)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Query string
            k: Number of documents to return
            **kwargs: Additional arguments for the search
            
        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search_with_score(query, k=k, **kwargs)
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever interface for the vector store.
        
        Args:
            search_kwargs: Keyword arguments for search
            
        Returns:
            Retriever object
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
