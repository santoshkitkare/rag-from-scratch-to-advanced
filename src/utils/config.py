"""
Configuration management for the RAG system.
"""
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class RAGConfig(BaseModel):
    """Configuration for RAG system parameters."""
    
    model_config = ConfigDict(extra="allow")
    
    # Document processing
    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    
    # Retrieval
    top_k: int = Field(default=4, description="Number of documents to retrieve")
    retrieval_strategy: str = Field(default="semantic", description="Retrieval strategy to use")
    
    # Generation
    temperature: float = Field(default=0.7, description="LLM temperature")
    max_tokens: int = Field(default=500, description="Maximum tokens in response")
    model_name: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    
    # Vector store
    vector_store_type: str = Field(default="faiss", description="Vector store type (faiss or chroma)")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    
    # API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
