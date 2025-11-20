"""
Basic tests for RAG system components.
"""
import pytest
from pathlib import Path
from langchain.schema import Document

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_splitter import TextSplitter
from src.retrieval.embeddings import EmbeddingModel
from src.utils.config import RAGConfig


class TestDocumentLoader:
    """Test document loading functionality."""
    
    def test_supported_extensions(self):
        """Test that supported extensions are defined."""
        assert '.pdf' in DocumentLoader.SUPPORTED_EXTENSIONS
        assert '.txt' in DocumentLoader.SUPPORTED_EXTENSIONS
        assert '.md' in DocumentLoader.SUPPORTED_EXTENSIONS
    
    def test_load_text_document(self, tmp_path):
        """Test loading a text document."""
        # Create a temporary text file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.")
        
        # Load the document
        documents = DocumentLoader.load_document(str(test_file))
        
        assert len(documents) > 0
        assert documents[0].page_content == "This is a test document."
        assert documents[0].metadata['source'] == "test.txt"
        assert documents[0].metadata['file_type'] == "text"
    
    def test_unsupported_format(self, tmp_path):
        """Test that unsupported formats raise an error."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("Content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            DocumentLoader.load_document(str(test_file))


class TestTextSplitter:
    """Test text splitting functionality."""
    
    def test_split_text(self):
        """Test splitting text into chunks."""
        text = "This is a sentence. " * 100  # Long text
        
        chunks = TextSplitter.split_text(
            text,
            chunk_size=100,
            chunk_overlap=20
        )
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 150 for chunk in chunks)  # Allow some buffer
    
    def test_split_documents(self):
        """Test splitting documents into chunks."""
        documents = [
            Document(
                page_content="This is a sentence. " * 100,
                metadata={"source": "test"}
            )
        ]
        
        chunks = TextSplitter.split_documents(
            documents,
            chunk_size=100,
            chunk_overlap=20
        )
        
        assert len(chunks) > 1
        assert all('chunk_id' in chunk.metadata for chunk in chunks)
        assert all('chunk_size' in chunk.metadata for chunk in chunks)


class TestEmbeddingModel:
    """Test embedding model functionality."""
    
    def test_initialization(self):
        """Test embedding model initialization."""
        model = EmbeddingModel()
        assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_embed_query(self):
        """Test embedding a query."""
        model = EmbeddingModel()
        embedding = model.embed_query("test query")
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_documents(self):
        """Test embedding multiple documents."""
        model = EmbeddingModel()
        texts = ["document 1", "document 2", "document 3"]
        embeddings = model.embed_documents(texts)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)


class TestRAGConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RAGConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.top_k == 4
        assert config.temperature == 0.7
        assert config.retrieval_strategy == "semantic"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RAGConfig(
            chunk_size=500,
            top_k=8,
            temperature=0.5
        )
        
        assert config.chunk_size == 500
        assert config.top_k == 8
        assert config.temperature == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
