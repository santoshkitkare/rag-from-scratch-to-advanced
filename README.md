# RAG from Scratch to Advanced

A scalable and extensible Retrieval-Augmented Generation (RAG) system designed to evolve from traditional RAG implementations to advanced retrieval strategies. This project provides a UI to "Chat with your Documents" and serves as a playground for experimenting with different ingestion, retrieval, and generation techniques.

## Features

- 📄 **Document Ingestion**: Support for multiple document formats (PDF, TXT, DOCX, PPTX, Markdown)
- 🔍 **Multiple Retrieval Strategies**: 
  - Basic semantic search
  - Hybrid search (semantic + keyword)
  - Re-ranking capabilities
  - Contextual compression
- 💬 **Interactive Chat UI**: User-friendly Streamlit interface
- 🧠 **LLM Integration**: Support for OpenAI, Anthropic, and Cohere models
- 📊 **Vector Store Options**: FAISS and ChromaDB support
- 🎯 **Advanced Features**:
  - Query refinement
  - Source attribution
  - Configurable chunk sizes and overlap
  - Session management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/santoshkitkare/rag-from-scratch-to-advanced.git
cd rag-from-scratch-to-advanced
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
COHERE_API_KEY=your_cohere_api_key  # Optional
```

## Usage

### Running the Chat UI

```bash
streamlit run app.py
```

This will launch the web interface where you can:
1. Upload documents (PDF, TXT, DOCX, etc.)
2. Select retrieval strategy
3. Configure RAG parameters
4. Chat with your documents

### Project Structure

```
rag-from-scratch-to-advanced/
├── app.py                      # Streamlit UI application
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py  # Document loading utilities
│   │   └── text_splitter.py    # Text chunking strategies
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py     # Vector store management
│   │   ├── retriever.py        # Retrieval strategies
│   │   └── embeddings.py       # Embedding generation
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm.py              # LLM integration
│   └── utils/
│       ├── __init__.py
│       └── config.py           # Configuration management
├── tests/
│   └── test_basic.py           # Basic tests
├── examples/
│   └── sample_doc.txt          # Sample document
├── requirements.txt
├── .env.example
└── README.md
```

## Retrieval Strategies

### 1. Basic Semantic Search
Traditional vector similarity search using embeddings.

### 2. Hybrid Search
Combines semantic search with keyword-based search (BM25) for better accuracy.

### 3. Re-ranking
Applies a re-ranking model to improve the relevance of retrieved chunks.

### 4. Contextual Compression
Compresses retrieved documents to include only relevant information.

## Configuration

You can configure various parameters through the UI or programmatically:

- **Chunk Size**: Size of text chunks (default: 1000)
- **Chunk Overlap**: Overlap between chunks (default: 200)
- **Number of Results**: Top-k documents to retrieve (default: 4)
- **Temperature**: LLM temperature for generation (default: 0.7)
- **Retrieval Strategy**: Choose from available strategies

## Examples

See the `examples/` directory for sample documents and usage patterns.

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Retrieval Strategies

1. Create a new retriever class in `src/retrieval/retriever.py`
2. Implement the retrieval logic
3. Register it in the UI dropdown

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

- [ ] Multi-modal RAG (images, tables)
- [ ] Graph-based retrieval
- [ ] Query decomposition
- [ ] Retrieval evaluation metrics
- [ ] Multi-language support
- [ ] API endpoints for programmatic access

## Acknowledgments

Built with:
- LangChain for RAG orchestration
- Streamlit for UI
- Sentence-Transformers for embeddings
- FAISS/ChromaDB for vector storage