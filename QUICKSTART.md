# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (for LLM generation)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/santoshkitkare/rag-from-scratch-to-advanced.git
cd rag-from-scratch-to-advanced
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Using the Application

### 1. Upload Documents
- Click "Browse files" or drag and drop documents
- Supported formats: PDF, TXT, MD
- Click "Process Documents" to index them

### 2. Configure Settings
Use the sidebar to adjust:
- **Chunk Size**: Size of text chunks (default: 1000)
- **Chunk Overlap**: Overlap between chunks (default: 200)
- **Retrieval Strategy**: Choose between semantic or hybrid search
- **Top K Results**: Number of documents to retrieve (default: 4)
- **Model**: Select the LLM model (default: gpt-3.5-turbo)
- **Temperature**: Control response creativity (default: 0.7)

### 3. Chat with Your Documents
- Type your question in the chat input
- Press Enter or click the send button
- View the AI-generated response with source citations
- Click "View Sources" to see the retrieved document chunks

## Example Usage

Try the included example documents:
1. Upload `examples/sample_doc.txt` (about AI and Machine Learning)
2. Upload `examples/rag_guide.md` (about RAG systems)
3. Process the documents
4. Ask questions like:
   - "What is RAG?"
   - "Explain deep learning"
   - "What are the advantages of RAG?"
   - "What challenges does AI face?"

## Retrieval Strategies

### Semantic Search
- Pure vector similarity search
- Fast and efficient
- Best for general queries

### Hybrid Search
- Combines semantic search with keyword matching (BM25)
- Better accuracy for specific terms
- Recommended for technical documents

## Tips

1. **Chunk Size**: Larger chunks preserve more context but may be less precise
2. **Top K**: More results provide more context but may include noise
3. **Temperature**: Lower values (0.0-0.3) for factual responses, higher (0.7-1.0) for creative responses
4. **Clear Chat**: Use the "Clear Chat History" button to start fresh

## Troubleshooting

### "Please provide an OpenAI API key"
- Make sure you've added your API key to the `.env` file or entered it in the sidebar

### "Please upload and process documents first"
- Upload at least one document and click "Process Documents" before asking questions

### Slow Processing
- Large documents take time to process and embed
- Consider splitting very large documents into smaller files

## Next Steps

- Try different retrieval strategies
- Experiment with chunk sizes
- Upload your own documents
- Adjust temperature and other parameters to optimize responses

## Support

For issues or questions, please check the main README.md or open an issue on GitHub.
