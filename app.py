"""
Streamlit application for RAG system - Chat with Your Documents
"""
import os
import streamlit as st
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_splitter import TextSplitter
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.retriever import Retriever, RetrievalStrategy
from src.generation.llm import LLMManager
from src.utils.config import RAGConfig

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG: Chat with Your Documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "config" not in st.session_state:
    st.session_state.config = RAGConfig()


def initialize_rag_system(config: RAGConfig):
    """Initialize the RAG system components."""
    try:
        # Initialize embedding model
        embedding_model = EmbeddingModel(model_name=config.embedding_model)
        
        # Initialize vector store manager
        vector_store_manager = VectorStoreManager(
            embeddings=embedding_model.get_embeddings(),
            store_type=config.vector_store_type,
            persist_directory="./vector_store"
        )
        
        return embedding_model, vector_store_manager
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None


def process_documents(uploaded_files, config: RAGConfig):
    """Process uploaded documents."""
    all_documents = []
    
    # Create temporary directory for uploads
    upload_dir = Path("./uploads")
    upload_dir.mkdir(exist_ok=True)
    
    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            # Save uploaded file
            file_path = upload_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Load document
                documents = DocumentLoader.load_document(str(file_path))
                all_documents.extend(documents)
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        
        if all_documents:
            # Split documents into chunks
            chunked_docs = TextSplitter.split_documents(
                all_documents,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            
            st.success(f"Processed {len(uploaded_files)} files into {len(chunked_docs)} chunks")
            return chunked_docs
        
    return []


def query_rag_system(query: str, config: RAGConfig):
    """Query the RAG system."""
    if st.session_state.vector_store is None:
        return "Please upload and process documents first."
    
    try:
        # Initialize components
        embedding_model, vector_store_manager = initialize_rag_system(config)
        if vector_store_manager is None:
            return "Error initializing RAG system."
        
        # Load vector store from session state
        vector_store_manager.vector_store = st.session_state.vector_store
        
        # Initialize retriever
        retriever = Retriever(
            vector_store_manager=vector_store_manager,
            strategy=RetrievalStrategy(config.retrieval_strategy),
            top_k=config.top_k
        )
        
        # Set documents cache for hybrid search
        if st.session_state.documents:
            retriever.set_documents_cache(st.session_state.documents)
        
        # Initialize LLM
        llm_manager = LLMManager(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.openai_api_key
        )
        
        # Retrieve documents
        retrieved_docs = retriever.retrieve(query)
        
        # Generate response
        result = llm_manager.generate_response(query, retrieved_docs)
        
        return result
        
    except Exception as e:
        return f"Error querying RAG system: {str(e)}"


# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# API Key input
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=os.getenv("OPENAI_API_KEY", ""),
    help="Enter your OpenAI API key"
)
st.session_state.config.openai_api_key = api_key if api_key else None

st.sidebar.markdown("---")

# Document processing settings
st.sidebar.subheader("📄 Document Processing")
chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100,
    help="Maximum size of text chunks"
)
st.session_state.config.chunk_size = chunk_size

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=500,
    value=200,
    step=50,
    help="Overlap between consecutive chunks"
)
st.session_state.config.chunk_overlap = chunk_overlap

st.sidebar.markdown("---")

# Retrieval settings
st.sidebar.subheader("🔍 Retrieval Settings")
retrieval_strategy = st.sidebar.selectbox(
    "Retrieval Strategy",
    options=["semantic", "hybrid"],
    index=0,
    help="Strategy for document retrieval"
)
st.session_state.config.retrieval_strategy = retrieval_strategy

top_k = st.sidebar.slider(
    "Top K Results",
    min_value=1,
    max_value=10,
    value=4,
    step=1,
    help="Number of documents to retrieve"
)
st.session_state.config.top_k = top_k

vector_store_type = st.sidebar.selectbox(
    "Vector Store",
    options=["faiss", "chroma"],
    index=0,
    help="Type of vector store to use"
)
st.session_state.config.vector_store_type = vector_store_type

st.sidebar.markdown("---")

# Generation settings
st.sidebar.subheader("🤖 Generation Settings")
model_name = st.sidebar.selectbox(
    "Model",
    options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
    index=0,
    help="Language model to use"
)
st.session_state.config.model_name = model_name

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Creativity of responses"
)
st.session_state.config.temperature = temperature

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=100,
    max_value=2000,
    value=500,
    step=100,
    help="Maximum length of response"
)
st.session_state.config.max_tokens = max_tokens

# Main content
st.title("📚 RAG: Chat with Your Documents")
st.markdown("Upload documents and ask questions about their content using advanced retrieval techniques.")

# Document upload section
st.header("📤 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    help="Upload PDF, TXT, or Markdown files"
)

if uploaded_files:
    if st.button("🔄 Process Documents", type="primary"):
        processed_docs = process_documents(uploaded_files, st.session_state.config)
        
        if processed_docs:
            # Store documents in session state
            st.session_state.documents = processed_docs
            
            # Initialize RAG components
            embedding_model, vector_store_manager = initialize_rag_system(st.session_state.config)
            
            if vector_store_manager:
                with st.spinner("Creating vector store..."):
                    vector_store_manager.create_vector_store(processed_docs)
                    st.session_state.vector_store = vector_store_manager.vector_store
                    st.success("✅ Documents processed and indexed successfully!")

# Chat interface
st.header("💬 Chat Interface")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("📖 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f"```\n{source['content']}\n```")
                        if source.get("metadata"):
                            st.caption(f"Metadata: {source['metadata']}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if API key is provided
    if not st.session_state.config.openai_api_key:
        st.error("⚠️ Please provide an OpenAI API key in the sidebar.")
    elif st.session_state.vector_store is None:
        st.error("⚠️ Please upload and process documents first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_rag_system(prompt, st.session_state.config)
                
                if isinstance(result, dict):
                    response = result["answer"]
                    sources = result.get("sources", [])
                else:
                    response = str(result)
                    sources = []
                
                st.markdown(response)
                
                # Display sources
                if sources:
                    with st.expander("📖 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(f"```\n{source['content']}\n```")
                            if source.get("metadata"):
                                st.caption(f"Metadata: {source['metadata']}")
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using LangChain, Streamlit, and Sentence-Transformers</p>
    </div>
    """,
    unsafe_allow_html=True
)
