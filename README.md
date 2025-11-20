# Rag From Scratch to Advanced 🧩🤖

A scalable and extensible **Retrieval-Augmented Generation (RAG)** system designed to evolve from traditional RAG implementations to advanced retrieval strategies. This project provides a UI to "Chat with your Documents" and serves as a playground for experimenting with different ingestion, retrieval, and generation techniques.

## 📖 Overview

This repository demonstrates the complete lifecycle of a RAG application. Currently, it implements a **Traditional RAG** architecture using **FAISS** for vector storage and **Streamlit** for the frontend. The codebase is designed to be modular, allowing for the future integration of advanced features like hybrid search, re-ranking, and agentic workflows.

## 🌟 Features (Phase 1: Traditional RAG)

* **Multi-Format Ingestion:** Distinct processing pipelines for `.pdf`, `.docx`, and `.txt` files.
* **Vector Storage:** Efficient similarity search using **FAISS** (Facebook AI Similarity Search).
* **Transparent Retrieval:** The UI displays not just the answer, but the **Matching Context** (source chunks) used to derive that answer.
* **Interactive UI:** A clean, user-friendly chat interface built with **Streamlit**.
* **Metadata Management:** Maintains document metadata for source tracking.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangChain (Core, Community)
* **Vector Database:** FAISS (CPU)
* **Embeddings:** Sentence Transformers / OpenAI
* **Language Model:** OpenAI / LangChain
* **PDF/Doc Processing:** PyPDF, PyMuPDF, docx2txt
* **Dependency Management:** uv

## 📂 Project Structure

```text
|──TraditionalRag/
   ├── streamlit_app.py    # Frontend UI: Handles file upload and chat interaction
   ├── llm_prompts.py    # Backend: LLM Prompt
   ├── data_ingestion.py   # Backend: Modular pipeline for loading, chunking, and embedding
   ├── .env                # API Keys (not committed)
├── pyproject.toml      # Project metadata and dependencies (uv)
└── README.md           # Project documentation
```


## 🚀 Setup & Installation
This project uses uv for lightning-fast dependency management.

### 1. Install uv (if not already installed)
```bash
# On macOS/Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"

# Or simply via pip
pip install uv
```
### 2. Clone the repository
```bash
git clone https://github.com/santoshkitkare/rag-from-scratch-to-advanced.git
cd rag-from-scratch-to-advanced
```

### 3. Install Dependencies
Run the following command to create a virtual environment and install all dependencies defined in pyproject.toml:
```bash
uv sync
```

### 4. Configure Environment Variables
Create a .env file in the TraditionalRag directory to store your API keys:
```bash
cd TraditionalRag
OPENAI_API_KEY=your_api_key_here
```

### 5. Run the Application
You can run the Streamlit app directly using uv:
```bash
uv run streamlit run streamlit_app.py
```

## 🔮 Roadmap: Moving to Advanced RAG
This repository is actively being upgraded. The following modules are planned:

*  Multi-Document Persistence: Query across a library of uploaded documents rather than just the latest file.

* Advanced Chunking Strategies: Moving from fixed-size overlap to Semantic Chunking and Recursive Character Splitting.

* Memory/Conversational Awareness: Enabling the LLM to recall previous turns in the chat.

* Hybrid Search: Combining dense vector retrieval with sparse keyword search (BM25).

* Reranking: Adding a Cross-Encoder step to re-rank retrieved results for higher accuracy.

## 🙌 Acknowledgements
A huge thank you to Krish Naik for his educational resources on Traditional RAG. This project's foundational logic was inspired by his video tutorials: https://www.youtube.com/watch?v=o126p1QN_RI&t=46s

## 📝 License
This project is open source and available under the MIT License.