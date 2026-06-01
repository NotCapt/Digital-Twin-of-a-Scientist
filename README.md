# 🧠 Digital Twin of Alan Turing

An interactive AI application that embodies the knowledge, reasoning style, and personality of Alan Mathison Turing. Built as a demonstration of advanced Retrieval-Augmented Generation (RAG) and long-term memory systems.

## 🌟 Features

- **Persona Emulation**: Prompt engineering that deeply captures Turing's voice—scholarly, occasionally hesitant, deeply analytical, and unapologetically visionary. The agent fully embodies Turing and will never break character.
- **RAG Knowledge Base**: Responses are grounded in Turing's actual writings, papers (like *On Computable Numbers* and *The Chemical Basis of Morphogenesis*), historical letters, and comprehensively scraped Wikipedia articles covering his life and work.
- **Dual Memory System**:
  - *Short-term (Episodic)*: Remembers the context of the current conversation session.
  - *Long-term (Semantic)*: Learns facts across multiple sessions and persists them using ChromaDB.
- **Timeline Engine**: Awareness of Turing's historical context. Ask a question "as if it's 1940" and the Twin restricts its knowledge to that era.
- **Memory Dashboard**: A visual representation built natively in Streamlit showing what the Digital Twin has "learned" about the user over time.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Google Gemini API Key (`GEMINI_API_KEY`)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   Copy `.env.example` to `.env` and add your Gemini API key.
   ```bash
   cp .env.example .env
   ```

### Data Ingestion

Before running the application, you must populate the RAG knowledge base. 
First, run the automated scraper to fetch Wikipedia articles and save them to `data/raw/papers`:

```bash
python scripts/scrape_sources.py
```

Then, run the ingestion script to chunk the files and embed them into the ChromaDB vector database:

```bash
python scripts/ingest_documents.py
```

### Running the App

Start the Streamlit interface:

```bash
streamlit run streamlit_app.py --server.port 8501
```
Then navigate to `http://localhost:8501` in your browser.

## 🏗️ Architecture

- `core/`: The central orchestrator and prompt builders.
- `rag/`: Document chunking, embedding, and retrieval systems.
- `memory/`: Short-term and long-term memory management.
- `persona/`: System prompts and response validators.
- `timeline/`: Historical context and era-specific knowledge gating.
- `dashboard/`: Visualizations for the memory and timeline states.
- `scripts/`: Utilities for data ingestion and data scraping.

## 🧪 Testing

Run the test suite using pytest:

```bash
python -m pytest tests/
```

## 📜 License
MIT License
