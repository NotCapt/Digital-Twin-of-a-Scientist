# 🧠 Digital Twin of Alan Turing

An interactive AI application that precisely embodies the knowledge, reasoning style, and personality of Alan Mathison Turing. Built as an advanced demonstration of Retrieval-Augmented Generation (RAG), persistent long-term memory, and persona-driven prompt engineering.

![Streamlit Framework](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge)
![Google Gemini](https://img.shields.io/badge/Gemini_1.5_Flash-AI-blueviolet?style=for-the-badge)

---

## 🌟 Core Features

- **Unbreakable Persona Emulation**: Advanced prompt engineering deeply captures Turing's unique voice—scholarly, occasionally hesitant, deeply analytical, and unapologetically visionary. The agent is strictly prompted to never break character or admit to being a digital construct.
- **RAG Knowledge Base**: Turing's responses are dynamically grounded in his actual writings. The system uses a vector database containing chunked text from his seminal papers (*On Computable Numbers*, *The Chemical Basis of Morphogenesis*), his historical letters, and comprehensively scraped Wikipedia encyclopedic data.
- **Dual Memory System**:
  - *Short-term (Episodic)*: Tracks the immediate context of the current conversation session using a rolling window.
  - *Long-term (Semantic)*: Automatically extracts facts about the user during conversation and persists them in a secondary vector space, allowing the Twin to "remember" you across multiple different sessions.
- **Dynamic Memory Dashboard**: A visual metrics dashboard built natively in Streamlit, showing exactly what the Digital Twin has "learned" about the user over time, exposing the inner workings of the long-term memory system.
- **Timeline Engine**: Imbues the Twin with awareness of his historical context. If you ask a question "as if it's 1940", the engine instructs the LLM to restrict its knowledge strictly to events before that era.

---

## 🏗️ System Architecture

The application is built entirely in Python, orchestrated by a central pipeline that manages context before calling the LLM. 

### 1. The RAG Pipeline (`rag/`)
When the user sends a message, the `Retriever` queries a local **ChromaDB** collection (`turing_knowledge`) using cosine similarity against the `all-MiniLM-L6-v2` embedding model. It fetches the top-k most relevant historical chunks from Turing's papers or Wikipedia, passing them to the Orchestrator to ground the LLM's response.

### 2. The Memory Manager (`memory/`)
The memory system operates on two tracks:
- **Short-Term Memory**: A standard queue of the most recent `N` messages in the current session.
- **Long-Term Memory**: A separate background LLM call runs continuously, analyzing the user's messages for facts (e.g., "The user is 16 years old," "The user likes cryptography"). If a fact is detected, it is embedded into a separate ChromaDB collection (`user_facts`). On subsequent turns, the Retriever searches `user_facts` alongside the knowledge base, injecting past memories into the prompt.

### 3. The Orchestrator (`core/orchestrator.py`)
The orchestrator acts as the "brain." It intercepts the user's message and performs the following synchronous workflow:
1. Retrieves historical documents from RAG.
2. Retrieves personal user facts from Long-Term Memory.
3. Retrieves the simulated current year from the Timeline Engine.
4. Compiles the Master Prompt (System Persona + RAG Context + Memory Context + Conversation History).
5. Calls the Google Gemini 1.5 API.
6. Returns the response to the Streamlit UI.
7. Asynchronously fires a background job to extract new long-term facts.

---

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

Then, run the ingestion script to chunk the text files, generate embeddings, and store them into the ChromaDB vector database:

```bash
python scripts/ingest_documents.py
```

### Running the App

Start the Streamlit interface:

```bash
streamlit run streamlit_app.py --server.port 8501
```
Then navigate to `http://localhost:8501` in your browser.

---

## 🧪 Testing

The codebase includes an extensive test suite verifying RAG retrieval accuracy, memory extraction, and persona consistency.

Run the test suite using pytest:

```bash
python -m pytest tests/
```

## 📜 License
MIT License
