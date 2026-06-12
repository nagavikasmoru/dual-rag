# Dual RAG: Manual vs LangChain

Side-by-side comparison of two RAG architectures, both powered by **Groq + LLaMA 3.3 70B**.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Usage

1. Enter your **Groq API key** in the sidebar
2. Provide your **CSV file path** or upload it
3. Click **"Ingest Both"** to build both RAG systems
4. Ask questions in the **Chat tab**
5. Compare response times in the **Performance tab**
6. Understand the differences in the **Architecture tab**

## Project Structure

```
dual_rag/
├── app.py                        # Streamlit UI (3 tabs)
├── requirements.txt
├── manual_rag/
│   └── manual_rag.py             # ManualRAGEngine (pure Python)
└── langchain_rag/
    └── langchain_rag.py          # LangChainRAGEngine (LCEL chain)
```

## Tech Stack

| Component | Manual RAG | LangChain RAG |
|---|---|---|
| Embedding | TF-IDF (custom) | all-MiniLM-L6-v2 |
| Vector Store | Python list | Chroma |
| Similarity | Cosine (manual) | HNSW index |
| LLM | groq SDK | langchain_groq |
| Memory | List of dicts | HumanMessage/AIMessage |
