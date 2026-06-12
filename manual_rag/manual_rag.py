"""
Manual RAG Implementation — built from scratch, no frameworks.
Supports: Any CSV, Any PDF (text, scanned, resume, report, book)
"""

import re
import math
import time
import pandas as pd
from typing import List, Dict, Tuple
from groq import Groq


# ── 1. Document Loader ────────────────────────────────────────────────────────

def load_document(file_path: str) -> Tuple[str, Dict]:
    ext = file_path.lower().split(".")[-1]
    if ext == "csv":
        return load_csv(file_path)
    elif ext == "pdf":
        return load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


def load_csv(file_path: str) -> Tuple[str, Dict]:
    """Load any CSV — tries multiple encodings, handles messy data."""
    for encoding in ["utf-8", "latin-1", "cp1252", "utf-8-sig"]:
        try:
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")
            break
        except Exception:
            continue
    else:
        raise ValueError("Could not read CSV with any encoding.")

    # Drop fully empty rows/cols
    df = df.dropna(how="all").fillna("")
    rows = []
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
        if row_text.strip():
            rows.append(row_text)
    text = "\n".join(rows)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    stats = {
        "file_type": "CSV",
        "rows": len(df),
        "columns": list(df.columns),
        "total_chars": len(text),
    }
    return text, stats


def load_pdf(file_path: str) -> Tuple[str, Dict]:
    """
    Load any PDF using multiple strategies:
    1. pdfplumber  — best for resumes, tables, columns
    2. pypdf       — fallback for standard PDFs
    3. pdfminer    — fallback for complex layouts
    """
    text = ""
    pages_count = 0
    method_used = ""

    # Strategy 1: pdfplumber
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            pages_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                # Also extract tables
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        if row:
                            table_text += " | ".join([str(c) for c in row if c]) + "\n"
                combined = page_text + ("\n" + table_text if table_text else "")
                if combined.strip():
                    pages.append(f"[Page {i+1}]\n{combined.strip()}")
        text = "\n\n".join(pages)
        method_used = "pdfplumber"
    except Exception:
        pass

    # Strategy 2: pypdf fallback
    if not text.strip():
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            pages_count = len(reader.pages)
            pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"[Page {i+1}]\n{page_text.strip()}")
            text = "\n\n".join(pages)
            method_used = "pypdf"
        except Exception:
            pass

    # Strategy 3: pdfminer fallback
    if not text.strip():
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            text = pdfminer_extract(file_path)
            method_used = "pdfminer"
        except Exception:
            pass

    if not text.strip():
        raise ValueError("Could not extract text from PDF. The file may be scanned/image-only.")

    stats = {
        "file_type": "PDF",
        "pages": pages_count,
        "total_chars": len(text),
        "extraction_method": method_used,
    }
    return text, stats


# ── 2. Text Splitter ──────────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "start_word": i,
            "end_word": i + len(chunk_words),
        })
        chunk_id += 1
        i += chunk_size - overlap
    return chunks


# ── 3. TF-IDF Embedder ────────────────────────────────────────────────────────

class ManualTFIDFEmbedder:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.fitted = False

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def fit(self, documents: List[str]):
        N = len(documents)
        df: Dict[str, int] = {}
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        self.vocab = {word: idx for idx, word in enumerate(df.keys())}
        self.idf = {word: math.log(N / (df[word] + 1)) for word in df}
        self.fitted = True

    def embed(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        total = len(tokens) if tokens else 1
        vector = [0.0] * len(self.vocab)
        for token, count in tf.items():
            if token in self.vocab:
                tfidf = (count / total) * self.idf.get(token, 0)
                vector[self.vocab[token]] = tfidf
        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# ── 4. Vector Store ───────────────────────────────────────────────────────────

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


class ManualVectorStore:
    def __init__(self):
        self.vectors: List[List[float]] = []
        self.chunks: List[Dict] = []

    def add_documents(self, chunks: List[Dict], embedder: ManualTFIDFEmbedder):
        texts = [c["text"] for c in chunks]
        self.vectors = embedder.embed_batch(texts)
        self.chunks = chunks

    def similarity_search(self, query: str, embedder: ManualTFIDFEmbedder, k: int = 4) -> List[Tuple[Dict, float]]:
        query_vector = embedder.embed(query)
        scores = [
            (chunk, cosine_similarity(query_vector, vec))
            for chunk, vec in zip(self.chunks, self.vectors)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ── 5. RAG Engine ─────────────────────────────────────────────────────────────

class ManualRAGEngine:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.embedder = ManualTFIDFEmbedder()
        self.vector_store = ManualVectorStore()
        self.conversation_history: List[Dict] = []
        self.is_ready = False
        self.stats: Dict = {}

    def ingest(self, file_path: str, chunk_size: int = 500, overlap: int = 100):
        t0 = time.time()
        text, load_stats = load_document(file_path)
        self.stats.update(load_stats)
        chunks = split_text(text, chunk_size, overlap)
        self.stats["chunks"] = len(chunks)
        texts = [c["text"] for c in chunks]
        self.embedder.fit(texts)
        self.vector_store.add_documents(chunks, self.embedder)
        self.stats["vocab_size"] = self.embedder.vocab_size
        self.stats["ingest_time"] = round(time.time() - t0, 2)
        self.is_ready = True
        return self.stats

    def ask(self, question: str, k: int = 4) -> Dict:
        t0 = time.time()
        results = self.vector_store.similarity_search(question, self.embedder, k)
        context_parts = [
            f"[Chunk {r[0]['id']} | Score: {r[1]:.3f}]\n{r[0]['text']}"
            for r in results
        ]
        context = "\n\n".join(context_parts)
        sources = [
            {"chunk_id": r[0]["id"], "score": round(r[1], 4), "preview": r[0]["text"][:120]}
            for r in results
        ]
        self.conversation_history.append({"role": "user", "content": question})
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant.\n"
                    "Answer using ONLY the context below.\n"
                    "If the answer is not in the context, say 'I don't know.'\n\n"
                    f"Context:\n{context}"
                ),
            }
        ] + self.conversation_history
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=1024,
            messages=messages,
        )
        answer = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": answer})
        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "top_score": results[0][1] if results else 0,
            "response_time": round(time.time() - t0, 2),
            "tokens_used": response.usage.total_tokens,
        }

    def clear_history(self):
        self.conversation_history.clear()
