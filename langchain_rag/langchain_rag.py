"""
LangChain RAG Implementation — supports any CSV or PDF.
Uses pdfplumber + pypdf + pdfminer fallback chain for maximum compatibility.
"""

import os
import time
from typing import List, Dict

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


# ── PDF Loader with fallback chain ────────────────────────────────────────────

def load_pdf_robust(file_path: str) -> List[Document]:
    docs = []

    # Strategy 1: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            text += "\n" + " | ".join([str(c) for c in row if c])
                if text.strip():
                    docs.append(Document(
                        page_content=text.strip(),
                        metadata={"source": file_path, "page": i + 1, "method": "pdfplumber"}
                    ))
        if docs:
            return docs
    except Exception:
        pass

    # Strategy 2: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={"source": file_path, "page": i + 1, "method": "pypdf"}
                ))
        if docs:
            return docs
    except Exception:
        pass

    # Strategy 3: pdfminer
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        import io
        output = io.StringIO()
        with open(file_path, "rb") as f:
            extract_text_to_fp(f, output, laparams=LAParams())
        full_text = output.getvalue()
        if full_text.strip():
            lines = full_text.split("\n")
            chunk_lines = 50
            for i in range(0, len(lines), chunk_lines):
                chunk = "\n".join(lines[i:i+chunk_lines]).strip()
                if chunk:
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"source": file_path, "page": i // chunk_lines + 1, "method": "pdfminer"}
                    ))
        if docs:
            return docs
    except Exception:
        pass

    raise ValueError("Could not extract text from PDF. File may be scanned/image-only.")


def load_csv_robust(file_path: str) -> List[Document]:
    """Load any CSV with encoding auto-detection."""
    import pandas as pd
    for encoding in ["utf-8", "latin-1", "cp1252", "utf-8-sig"]:
        try:
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")
            df = df.dropna(how="all").fillna("")
            docs = []
            for i, row in df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
                row_text = row_text.encode("utf-8", errors="ignore").decode("utf-8")
                if row_text.strip():
                    docs.append(Document(
                        page_content=row_text,
                        metadata={"source": file_path, "row": i}
                    ))
            return docs
        except Exception:
            continue
    raise ValueError("Could not read CSV with any encoding.")


# ── LangChain RAG Engine ──────────────────────────────────────────────────────

class LangChainRAGEngine:
    def __init__(self, groq_api_key: str):
        os.environ["GROQ_API_KEY"] = groq_api_key
        self.groq_api_key = groq_api_key
        self.chat_history: List = []
        self.vector_store = None
        self.chain = None
        self.retriever = None
        self.is_ready = False
        self.stats: Dict = {}

    def ingest(self, file_path: str, chunk_size: int = 500, overlap: int = 100):
        t0 = time.time()

        ext = file_path.lower().split(".")[-1]
        if ext == "csv":
            docs = load_csv_robust(file_path)
            self.stats["file_type"] = "CSV"
        elif ext == "pdf":
            docs = load_pdf_robust(file_path)
            self.stats["file_type"] = "PDF"
            self.stats["extraction_method"] = docs[0].metadata.get("method", "unknown") if docs else "none"
        else:
            raise ValueError(f"Unsupported file type: .{ext}")

        self.stats["docs_loaded"] = len(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        chunks = splitter.split_documents(docs)
        self.stats["chunks"] = len(chunks)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.stats["embedding_model"] = "all-MiniLM-L6-v2 (384-dim)"

        self.vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        self.stats["vectors_stored"] = self.vector_store._collection.count()
        self.stats["ingest_time"] = round(time.time() - t0, 2)

        self._build_chain()
        self.is_ready = True
        return self.stats

    def _build_chain(self):
        llm = ChatGroq(temperature=0.0, model="llama-3.3-70b-versatile")
        parser = StrOutputParser()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        def format_docs(docs):
            return "\n\n".join(
                f"[Doc {i+1} | Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant.\n"
             "Answer using ONLY the context below.\n"
             "If the answer is not in the context, say 'I don't know.'\n\n"
             "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        self.chain = (
            RunnableParallel({
                "context": self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda _: self.chat_history),
            })
            | prompt
            | llm
            | parser
        )

    def ask(self, question: str, k: int = 4) -> Dict:
        t0 = time.time()
        raw_docs = self.retriever.invoke(question)
        sources = [
            {
                "doc_num": i + 1,
                "preview": doc.page_content[:120],
                "source": doc.metadata.get("source", "N/A"),
                "row": doc.metadata.get("row", doc.metadata.get("page", "N/A")),
            }
            for i, doc in enumerate(raw_docs)
        ]
        answer = self.chain.invoke(question)
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        return {
            "answer": answer,
            "sources": sources,
            "response_time": round(time.time() - t0, 2),
            "history_turns": len(self.chat_history) // 2,
        }

    def clear_history(self):
        self.chat_history.clear()