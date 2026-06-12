"""
Dual RAG Implementation - Streamlit UI
Side-by-side comparison: Manual RAG vs LangChain RAG
"""

import streamlit as st
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from manual_rag.manual_rag import ManualRAGEngine
from langchain_rag.langchain_rag import LangChainRAGEngine
from utils.evaluator import RAGEvaluator

# -- Page Config ---------------------------------------------------------------

st.set_page_config(
    page_title="Dual RAG: Manual vs LangChain",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- CSS -----------------------------------------------------------------------

st.markdown("""
<style>
  /* Global */
  .stApp { background-color: #0a1a0f; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #0f2318;
    border: 1px solid #1f4a2e;
    border-radius: 10px;
    padding: 12px 16px;
  }

  /* Chat bubbles */
  .chat-user {
    background: #0f2d1a;
    border-left: 3px solid #4ade80;
    padding: 10px 14px;
    border-radius: 0 10px 10px 0;
    margin: 6px 0;
    font-size: 0.93rem;
    color: #d1fae5;
  }
  .chat-bot {
    background: #0a1f10;
    border-left: 3px solid #86efac;
    padding: 10px 14px;
    border-radius: 0 10px 10px 0;
    margin: 6px 0;
    font-size: 0.93rem;
    color: #bbf7d0;
  }

  /* Source cards */
  .source-card {
    background: #081510;
    border: 1px solid #1a3d22;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.82rem;
    color: #86efac;
  }

  /* Panel headers */
  .panel-header-manual {
    background: linear-gradient(135deg, #14532d, #0f3d20);
    border: 1px solid #4ade8044;
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 12px;
  }
  .panel-header-lc {
    background: linear-gradient(135deg, #166534, #0a2e18);
    border: 1px solid #86efac44;
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 12px;
  }

  /* Status badge */
  .badge-ready {
    display: inline-block;
    background: #14532d;
    color: #4ade80;
    border: 1px solid #4ade8055;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .badge-pending {
    display: inline-block;
    background: #3d2a00;
    color: #fbbf24;
    border: 1px solid #fbbf2455;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
  }

  /* Comparison table */
  .cmp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  .cmp-table th {
    background: #0f2318;
    color: #86efac;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #1f4a2e;
  }
  .cmp-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #0f2318;
    color: #bbf7d0;
  }
  .cmp-table tr:hover td { background: #0f2318; }
  .manual-col { color: #4ade80; font-weight: 600; }
  .lc-col { color: #86efac; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# -- Session State -------------------------------------------------------------

def init_state():
    defaults = {
        "manual_engine": None,
        "lc_engine": None,
        "manual_chat": [],       # [{role, content, sources, time, tokens}]
        "lc_chat": [],
        "manual_ready": False,
        "lc_ready": False,
        "manual_stats": {},
        "lc_stats": {},
        "perf_log": [],          # [{question, manual_time, lc_time}]
        "api_key": "",
        "csv_path": "",
        "ingesting": False,
        "eval_log": [],        # [{question, manual_eval, lc_eval}]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# -- Sidebar -------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="gsk_...",
    )
    if api_key:
        st.session_state.api_key = api_key

    st.markdown("---")
    st.markdown("### 📄 Document")

    file_path = st.text_input(
        "File Path (CSV or PDF)",
        value=st.session_state.csv_path,
        placeholder="/path/to/your/file.pdf",
    )
    if file_path:
        st.session_state.csv_path = file_path

    uploaded = st.file_uploader("Or upload CSV / PDF", type=["csv", "pdf"])
    if uploaded:
        import tempfile, os
        save_path = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.read())
        st.session_state.csv_path = save_path
        ext = uploaded.name.split(".")[-1].upper()
        st.success(f"✅ {ext} uploaded: {uploaded.name}")

    st.markdown("---")
    st.markdown("### 🔧 Chunking")
    chunk_size = st.slider("Chunk Size (words)", 100, 1000, 500, 50)
    overlap = st.slider("Overlap (words)", 0, 200, 100, 10)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        ingest_manual = st.button("⚡ Manual", use_container_width=True, type="primary")
    with col_b:
        ingest_lc = st.button("🦜 LangChain", use_container_width=True, type="primary")

    ingest_both = st.button("🚀 Ingest Both", use_container_width=True)

    st.markdown("---")
    if st.button("🗑️ Clear All Chats", use_container_width=True):
        st.session_state.manual_chat = []
        st.session_state.lc_chat = []
        if st.session_state.manual_engine:
            st.session_state.manual_engine.clear_history()
        if st.session_state.lc_engine:
            st.session_state.lc_engine.clear_history()
        st.session_state.perf_log = []
        st.rerun()

    # Stats panel
    st.markdown("---")
    st.markdown("### 📊 Ingest Stats")
    if st.session_state.manual_ready:
        s = st.session_state.manual_stats
        st.markdown("**Manual RAG**")
        st.caption(f"⏱ {s.get('ingest_time','?')}s | 📦 {s.get('chunks','?')} chunks | 📖 {s.get('vocab_size','?')} vocab")
    if st.session_state.lc_ready:
        s = st.session_state.lc_stats
        st.markdown("**LangChain RAG**")
        st.caption(f"⏱ {s.get('ingest_time','?')}s | 📦 {s.get('chunks','?')} chunks | 🔢 {s.get('vectors_stored','?')} vectors")


# -- Ingest Logic --------------------------------------------------------------

def do_ingest_manual():
    if not st.session_state.api_key:
        st.error("Enter your Groq API key first.")
        return
    if not st.session_state.csv_path:
        st.error("Provide a CSV file path or upload a file.")
        return
    with st.spinner("⚡ Building Manual RAG (TF-IDF + custom vector store)…"):
        engine = ManualRAGEngine(st.session_state.api_key)
        stats = engine.ingest(st.session_state.csv_path, chunk_size, overlap)
        st.session_state.manual_engine = engine
        st.session_state.manual_stats = stats
        st.session_state.manual_ready = True
    st.success(f"✅ Manual RAG ready in {stats['ingest_time']}s")


def do_ingest_lc():
    if not st.session_state.api_key:
        st.error("Enter your Groq API key first.")
        return
    if not st.session_state.csv_path:
        st.error("Provide a CSV file path or upload a file.")
        return
    with st.spinner("🦜 Building LangChain RAG (HuggingFace + Chroma)…"):
        engine = LangChainRAGEngine(st.session_state.api_key)
        stats = engine.ingest(st.session_state.csv_path, chunk_size, overlap)
        st.session_state.lc_engine = engine
        st.session_state.lc_stats = stats
        st.session_state.lc_ready = True
    st.success(f"✅ LangChain RAG ready in {stats['ingest_time']}s")


if ingest_manual:
    do_ingest_manual()
if ingest_lc:
    do_ingest_lc()
if ingest_both:
    do_ingest_manual()
    do_ingest_lc()


# -- Main Header ---------------------------------------------------------------

st.markdown("""
<h1 style='text-align:center; background: linear-gradient(90deg, #4ade80, #86efac);
-webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:4px;'>
🔍 Dual RAG: Manual vs LangChain</h1>
<p style='text-align:center; color:#718096; margin-bottom:20px;'>
Side-by-side comparison of two RAG architectures powered by Groq + LLaMA 3.3 70B</p>
""", unsafe_allow_html=True)

# Tabs
tab_chat, tab_compare, tab_arch, tab_eval = st.tabs(["💬 Chat", "📊 Performance", "🏗️ Architecture", "🧪 RAGAS Eval"])


# -- TAB 1: CHAT ---------------------------------------------------------------

with tab_chat:
    col_left, col_right = st.columns(2)

    # -- Manual Panel ----------------------------------------------------------
    with col_left:
        badge = '<span class="badge-ready">● Ready</span>' if st.session_state.manual_ready else '<span class="badge-pending">○ Not Ingested</span>'
        st.markdown(f"""
        <div class="panel-header-manual">
          <h3 style='margin:0; color:#4ade80;'>⚡ Manual RAG</h3>
          <p style='margin:4px 0 0 0; color:#718096; font-size:0.82rem;'>TF-IDF · Custom Cosine · Pure Python</p>
          {badge}
        </div>""", unsafe_allow_html=True)

        # Chat history
        chat_container_m = st.container(height=420)
        with chat_container_m:
            if not st.session_state.manual_chat:
                st.markdown("<p style='color:#4a6080; text-align:center; margin-top:60px;'>Ingest a CSV then ask a question →</p>", unsafe_allow_html=True)
            for msg in st.session_state.manual_chat:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🙋 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bot">⚡ {msg["content"]}</div>', unsafe_allow_html=True)
                    # Metrics row
                    m1, m2, m3 = st.columns(3)
                    m1.caption(f"⏱ {msg.get('time','?')}s")
                    m2.caption(f"🎯 {msg.get('top_score','?')}")
                    m3.caption(f"🔢 {msg.get('tokens','?')} tok")
                    # Sources
                    with st.expander("📎 Sources", expanded=False):
                        for s in msg.get("sources", []):
                            st.markdown(f'<div class="source-card">Chunk #{s["chunk_id"]} | Score: {s["score"]}<br>{s["preview"]}…</div>', unsafe_allow_html=True)

    # -- LangChain Panel -------------------------------------------------------
    with col_right:
        badge = '<span class="badge-ready">● Ready</span>' if st.session_state.lc_ready else '<span class="badge-pending">○ Not Ingested</span>'
        st.markdown(f"""
        <div class="panel-header-lc">
          <h3 style='margin:0; color:#86efac;'>🦜 LangChain RAG</h3>
          <p style='margin:4px 0 0 0; color:#718096; font-size:0.82rem;'>MiniLM · Chroma · LCEL Chain</p>
          {badge}
        </div>""", unsafe_allow_html=True)

        chat_container_lc = st.container(height=420)
        with chat_container_lc:
            if not st.session_state.lc_chat:
                st.markdown("<p style='color:#4a3080; text-align:center; margin-top:60px;'>Ingest a CSV then ask a question →</p>", unsafe_allow_html=True)
            for msg in st.session_state.lc_chat:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🙋 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bot">🦜 {msg["content"]}</div>', unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.caption(f"⏱ {msg.get('time','?')}s")
                    m2.caption(f"🔄 Turn {msg.get('turn','?')}")
                    with st.expander("📎 Sources", expanded=False):
                        for s in msg.get("sources", []):
                            st.markdown(f'<div class="source-card">Doc #{s["doc_num"]} | Row: {s["row"]}<br>{s["preview"]}…</div>', unsafe_allow_html=True)

    # -- Question Input --------------------------------------------------------
    st.markdown("---")
    with st.form("question_form", clear_on_submit=True):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            question = st.text_input(
                "Ask a question",
                placeholder="e.g. List all shirts with sun protection",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("Send →", use_container_width=True, type="primary")

    # Sample questions
    st.markdown("**Try:** ")
    sample_cols = st.columns(4)
    samples = [
        "List all shirts with sun protection",
        "What are the cheapest jackets?",
        "Which products have waterproof features?",
        "Tell me about hiking boots",
    ]
    for i, (sc, sq) in enumerate(zip(sample_cols, samples)):
        if sc.button(sq, key=f"sample_{i}", use_container_width=True):
            question = sq
            submitted = True

    if submitted and question:
        any_ready = st.session_state.manual_ready or st.session_state.lc_ready
        if not any_ready:
            st.warning("Please ingest a document first.")
        else:
            perf = {"question": question, "manual_time": None, "lc_time": None}

            # Add user message to both chats
            if st.session_state.manual_ready:
                st.session_state.manual_chat.append({"role": "user", "content": question})
            if st.session_state.lc_ready:
                st.session_state.lc_chat.append({"role": "user", "content": question})

            # Run Manual RAG
            if st.session_state.manual_ready:
                with st.spinner("⚡ Manual RAG thinking…"):
                    result = st.session_state.manual_engine.ask(question)
                st.session_state.manual_chat.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "context": result.get("context", ""),
                    "time": result["response_time"],
                    "top_score": round(result["top_score"], 4),
                    "tokens": result["tokens_used"],
                })
                perf["manual_time"] = result["response_time"]

            # Run LangChain RAG
            if st.session_state.lc_ready:
                with st.spinner("🦜 LangChain RAG thinking…"):
                    result = st.session_state.lc_engine.ask(question)
                st.session_state.lc_chat.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "context": result.get("context", ""),
                    "time": result["response_time"],
                    "turn": result["history_turns"],
                })
                perf["lc_time"] = result["response_time"]

            # ── RAGAS Evaluation ──────────────────────────────────────
            if st.session_state.manual_ready and st.session_state.lc_ready:
                evaluator = RAGEvaluator(st.session_state.api_key)
                eval_entry = {"question": question, "manual_eval": None, "lc_eval": None}

                manual_result = next((m for m in reversed(st.session_state.manual_chat) if m["role"] == "assistant"), None)
                lc_result     = next((m for m in reversed(st.session_state.lc_chat)     if m["role"] == "assistant"), None)

                if manual_result and manual_result.get("context"):
                    with st.spinner("Evaluating Manual RAG..."):
                        eval_entry["manual_eval"] = evaluator.evaluate_all(
                            question, manual_result["content"], manual_result["context"]
                        )
                if lc_result and lc_result.get("context"):
                    with st.spinner("Evaluating LangChain RAG..."):
                        eval_entry["lc_eval"] = evaluator.evaluate_all(
                            question, lc_result["content"], lc_result["context"]
                        )
                st.session_state.eval_log.append(eval_entry)

            st.session_state.perf_log.append(perf)
            st.rerun()


# -- TAB 2: PERFORMANCE --------------------------------------------------------

with tab_compare:
    st.markdown("### 📊 Response Time Comparison")

    if not st.session_state.perf_log:
        st.info("Ask some questions in the Chat tab to see performance data here.")
    else:
        import pandas as pd

        log = st.session_state.perf_log
        df_perf = pd.DataFrame(log)
        df_perf.index += 1

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        manual_times = [x for x in df_perf["manual_time"].dropna()]
        lc_times = [x for x in df_perf["lc_time"].dropna()]

        if manual_times:
            m1.metric("⚡ Manual Avg", f"{sum(manual_times)/len(manual_times):.2f}s")
            m2.metric("⚡ Manual Best", f"{min(manual_times):.2f}s")
        if lc_times:
            m3.metric("🦜 LC Avg", f"{sum(lc_times)/len(lc_times):.2f}s")
            m4.metric("🦜 LC Best", f"{min(lc_times):.2f}s")

        # Table
        st.dataframe(
            df_perf.rename(columns={
                "question": "Question",
                "manual_time": "⚡ Manual (s)",
                "lc_time": "🦜 LangChain (s)",
            }),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### 🔬 Technical Comparison")

    st.markdown("""
    <table class="cmp-table">
      <tr>
        <th>Aspect</th>
        <th class="manual-col">⚡ Manual RAG</th>
        <th class="lc-col">🦜 LangChain RAG</th>
      </tr>
      <tr><td>Embedding Type</td><td>TF-IDF (sparse)</td><td>all-MiniLM-L6-v2 (dense, 384-dim)</td></tr>
      <tr><td>Similarity</td><td>Manual cosine (pure Python)</td><td>Chroma HNSW index</td></tr>
      <tr><td>Vector Store</td><td>In-memory list</td><td>Chroma (persistent)</td></tr>
      <tr><td>Semantic Understanding</td><td>Keyword overlap (TF-IDF)</td><td>Semantic meaning (neural)</td></tr>
      <tr><td>Memory</td><td>List of {role, content} dicts</td><td>HumanMessage / AIMessage objects</td></tr>
      <tr><td>Chain Logic</td><td>Custom ask() function</td><td>LCEL RunnableParallel chain</td></tr>
      <tr><td>LLM Call</td><td>groq SDK directly</td><td>langchain_groq ChatGroq wrapper</td></tr>
      <tr><td>Source Metadata</td><td>Chunk ID + TF-IDF score</td><td>Row number + file path</td></tr>
      <tr><td>Scalability</td><td>Limited (O(n) search)</td><td>Good (HNSW ANN index)</td></tr>
      <tr><td>Lines of Code</td><td>~200 (all from scratch)</td><td>~80 (framework abstractions)</td></tr>
      <tr><td>Resume Value</td><td>Shows deep understanding</td><td>Shows production readiness</td></tr>
    </table>
    """, unsafe_allow_html=True)


# -- TAB 3: ARCHITECTURE -------------------------------------------------------

with tab_arch:
    a_left, a_right = st.columns(2)

    with a_left:
        st.markdown("### ⚡ Manual RAG Pipeline")
        st.markdown("""
```
CSV File
    |
    ▼
pandas.read_csv()
→ row-by-row text string
    |
    ▼
split_text()
→ word-based chunks (size=500, overlap=100)
→ [{id, text, start_word, end_word}, ...]
    |
    ▼
ManualTFIDFEmbedder.fit()
→ build vocabulary from all chunks
→ compute IDF = log(N / df+1) per word
    |
    ▼
ManualTFIDFEmbedder.embed_batch()
→ sparse TF-IDF vector per chunk
→ vocab_size-dimensional float list
    |
    ▼
ManualVectorStore
→ stores vectors + chunks in Python lists
    |
    ▼  [at query time]
cosine_similarity(query_vec, all_vecs)
→ top-k chunks by score
    |
    ▼
Groq SDK - llama-3.3-70b-versatile
→ system: context chunks
→ messages: full conversation history
    |
    ▼
Answer + Sources + Metrics
```
        """)

    with a_right:
        st.markdown("### 🦜 LangChain RAG Pipeline")
        st.markdown("""
```
CSV File
    |
    ▼
CSVLoader (langchain_community)
→ List[Document] with metadata
    |
    ▼
RecursiveCharacterTextSplitter
→ char-aware splitting (size=500, overlap=100)
→ preserves Document metadata
    |
    ▼
HuggingFaceEmbeddings
→ all-MiniLM-L6-v2 model
→ dense 384-dim semantic vectors
    |
    ▼
Chroma.from_documents()
→ HNSW index for fast ANN search
→ persistent vector store
    |
    ▼  [at query time - LCEL chain]
RunnableParallel
├-- retriever → format_docs   → context
├-- RunnablePassthrough        → question
└-- lambda chat_history        → history
    |
    ▼
ChatPromptTemplate (MessagesPlaceholder)
    |
    ▼
ChatGroq - llama-3.3-70b-versatile
    |
    ▼
StrOutputParser
    |
    ▼
Answer + Sources + Metrics
```
        """)

    st.markdown("---")
    st.markdown("### 💡 When to Use Each Approach")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Use Manual RAG when:**
- 📚 Learning RAG fundamentals
- 🔬 You need full control over every step
- 🧪 Prototyping custom retrieval logic
- 📝 Interview/portfolio demonstration
- ⚡ No external dependencies allowed
- 🎯 Custom domain-specific similarity logic
        """)
    with c2:
        st.markdown("""
**Use LangChain RAG when:**
- 🚀 Building production systems
- 🔗 Integrating multiple tools/chains
- 📈 Need scalable vector search
- 🤝 Team collaboration (standard patterns)
- 🔄 Rapid prototyping with best practices
- 🌐 Semantic search quality matters
        """)



# ── TAB 4: RAGAS EVALUATION ───────────────────────────────────────────────────

with tab_eval:
    st.markdown("### 🧪 RAGAS Evaluation — LLM as Judge")
    st.markdown(
        "Automatically scores both RAGs on **3 metrics** using LLaMA 3.3 70B as an evaluator. "
        "Runs after every question when both RAGs are active."
    )

    # Metric explanation cards
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.markdown("""
        <div style='background:#0f2318; border:1px solid #4ade8044; border-radius:10px; padding:12px;'>
        <h4 style='color:#4ade80; margin:0;'>Faithfulness</h4>
        <p style='color:#86efac; font-size:0.82rem; margin:6px 0 0 0;'>
        Are all claims in the answer supported by the retrieved context?<br>
        <b>1.0</b> = fully grounded &nbsp; <b>0.0</b> = hallucinated
        </p></div>""", unsafe_allow_html=True)
    with ec2:
        st.markdown("""
        <div style='background:#0f2318; border:1px solid #4ade8044; border-radius:10px; padding:12px;'>
        <h4 style='color:#4ade80; margin:0;'>Answer Relevance</h4>
        <p style='color:#86efac; font-size:0.82rem; margin:6px 0 0 0;'>
        Does the answer actually address the question asked?<br>
        <b>1.0</b> = perfectly answers &nbsp; <b>0.0</b> = off-topic
        </p></div>""", unsafe_allow_html=True)
    with ec3:
        st.markdown("""
        <div style='background:#0f2318; border:1px solid #4ade8044; border-radius:10px; padding:12px;'>
        <h4 style='color:#4ade80; margin:0;'>Context Relevance</h4>
        <p style='color:#86efac; font-size:0.82rem; margin:6px 0 0 0;'>
        Did the retriever fetch chunks useful for answering?<br>
        <b>1.0</b> = perfect retrieval &nbsp; <b>0.0</b> = wrong chunks
        </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    if not st.session_state.eval_log:
        st.info("Ask questions in the Chat tab with both RAGs active to see evaluation scores here.")
    else:
        import pandas as pd

        log = st.session_state.eval_log

        # ── Summary Metrics ───────────────────────────────────────────────────
        st.markdown("#### Overall Scores")

        def avg_metric(log, rag_key, metric):
            vals = [
                e[rag_key][metric]["score"]
                for e in log
                if e.get(rag_key) and e[rag_key].get(metric)
            ]
            return round(sum(vals) / len(vals), 2) if vals else None

        metrics = ["faithfulness", "answer_relevance", "context_relevance", "overall"]
        labels  = ["Faithfulness", "Answer Relevance", "Context Relevance", "Overall"]

        s1, s2, s3, s4 = st.columns(4)
        cols = [s1, s2, s3, s4]

        for col, metric, label in zip(cols, metrics, labels):
            m_score = avg_metric(log, "manual_eval", metric) if metric != "overall" else round(sum([e["manual_eval"]["overall"] for e in log if e.get("manual_eval")]) / max(1, len([e for e in log if e.get("manual_eval")])), 2)
            lc_score = avg_metric(log, "lc_eval", metric) if metric != "overall" else round(sum([e["lc_eval"]["overall"] for e in log if e.get("lc_eval")]) / max(1, len([e for e in log if e.get("lc_eval")])), 2)
            with col:
                st.markdown(f"**{label}**")
                if m_score is not None:
                    delta = round(lc_score - m_score, 2) if lc_score else None
                    st.metric("Manual", m_score, delta=None)
                    st.metric("LangChain", lc_score if lc_score else "N/A",
                             delta=f"{delta:+.2f}" if delta else None)

        st.markdown("---")

        # ── Per Question Detail ───────────────────────────────────────────────
        st.markdown("#### Per Question Breakdown")

        for i, entry in enumerate(log):
            with st.expander(f"Q{i+1}: {entry['question'][:80]}...", expanded=(i == len(log)-1)):
                if not entry.get("manual_eval") or not entry.get("lc_eval"):
                    st.warning("Evaluation only runs when both RAGs are active.")
                    continue

                d_left, d_right = st.columns(2)
                for col, key, label, color in [
                    (d_left,  "manual_eval", "Manual RAG",    "#4ade80"),
                    (d_right, "lc_eval",     "LangChain RAG", "#86efac"),
                ]:
                    e = entry[key]
                    with col:
                        st.markdown(f"<h5 style='color:{color};'>{label} — Overall: {e['overall']}</h5>",
                                   unsafe_allow_html=True)
                        eval_data = [
                            ["Metric", "Score", "Reason"],
                            ["Faithfulness",
                             f"{e['faithfulness']['score']:.2f}",
                             e['faithfulness']['reason'][:80]],
                            ["Answer Relevance",
                             f"{e['answer_relevance']['score']:.2f}",
                             e['answer_relevance']['reason'][:80]],
                            ["Context Relevance",
                             f"{e['context_relevance']['score']:.2f}",
                             e['context_relevance']['reason'][:80]],
                        ]
                        st.table(eval_data)
                        st.caption(f"Eval time: {e.get('eval_time','?')}s")

        st.markdown("---")

        # ── Export CSV ────────────────────────────────────────────────────────
        st.markdown("#### Export Results")

        rows = []
        for e in log:
            row = {"Question": e["question"]}
            for rag_key, prefix in [("manual_eval", "Manual"), ("lc_eval", "LC")]:
                if e.get(rag_key):
                    ev = e[rag_key]
                    row[f"{prefix} Faithfulness"]      = ev["faithfulness"]["score"]
                    row[f"{prefix} Answer Relevance"]  = ev["answer_relevance"]["score"]
                    row[f"{prefix} Context Relevance"] = ev["context_relevance"]["score"]
                    row[f"{prefix} Overall"]           = ev["overall"]
            rows.append(row)

        df_export = pd.DataFrame(rows)
        st.dataframe(df_export, use_container_width=True)

        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Evaluation CSV",
            csv_data,
            file_name="ragas_evaluation.csv",
            mime="text/csv",
        )
