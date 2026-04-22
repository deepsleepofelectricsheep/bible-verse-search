import streamlit as st

st.set_page_config(page_title="Bible Verse Search", layout="centered")

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Reduce Streamlit's default top padding */
    .stMainBlockContainer { padding-top: 1rem !important; }

    /* Sticky header container */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:first-child {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: var(--background-color);
        padding-bottom: 0.5rem;
    }

    /* Column gap */
    div[data-testid="stHorizontalBlock"] { gap: 0.75rem !important; }

</style>
    """,
    unsafe_allow_html=True,
)

# ── Index ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_verses_cached():
    from data.loader import load_verses
    return load_verses()

@st.cache_resource(show_spinner=False)
def load_tfidf():
    from search.tfidf import TFIDFSearcher
    verses = load_verses_cached()
    s = TFIDFSearcher()
    s.index(verses)
    return s

@st.cache_resource(show_spinner=False)
def load_bm25():
    from search.bm25 import BM25Searcher
    verses = load_verses_cached()
    s = BM25Searcher()
    s.index(verses)
    return s

@st.cache_resource(show_spinner=False)
def load_embedding():
    from search.embedding import EmbeddingSearcher
    verses = load_verses_cached()
    s = EmbeddingSearcher()
    s.index(verses)
    return s

LOADERS = {
    "TF-IDF": load_tfidf,
    "BM25": load_bm25,
    "BGE (bi-encoder)": load_embedding,
}

# ── Sticky header (hero + search bar) ────────────────────────────────────────

with st.container():
    st.markdown(
        """
        <div style="text-align:center; padding: 0 0 0.75rem 0;">
            <h1 style="margin-bottom:0.25rem;">Search engine for Bible verses</h1>
            <p style="margin:0; font-size:1.1rem; color:#888;">
                Find the most relevant among 31,102 verses from the King James Bible
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    METHODS = ["TF-IDF", "BM25", "BGE (bi-encoder)"]

    col_query, col_method = st.columns([4, 1])
    with col_query:
        query = st.text_input("Search query", placeholder="e.g. love your neighbor", label_visibility="collapsed")
    with col_method:
        method = st.selectbox("Method", METHODS, index=2, label_visibility="collapsed")

# ── Results ───────────────────────────────────────────────────────────────────

if query.strip():
    with st.spinner("Loading…"):
        searcher = LOADERS[method]()
        results = searcher.search(query.strip(), top_k=10)

    if not results:
        st.warning("No results found.")
    else:
        st.markdown(f"**{len(results)} results** for *{query}* using **{method}**")
        for i, r in enumerate(results, start=1):
            ref = f"{r.book} {r.chapter}:{r.verse}"
            st.markdown(f"**{i}. {ref}** &nbsp;&nbsp; `score: {r.score:.4f}`  \n{r.text}")
            st.markdown('<hr style="margin: 0.4rem 0; border: none; border-top: 1px solid #e0e0e0;"/>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .sticky-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 0.75rem 0;
        font-size: 0.85rem;
        color: #888;
        background: #ffffff;
    }
    .sticky-footer a { color: #888; }
    section.main > div { padding-bottom: 3rem; }
    </style>
    <div class="sticky-footer">
        Built by <strong>Nikhil Raman</strong> · ML Engineer ·
        <a href="https://www.linkedin.com/in/nikhil-raman/" target="_blank">LinkedIn ↗</a>
    </div>
    """,
    unsafe_allow_html=True,
)
