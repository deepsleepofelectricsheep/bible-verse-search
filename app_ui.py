import streamlit as st

st.set_page_config(page_title="Bible Verse Search", layout="centered")

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    :root {
        --sp-xs: 0.25rem;
        --sp-sm: 0.5rem;
        --sp-md: 0.75rem;
        --sp-lg: 1rem;
        --sp-xl: 1.5rem;
    }

    .stMainBlockContainer { padding-top: var(--sp-lg) !important; }

    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:first-child {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: var(--background-color);
        padding-bottom: var(--sp-sm);
    }

    div[data-testid="stHorizontalBlock"] { gap: var(--sp-md) !important; }

    div[data-testid="stSelectbox"] > div[data-baseweb="select"] * { cursor: pointer !important; }

    .method-desc {
        font-size: 0.75rem;
        color: #888;
        opacity: 0;
        transition: opacity 0.15s;
        padding-top: var(--sp-xs);
    }
    div[data-testid="stColumn"]:last-child:hover .method-desc { opacity: 1; }

    .result-ref { margin-bottom: 0 !important; }
    .result-ref p { margin-bottom: 0 !important; }
    .result-text p { margin-bottom: 0 !important; }

    /* ── Load more ──────────────────────────────────────────────────────────── */
    [class*="st-key-load-more"] div[data-testid="stButton"] > button,
    [class*="st-key-load-more"] [data-testid*="baseButton"] {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        color: #888 !important;
        font-size: 0.85rem !important;
        font-weight: normal !important;
        padding: 2px 0 !important;
        min-height: unset !important;
        height: auto !important;
        width: auto !important;
        border-radius: 0 !important;
        cursor: pointer !important;
        letter-spacing: 0.01em;
    }
    [class*="st-key-load-more"] div[data-testid="stButton"] > button:hover,
    [class*="st-key-load-more"] [data-testid*="baseButton"]:hover {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        color: #333 !important;
        text-decoration: underline !important;
        text-underline-offset: 3px;
    }
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


# ── Sticky header ─────────────────────────────────────────────────────────────

with st.container():
    st.markdown(
        """
        <div style="text-align:center; padding: 0 0 0.75rem 0;">
            <h1 style="margin-bottom:0.25rem; margin-top:0;">Search engine for Bible verses</h1>
            <p style="margin:0; font-size:1.1rem; color:#888;">
                Find the most relevant among 31,102 verses from the King James Bible
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    METHODS = ["TF-IDF", "BM25", "BGE (bi-encoder)"]
    METHOD_DESCRIPTIONS = {
        "TF-IDF": "Keyword matching weighted by term frequency and rarity.",
        "BM25": "Keyword matching with document-length normalization.",
        "BGE (bi-encoder)": "Semantic search using a small transformer model.",
    }

    col_query, col_method = st.columns([4, 1.35])
    with col_query:
        query = st.text_input("Search query", placeholder="e.g. love your neighbor", label_visibility="collapsed")
    with col_method:
        method = st.selectbox("Method", METHODS, index=2, label_visibility="collapsed")
        st.markdown(
            f'<div class="method-desc">{METHOD_DESCRIPTIONS[method]}</div>',
            unsafe_allow_html=True,
        )

# ── Session state ─────────────────────────────────────────────────────────────

if "displayed_count" not in st.session_state:
    st.session_state.displayed_count = 10

# ── Results ───────────────────────────────────────────────────────────────────

if query.strip():
    if (st.session_state.get("_last_query") != query.strip() or
            st.session_state.get("_last_method") != method):
        st.session_state.displayed_count = 10
        st.session_state["_last_query"] = query.strip()
        st.session_state["_last_method"] = method

    with st.spinner("Loading…"):
        searcher = LOADERS[method]()
        results = searcher.search(query.strip(), top_k=50)

    top_score = results[0].score
    filtered = [r for r in results if r.score >= 0.5 * top_score]

    if not filtered:
        st.warning("No strong matches found.")
    else:
        n = len(filtered)
        count_label = f"**{n} result{'s' if n != 1 else ''}**"
        st.markdown(f"{count_label} for *{query.strip()}* using **{method}**")

        visible = filtered[:st.session_state.displayed_count]
        for i, r in enumerate(visible, start=1):
            ref = f"{r.book} {r.chapter}:{r.verse}"
            st.markdown(f"**{i}. {ref}**")
            st.markdown(r.text)
            st.markdown('<hr style="margin: 0.5rem 0; border: none; border-top: 1px solid #e0e0e0;"/>', unsafe_allow_html=True)

        remaining = n - len(visible)
        if remaining > 0:
            to_load = min(5, remaining)
            cols = st.columns([2, 1, 2])
            with cols[1]:
                with st.container(key="load-more"):
                    if st.button(f"Load {to_load} more", key="load_more_btn"):
                        st.session_state.displayed_count += 5
                        st.rerun()

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
        color: #555;
        background: #ffffff;
    }
    .sticky-footer a { color: #555; }
    section.main > div { padding-bottom: 3rem; }
    </style>
    <div class="sticky-footer">
        Built by <strong>Nikhil Raman</strong> · ML Engineer ·
        <a href="https://www.linkedin.com/in/nikhil-raman/" target="_blank">LinkedIn ↗</a>
    </div>
    """,
    unsafe_allow_html=True,
)
