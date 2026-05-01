import streamlit as st
import streamlit.components.v1 as components
from html import escape

st.set_page_config(page_title="Bible Verse Search", layout="centered")

SNIPPET_LEN = 80

# ── Styles ─────────────────────────────────────────────────────────────────────

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

    .stMainBlockContainer {
        padding-top: var(--sp-lg) !important;
        max-width: 900px !important;
    }

    div[data-testid="stHorizontalBlock"] { gap: var(--sp-md) !important; }

    div[data-testid="stSelectbox"] > div[data-baseweb="select"] * { cursor: pointer !important; }

    /* "Open →" card buttons */
    [class*="st-key-card_open"] button {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        color: #555 !important;
        padding: 0 !important;
        font-size: 0.85rem !important;
        min-height: unset !important;
    }
    [class*="st-key-card_open"] button:hover {
        color: #111 !important;
        text-decoration: underline !important;
        text-underline-offset: 3px;
    }

    /* "← Back" button */
    [class*="st-key-back_btn"] button {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        color: #555 !important;
        padding: 0 !important;
        font-size: 0.9rem !important;
        min-height: unset !important;
    }
    [class*="st-key-back_btn"] button:hover {
        color: #111 !important;
        text-decoration: underline !important;
        text-underline-offset: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Index ──────────────────────────────────────────────────────────────────────

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

# ── Chapter HTML builder ───────────────────────────────────────────────────────

def _build_chapter_html(result) -> str:
    first_match = min(result.matched_verses)

    rows = []
    for v in result.verses:
        vnum = v["verse"]
        is_match = vnum in result.matched_verses
        text = escape(v["text"])

        if is_match:
            row_style = (
                "display:flex;align-items:baseline;padding:6px 12px;"
                "border-left:3px solid #888;background:rgba(0,0,0,0.04);"
                "margin:2px 0;border-radius:0 3px 3px 0;"
            )
        else:
            row_style = "display:flex;align-items:baseline;padding:6px 12px;margin:2px 0;"

        rows.append(
            f'<div id="v{vnum}" style="{row_style}">'
            f'<span style="min-width:2em;font-size:0.72em;color:#aaa;margin-right:10px;flex-shrink:0;">{vnum}</span>'
            f'<span>{text}</span>'
            f'</div>'
        )

    verses_html = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #fff; font-family: 'Source Sans Pro', 'Segoe UI', sans-serif; font-size: 15px; line-height: 1.65; color: #333; }}
  #container {{ height: 480px; overflow-y: auto; padding: 4px 0; }}
</style>
</head>
<body>
<div id="container">
{verses_html}
</div>
<script>
  document.addEventListener('DOMContentLoaded', function() {{
    var container = document.getElementById('container');
    var el = document.getElementById('v{first_match}');
    if (container && el) {{
      container.scrollTop = el.offsetTop - container.clientHeight / 2 + el.offsetHeight / 2;
    }}
  }});
</script>
</body>
</html>"""

# ── Session state defaults ─────────────────────────────────────────────────────

for _key, _default in [("view", "results"), ("selected_idx", 0), ("results", [])]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Sticky header ──────────────────────────────────────────────────────────────

with st.container():
    st.markdown(
        """
        <div style="text-align:center; padding: 0 0 0.75rem 0;">
            <h1 style="margin-bottom:0.25rem; margin-top:0;">Search engine for Bible verses</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    METHODS = ["TF-IDF", "BM25", "BGE (bi-encoder)"]

    col_query, col_method = st.columns([4, 1.35])
    with col_query:
        query = st.text_input("Search query", placeholder="e.g. love your neighbor", label_visibility="collapsed")
    with col_method:
        method = st.selectbox("Method", METHODS, index=2, label_visibility="collapsed")

# ── Search & result caching ────────────────────────────────────────────────────

if query.strip():
    if (st.session_state.get("_last_query") != query.strip() or
            st.session_state.get("_last_method") != method):
        st.session_state.view = "results"
        st.session_state.selected_idx = 0
        st.session_state["_last_query"] = query.strip()
        st.session_state["_last_method"] = method

        with st.spinner("Loading…"):
            searcher = LOADERS[method]()
            all_results = searcher.search_chapters(query.strip(), top_k=5)

        top_score = all_results[0].score if all_results else 0
        st.session_state.results = [r for r in all_results if r.score >= 0.5 * top_score]

    results = st.session_state.results

    if not results:
        st.warning("No results found.")

    elif st.session_state.view == "chapter":
        # ── Chapter reader ─────────────────────────────────────────────────────
        chapter = results[st.session_state.selected_idx]

        if st.button("← Back to results", key="back_btn"):
            st.session_state.view = "results"
            st.rerun()

        st.markdown(
            f"<h2 style='margin-top:0.5rem;margin-bottom:0.25rem'>{escape(chapter.book)} {chapter.chapter}</h2>",
            unsafe_allow_html=True,
        )
        components.html(_build_chapter_html(chapter), height=500, scrolling=False)

    else:
        # ── Results list ───────────────────────────────────────────────────────
        n = len(results)
        st.markdown(
            f"<div style='color:#888;font-size:0.9em;margin-bottom:0.75rem'>"
            f"{n} result{'s' if n != 1 else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )

        for i, chapter in enumerate(results):
            verse_lookup = {v["verse"]: v["text"] for v in chapter.verses}
            verse_nums = ", ".join(str(v) for v in sorted(chapter.matched_verses))

            with st.container(border=True):
                col_info, col_btn = st.columns([6, 1])
                with col_info:
                    st.markdown(
                        f"**{escape(chapter.book)} {chapter.chapter}**"
                        f"<span style='color:#888;font-size:0.9em'> · verses {verse_nums}</span>",
                        unsafe_allow_html=True,
                    )
                    for vnum in sorted(chapter.matched_verses):
                        text = verse_lookup.get(vnum, "")
                        snippet = text[:SNIPPET_LEN] + "…" if len(text) > SNIPPET_LEN else text
                        st.markdown(
                            f"<div style='display:flex;gap:10px;margin-top:4px'>"
                            f"<span style='min-width:1.5em;color:#aaa;font-size:0.85em;flex-shrink:0'>{vnum}</span>"
                            f"<span style='font-size:0.9em;color:#555'>{escape(snippet)}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                with col_btn:
                    if st.button("Open →", key=f"card_open_{i}"):
                        st.session_state.selected_idx = i
                        st.session_state.view = "chapter"
                        st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────

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
