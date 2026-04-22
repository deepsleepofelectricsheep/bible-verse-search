"""
Entry point for the Bible verse search API.
Run with: flask --app app run
"""
from flask import Flask, request, jsonify
from data.loader import load_verses
from search.tfidf import TFIDFSearcher

app = Flask(__name__)

_verses = load_verses()
searcher = TFIDFSearcher()
searcher.index(_verses)


@app.get("/search")
def search():
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("k", 10))
    if not query:
        return jsonify({"error": "missing query parameter 'q'"}), 400
    results = searcher.search(query, top_k=top_k)
    return jsonify([
        {"ref": f"{r.book} {r.chapter}:{r.verse}", "text": r.text, "score": r.score}
        for r in results
    ])
