import streamlit as st
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

# Set the layout and branding style
st.set_page_config(page_title="Ask the Video", layout="centered")

# Load data and initialize models
@st.cache_resource
def load_data():
    with open("chunked_transcript.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    model = SentenceTransformer("intfloat/e5-large-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return chunks, texts, tfidf_vectorizer, tfidf_matrix, bm25, model, faiss_index

# Load resources
chunks, texts, tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, faiss_index = load_data()

# Sidebar for settings
with st.sidebar:
    st.header("🔧 Settings")
    method = st.selectbox("Choose a method", ["FAISS", "TF-IDF", "BM25"])
    top_k = st.slider("Number of Results", 1, 5, 3)
    def_threshold = {"FAISS": 0.35, "TF-IDF": 0.05, "BM25": 3.0}
    min_score = st.number_input("Relevance Threshold", min_value=0.0, value=def_threshold[method], step=0.01)

# Main interface
st.title("🎬 Ask Your Video")
st.markdown("Type a natural question and we'll find matching video segments.")
query = st.text_input("What would you like to know?")

if st.button("Find Answer") and query:
    if method == "FAISS":
        q_emb = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = faiss_index.search(q_emb, top_k)
        results = [
            {"chunk": chunks[idx], "score": score}
            for idx, score in zip(I[0], D[0]) if score >= min_score
        ]

    elif method == "TF-IDF":
        q_vec = tfidf_vectorizer.transform([query])
        scores = (tfidf_matrix @ q_vec.T).toarray().ravel()
        max_score = scores.max()
        threshold = max(min_score, 0.10 * max_score)
        idxs = np.where(scores >= threshold)[0]
        sorted_idxs = idxs[np.argsort(scores[idxs])[::-1][:top_k]]
        results = [{"chunk": chunks[i], "score": scores[i]} for i in sorted_idxs]

    else:  # BM25
        scores = bm25.get_scores(query.lower().split())
        max_score = max(scores)
        if max_score < min_score:
            results = []
        else:
            idxs = np.argsort(scores)[::-1][:top_k]
            results = [
                {"chunk": chunks[i], "score": scores[i]}
                for i in idxs if scores[i] >= min_score
            ]

    # Display Results
    st.subheader(f"🔍 Top Matches ({method})")
    if results:
        for r in results:
            c = r["chunk"]
            st.markdown(f"**📍 {c['start']}s – {c['end']}s**  ")
            st.markdown(f"**Score:** {r['score']:.3f}")
            st.markdown(f"📝 *{c['text']}*")
            st.components.v1.html(f"""
                <iframe width='100%' height='320'
                    src='https://www.youtube.com/embed/dARr3lGKwk8?start={int(c['start'])}&autoplay=0'
                    frameborder='0'
                    allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture'
                    allowfullscreen>
                </iframe>
            """, height=330)
    else:
        st.error("No relevant content was found for your question. Try rephrasing it or lowering the threshold.")
