import streamlit as st

# Set page config FIRST
st.set_page_config(page_title="Video QA", layout="wide")

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss


@st.cache_resource
def load_data():
    with open("chunked_transcript.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    starts = [c["start"] for c in chunks]
    ends = [c["end"] for c in chunks]

    # Normalize embeddings for cosine similarity
    model = SentenceTransformer("intfloat/e5-large-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return chunks, texts, tfidf_vectorizer, tfidf_matrix, bm25, model, faiss_index

chunks, texts, tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, faiss_index = load_data()

st.set_page_config(page_title="Video QA", layout="wide")
st.title("🎥 Semantic Video QA System")
st.video("https://www.youtube.com/watch?v=dARr3lGKwk8")

method = st.sidebar.selectbox("Retrieval Method", ["FAISS", "TF-IDF", "BM25"])
top_k = st.sidebar.slider("Top-k Results", 1, 5, 3)
def_threshold = {"FAISS": 0.35, "TF-IDF": 0.05, "BM25": 3.0}
min_score = st.sidebar.number_input("Min relevance score", min_value=0.0, value=def_threshold[method], step=0.01)

query = st.text_input("🔍 Ask a question:")

if st.button("Search") and query:
    if method == "FAISS":
        q_emb = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = faiss_index.search(q_emb, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if score >= min_score:
                c = chunks[idx]
                results.append({"chunk": c, "score": score})

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
            results = [{"chunk": chunks[i], "score": scores[i]} for i in idxs if scores[i] >= min_score]

    st.write(f"**Results ({method})**")
    if results:
        for r in results:
            c = r["chunk"]
            st.markdown(f"**⏱ {c['start']}s – {c['end']}s**  (Score: {r['score']:.3f})")
            st.markdown(f"> {c['text']}")
            st.components.v1.html(f"""
                <iframe width='700' height='400'
                    src='https://www.youtube.com/embed/dARr3lGKwk8?start={int(c['start'])}&autoplay=0'
                    frameborder='0'
                    allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture'
                    allowfullscreen>
                </iframe>
            """, height=420)
    else:
        st.warning("🤷‍♂️ Sorry, no relevant answer found in the video.")


# # === app.py ===
# import streamlit as st
# import json
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer
# import faiss

# @st.cache_resource
# def load_data():
#     with open("chunked_transcript.json", "r", encoding="utf-8") as f:
#         chunks = json.load(f)

#     texts = [c["text"] for c in chunks]
#     starts = [c["start"] for c in chunks]
#     ends = [c["end"] for c in chunks]

#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

#     tokenized_corpus = [t.lower().split() for t in texts]
#     bm25 = BM25Okapi(tokenized_corpus)

#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(texts, convert_to_tensor=True).detach().cpu().numpy().astype("float32")
#     faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
#     faiss_index.add(embeddings)

#     return chunks, tfidf_vectorizer, tfidf_matrix, bm25, model, faiss_index

# chunks, tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, faiss_index = load_data()

# st.title("🎥 Video Question Answering (RAG System)")
# st.markdown("Enter your question to find the most relevant video segment.")

# query = st.text_input("🔍 Ask a question:")

# YOUTUBE_URL = "https://www.youtube.com/embed/dARr3lGKwk8"

# if query:
#     q_emb = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
#     D, I = faiss_index.search(q_emb, k=5)

#     # Show top-1 result with non-empty transcript
#     top_idx = I[0][0]
#     result = chunks[top_idx]
    
#     if len(result['text'].strip()) > 15:
#         result_found = True
#     else:
#         result_found = False


#     if result_found:
#         st.subheader("🔎 Most Relevant Segment")
#         st.markdown(f"**Timestamp:** {result['start']}s – {result['end']}s")
#         st.markdown(f"**Transcript:** {result['text']}")

#         st.components.v1.html(f"""
#             <iframe width="700" height="400"
#                 src="{YOUTUBE_URL}?start={int(result['start'])}&autoplay=1&modestbranding=1&rel=0"
#                 frameborder="0"
#                 allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
#                 allowfullscreen>
#             </iframe>
#         """, height=420)
#     else:
#         st.warning("🤷‍♂️ Sorry, no relevant answer found in the video.")


