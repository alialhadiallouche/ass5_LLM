# === app.py ===
import streamlit as st
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

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=True).detach().cpu().numpy().astype("float32")
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    return chunks, tfidf_vectorizer, tfidf_matrix, bm25, model, faiss_index

chunks, tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, faiss_index = load_data()

st.title("🎥 Video Question Answering (RAG System)")
st.markdown("Enter your question to find the most relevant video segment.")

query = st.text_input("🔍 Ask a question:")

YOUTUBE_URL = "https://www.youtube.com/embed/dARr3lGKwk8"

if query:
    q_emb = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = faiss_index.search(q_emb, k=5)

    # Show top-1 result with non-empty transcript
    top_idx = I[0][0]
    result = chunks[top_idx]
    
    if len(result['text'].strip()) > 15:
        result_found = True
    else:
        result_found = False


    if result_found:
        st.subheader("🔎 Most Relevant Segment")
        st.markdown(f"**Timestamp:** {result['start']}s – {result['end']}s")
        st.markdown(f"**Transcript:** {result['text']}")

        st.components.v1.html(f"""
            <iframe width="700" height="400"
                src="{YOUTUBE_URL}?start={int(result['start'])}&autoplay=1&modestbranding=1&rel=0"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
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

#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

#     bm25 = BM25Okapi([t.lower().split() for t in texts])

#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(texts, convert_to_tensor=True).detach().cpu().numpy().astype("float32")
#     faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
#     faiss_index.add(embeddings)

#     return chunks, tfidf_vectorizer, tfidf_matrix, bm25, model, faiss_index

# chunks, tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, faiss_index = load_data()

# st.title("🎥 Video QA with RAG (TF-IDF, BM25, FAISS)")
# query = st.text_input("Enter your question:")

# def display_result(title, result):
#     st.subheader(title)
#     st.markdown(f"**Timestamp:** {result['start']}s – {result['end']}s")
#     st.markdown(f"**Transcript:** {result['text']}")
#     if result['start'] < 3600:
#         YOUTUBE_URL = "https://www.youtube.com/embed/dARr3lGKwk8"
#         st.components.v1.html(f"""
#             <iframe width="700" height="400"
#                 src="{YOUTUBE_URL}?start={int(result['start'])}&autoplay=1&modestbranding=1&rel=0"
#                 frameborder="0"
#                 allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
#                 allowfullscreen>
#             </iframe>
#         """, height=420)

# if query:
#     # FAISS
#     q_emb = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
#     _, faiss_ids = faiss_index.search(q_emb, k=1)
#     faiss_result = chunks[faiss_ids[0][0]]

#     # TF-IDF
#     tfidf_query = tfidf_vectorizer.transform([query])
#     tfidf_scores = np.dot(tfidf_matrix, tfidf_query.T).toarray().flatten()
#     tfidf_result = chunks[np.argmax(tfidf_scores)]

#     # BM25
#     bm25_scores = bm25.get_scores(query.lower().split())
#     bm25_result = chunks[np.argmax(bm25_scores)]

#     display_result("🔍 FAISS (Semantic Search)", faiss_result)
#     display_result("📚 TF-IDF (Lexical)", tfidf_result)
#     display_result("📖 BM25 (Probabilistic)", bm25_result)
