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
st.write("Ask a natural language question. The system will return the most relevant video segment.")

query = st.text_input("Enter your question:")
if query:
    q_emb = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss_dist, faiss_ids = faiss_index.search(q_emb, k=1)
    faiss_result = chunks[faiss_ids[0][0]]

    tfidf_query = tfidf_vectorizer.transform([query])
    tfidf_scores = np.dot(tfidf_matrix, tfidf_query.T).toarray().flatten()
    tfidf_idx = np.argmax(tfidf_scores)
    tfidf_result = chunks[tfidf_idx]

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_idx = np.argmax(bm25_scores)
    bm25_result = chunks[bm25_idx]

    st.subheader("Semantic - FAISS")
    st.markdown(f"*Timestamp:* {faiss_result['start']}s – {faiss_result['end']}s")
    st.markdown(f"*Transcript:* {faiss_result['text']}")

    YOUTUBE_URL = "https://www.youtube.com/embed/dARr3lGKwk8"
    start_time = int(faiss_result['start'])
    st.subheader("Video Segment")
    st.components.v1.html(f"""
        <iframe width="700" height="400"
            src="{YOUTUBE_URL}?start={start_time}&autoplay=1&modestbranding=1&rel=0"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
        </iframe>
    """, height=420)

    st.subheader("TF-IDF Result")
    st.markdown(f"*Timestamp:* {tfidf_result['start']}s – {tfidf_result['end']}s")
    st.markdown(f"*Transcript:* {tfidf_result['text']}")

    st.subheader("BM25 Result")
    st.markdown(f"*Timestamp:* {bm25_result['start']}s – {bm25_result['end']}s")
    st.markdown(f"*Transcript:* {bm25_result['text']}")

    if all(len(r["text"].strip()) == 0 for r in [faiss_result, tfidf_result, bm25_result]):
        st.warning("No relevant answer found in the video.")


# === evaluate.py ===
import time
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss


def timestamps_overlap(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or a_start > b_end)


def load_resources():
    with open("chunked_transcript.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open("gold_set_test.json", "r", encoding="utf-8") as f:
        gold_set = json.load(f)

    texts = [c["text"] for c in chunks]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    bm25 = BM25Okapi([t.lower().split() for t in texts])
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=True).detach().cpu().numpy().astype("float32")
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    return chunks, gold_set, {
        "tfidf": (tfidf_vectorizer, tfidf_matrix),
        "bm25": bm25,
        "faiss": (model, faiss_index)
    }


def evaluate_retrieval(chunks, gold_set, resources):
    methods = ["TF-IDF", "BM25", "FAISS"]
    results = {"Method": [], "Accuracy": [], "False Positive Rate": [], "Avg Latency (ms)": []}

    for method in methods:
        correct = 0
        latencies = []

        for q in gold_set["answerable_questions"]:
            start = time.time()
            if method == "TF-IDF":
                vectorizer, matrix = resources["tfidf"]
                qv = vectorizer.transform([q["question"]])
                scores = np.dot(matrix, qv.T).toarray().flatten()
                best_idx = np.argmax(scores)
            elif method == "BM25":
                bm25 = resources["bm25"]
                scores = bm25.get_scores(q["question"].lower().split())
                best_idx = np.argmax(scores)
            else:
                model, index = resources["faiss"]
                q_emb = model.encode([q["question"]], convert_to_numpy=True).astype("float32")
                _, ids = index.search(q_emb, k=1)
                best_idx = ids[0][0]

            latencies.append((time.time() - start) * 1000)
            retrieved = chunks[best_idx]
            gt_start, gt_end = map(float, q["timestamp"].split(" - "))
            if timestamps_overlap(retrieved["start"], retrieved["end"], gt_start, gt_end):
                correct += 1

        false_positives = 0
        for q in gold_set["unanswerable_questions"]:
            if method == "TF-IDF":
                vectorizer, matrix = resources["tfidf"]
                qv = vectorizer.transform([q["question"]])
                scores = np.dot(matrix, qv.T).toarray().flatten()
                if np.max(scores) > 0.05:
                    false_positives += 1
            elif method == "BM25":
                bm25 = resources["bm25"]
                scores = bm25.get_scores(q["question"].lower().split())
                if np.max(scores) > 1.5:
                    false_positives += 1
            else:
                model, index = resources["faiss"]
                q_emb = model.encode([q["question"]], convert_to_numpy=True).astype("float32")
                D, _ = index.search(q_emb, k=1)
                if D[0][0] < 1.5:
                    false_positives += 1

        results["Method"].append(method)
        results["Accuracy"].append(correct / len(gold_set["answerable_questions"]))
        results["False Positive Rate"].append(false_positives / len(gold_set["unanswerable_questions"]))
        results["Avg Latency (ms)"].append(np.mean(latencies))

    return pd.DataFrame(results)


def analyze_failures(chunks, gold_set, resources):
    print("\n=== Failure Analysis ===")
    model, index = resources["faiss"]
    for q in gold_set["answerable_questions"]:
        q_emb = model.encode([q["question"]], convert_to_numpy=True).astype("float32")
        _, ids = index.search(q_emb, k=1)
        retrieved = chunks[ids[0][0]]
        gt_start, gt_end = map(float, q["timestamp"].split(" - "))
        if not timestamps_overlap(retrieved["start"], retrieved["end"], gt_start, gt_end):
            print(f"\n❌ Question: {q['question']}\nExpected: {q['timestamp']}\nRetrieved: {retrieved['start']} - {retrieved['end']}\n")


if __name__ == "__main__":
    chunks, gold_set, resources = load_resources()
    df = evaluate_retrieval(chunks, gold_set, resources)
    print("\n=== Evaluation Results ===")
    print(df)
    analyze_failures(chunks, gold_set, resources)
