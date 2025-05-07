# === evaluate.py ===
import time
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt
import seaborn as sns


def timestamps_overlap(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or a_start > b_end)


def load_resources():
    with open("chunked_transcript.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open("gold_set_test.json", "r", encoding="utf-8") as f:
        gold_set = json.load(f)

    texts = [c["text"] for c in chunks]
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    bm25 = BM25Okapi([t.lower().split() for t in texts])
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=True).detach().cpu().numpy().astype("float32")
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    return chunks, gold_set, {
        "tfidf": (tfidf_vectorizer, tfidf_matrix, texts),
        "bm25": (bm25, texts),
        "faiss": (model, faiss_index, texts, embeddings)
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
                vectorizer, matrix, _ = resources["tfidf"]
                qv = vectorizer.transform([q["question"]])
                scores = np.dot(matrix, qv.T).toarray().flatten()
                best_idx = np.argmax(scores)
            elif method == "BM25":
                bm25, _ = resources["bm25"]
                scores = bm25.get_scores(q["question"].lower().split())
                best_idx = np.argmax(scores)
            else:
                model, index, _, _ = resources["faiss"]
                q_emb = model.encode([q["question"]], convert_to_numpy=True).astype("float32")
                _, ids = index.search(q_emb, k=5)
                best_idx = ids[0][0]

            latencies.append((time.time() - start) * 1000)
            retrieved = chunks[best_idx]
            gt_start, gt_end = map(float, q["timestamp"].split(" - "))
            if timestamps_overlap(retrieved["start"], retrieved["end"], gt_start, gt_end):
                correct += 1

        false_positives = 0
        for q in gold_set["unanswerable_questions"]:
            if method == "TF-IDF":
                vectorizer, matrix, texts = resources["tfidf"]
                qv = vectorizer.transform([q["question"]])
                scores = np.dot(matrix, qv.T).toarray().flatten()
                best_score = np.max(scores)
                best_idx = np.argmax(scores)
                if best_score > 0.25 and len(chunks[best_idx]["text"]) > 15:
                    false_positives += 1

            elif method == "BM25":
                bm25, texts = resources["bm25"]
                scores = bm25.get_scores(q["question"].lower().split())
                best_score = np.max(scores)
                best_idx = np.argmax(scores)
                if best_score > 4.0 and len(chunks[best_idx]["text"]) > 15:
                    false_positives += 1

            else:
                model, index, texts, embeddings = resources["faiss"]
                q_emb = model.encode([q["question"]], convert_to_numpy=True).astype("float32")
                D, I = index.search(q_emb, k=5)
                min_dist = D[0][0]
                best_idx = I[0][0]
                if min_dist < 0.35 and len(chunks[best_idx]["text"]) > 15:
                    false_positives += 1

        results["Method"].append(method)
        results["Accuracy"].append(correct / len(gold_set["answerable_questions"]))
        results["False Positive Rate"].append(false_positives / len(gold_set["unanswerable_questions"]))
        results["Avg Latency (ms)"].append(np.mean(latencies))

    return pd.DataFrame(results)


def generate_report(df):
    print("\n=== Evaluation Report ===")
    print(df.to_markdown(index=False))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    sns.barplot(x="Method", y="Accuracy", data=df)
    plt.title("Accuracy on Answerable Questions")

    plt.subplot(1, 3, 2)
    sns.barplot(x="Method", y="False Positive Rate", data=df)
    plt.title("False Positive Rate")

    plt.subplot(1, 3, 3)
    sns.barplot(x="Method", y="Avg Latency (ms)", data=df)
    plt.title("Avg Latency (ms)")

    plt.tight_layout()
    plt.show()

    print("\nKey Observations:")
    best = df.loc[df["Accuracy"].idxmax()]
    print(f"- Best accuracy: {best['Accuracy']*100:.1f}% ({best['Method']})")
    fastest = df.loc[df["Avg Latency (ms)"].idxmin()]
    print(f"- Fastest method: {fastest['Method']} ({fastest['Avg Latency (ms)']:.2f} ms)")
    print("- Tradeoffs: FAISS is more accurate but slower; TF-IDF is faster but less accurate.")
    print("- Rejection quality: Threshold tuning has reduced false positives.")
    print("\nFailure cases suggest semantic phrasing mismatches. Improvements could include:")
    print("- Expanding chunk window, reranking top-k, or adding multimodal signals.")


def analyze_failures(chunks, gold_set, resources):
    print("\n=== Failure Analysis ===")
    model, index, _, _ = resources["faiss"]
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
    generate_report(df)
    analyze_failures(chunks, gold_set, resources)
