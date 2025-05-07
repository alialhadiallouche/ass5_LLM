# 🎥 Multimodal Video Question Answering (QA) System

This project implements a **Video QA system** that allows users to ask natural language questions and retrieve the most relevant video segment based on semantic and lexical information. It supports multiple retrieval methods (FAISS, TF-IDF, BM25), visual indexing, and a Streamlit-based UI.

---

## 📌 Features

- 🔊 Transcription using OpenAI Whisper
- 📖 Re-chunked transcript into ~10 second blocks
- 🎞️ Keyframe extraction from video
- 🧠 Dense embeddings using `e5-large-v2` and OpenCLIP
- ⚡ Fast retrieval using FAISS
- 📊 Evaluation with a gold test set (answerable + unanswerable questions)
- 🌐 Clean Streamlit interface for interactive search
- 🧪 Quantitative evaluation across FAISS, TF-IDF, BM25

---

## 🔧 Setup Instructions
copy the github link and paste it in streamlit cloud


for Preparing Data:
The following steps are handled in Ass5_ana84.ipynb:

Transcribe video with Whisper

Extract frames with OpenCV

Generate chunked_transcript.json and frame_text_pairs.json

Encode text and images with sentence transformers and OpenCLIP

Build FAISS indices

Evaluation



🧪 Gold Test Set
Includes:

10 answerable questions with ground truth timestamps

5 unanswerable questions to evaluate rejection ability
