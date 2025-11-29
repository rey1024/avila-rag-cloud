import os
import logging
import json
import yaml
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


from sentence_transformers import SentenceTransformer, util
import torch

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from utils_generation.io_utils import *
import requests

def load_faiss_index(path):
    if not os.path.exists(path):
        raise ValueError(f"FAISS index tidak ditemukan: {path}")
    return faiss.read_index(path)


def load_faiss_chunks(index_path):
    base = Path(index_path).parent
    jsonl_path = base / "chunks.jsonl"

    if jsonl_path.exists():
        chunks = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    json_path = base / "chunks.json"
    if json_path.exists():
        return json.load(open(json_path, "r", encoding="utf-8"))

    raise ValueError("chunks.json[l] tidak ditemukan")

def load_embedder(model_name, device="cuda"):
    model = SentenceTransformer(model_name)
    model.to(device)
    return model

def faiss_retrieve(query, embedder, index, chunks, top_k=5):
    q_vec = embedder.encode([query], normalize_embeddings=True)
    q_np = to_numpy(q_vec)
    scores, idx = index.search(q_np, top_k)
    return [chunks[i] for i in idx[0]]

def estimate_tokens(text):
    # estimator sederhana: 1 token ≈ 0.75 kata
    # cocok untuk model LLaMA/Gemma
    words = len(text.split())
    return int(words / 0.75)
def safe_join_context(retrieved, max_tokens=6000):
    """
    Memotong context jika melebihi batas token llm (misalnya 8k).
    """
    joined = ""
    total = 0

    for chunk in retrieved:
        c_tok = estimate_tokens(chunk)
        if total + c_tok > max_tokens:
            continue
        joined += chunk + "\n"
        total += c_tok

    return joined.strip()

def build_prompt(context, question, bloom_level):
    return f"""
Anda adalah generator jawaban berbasis konteks.

Aturan:
1. Jawaban harus sepenuhnya berasal dari konteks.
2. Tidak boleh menambah asumsi, generalisasi, atau informasi baru.
3. Gunakan kalimat dari konteks sejauh mungkin.
4. Jika perlu parafrase, gunakan parafrase minimal tanpa mengubah makna.
5. Tidak boleh menyimpulkan hal yang tidak eksplisit di konteks.
6. Jika informasi tidak ditemukan, jawab: "Tidak ditemukan dalam konteks."
Jika ada kalimat yang secara langsung menjawab pertanyaan,
gunakan kalimat tersebut apa adanya tanpa parafrase.

[START KONTEN]
{context}
[END KONTEN]

Pertanyaan:
{question}

Berikan jawaban singkat, akurat, dan sepenuhnya grounded.
"""


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
rerank_tokenizer = AutoTokenizer.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    trust_remote_code=True
)

rerank_model = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    trust_remote_code=True
).to(device)

rerank_model.eval()

def rerank_bge(query, candidate_chunks, top_k=3):
    pairs = [(query, chunk) for chunk in candidate_chunks]

    inputs = rerank_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        scores = rerank_model(**inputs).logits.squeeze(-1)

    # urutkan berdasarkan skor tertinggi
    sorted_idx = torch.argsort(scores, descending=True).tolist()
    return [candidate_chunks[i] for i in sorted_idx[:top_k]]

def find_gt_chunk(answer, chunks, embedder):
    a_emb = embedder.encode([answer], normalize_embeddings=True)
    c_emb = embedder.encode(chunks, normalize_embeddings=True)
    sims = util.cos_sim(a_emb, c_emb)[0].tolist()
    best_idx = int(np.argmax(sims))
    return chunks[best_idx]

    # -----------------------------
    # Load embedder (harus sama dengan saat buat FAISS)
    # -----------------------------
embedder = SentenceTransformer(
        "models/embedding/multilingual-e5-base",
        device="cuda"   # WAJIB untuk GPU kecil
    )


      #"intfloat/multilingual-e5-base",