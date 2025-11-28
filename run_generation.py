# ===============================================================
# RAG GENERATION PIPELINE (VERSION 2)
# Menggunakan FAISS index yang sudah ada
# Evaluasi: RAGAS + retrieval diagnostics
# ===============================================================

import os
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

import requests
import openpyxl


# ===============================================================
# 0. LOAD ENV
# ===============================================================
from dotenv import load_dotenv
load_dotenv()
print("✔ Env loaded from .env")


# ===============================================================
# 1. BASIC UTILS
# ===============================================================

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def fmt(v, d=2):
    try:
        return round(float(v), d)
    except:
        return 0.0

def log_var(name, v, preview=200):
    """Print a compact summary of a variable: type, size/shape and a small preview."""
    try:
        t = type(v).__name__
        size = None
        preview_val = None

        # numpy / torch
        if hasattr(v, "shape"):
            size = getattr(v, "shape")
        elif isinstance(v, (list, tuple, dict)):
            size = len(v)
        elif isinstance(v, str):
            size = len(v)

        # helper to keep only first N words
        def first_n_words(s, n=5):
            try:
                words = str(s).split()
                if len(words) > n:
                    return " ".join(words[:n]) + "..."
                return " ".join(words)
            except Exception:
                return str(s)[:preview]

        # preview
        if isinstance(v, str):
            preview_val = first_n_words(v, 5)
        elif isinstance(v, dict):
            try:
                # show up to 3 key:shortvalue pairs
                items = list(v.items())[:3]
                preview_val = {k: first_n_words(vv, 5) for k, vv in items}
            except Exception:
                preview_val = str(list(v.keys())[:3])
        elif isinstance(v, (list, tuple)):
            # if list of strings, trim each element to first 5 words
            try:
                sample = v[:3]
                preview_list = []
                for el in sample:
                    if isinstance(el, str):
                        preview_list.append(first_n_words(el, 5))
                    else:
                        preview_list.append(el)
                preview_val = preview_list
            except Exception:
                preview_val = str(v)[:preview]
        else:
            # try numpy conversion
            try:
                arr = np.asarray(v)
                preview_flat = arr.flatten()
                preview_val = preview_flat[:3].tolist()
            except Exception:
                preview_val = str(v)[:preview]

        print(f"    - {name}: type={t}, size={size}, preview={preview_val}")
    except Exception as e:
        print(f"    - {name}: (error summarizing: {e})")

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

import asyncio
import aiohttp

async def async_generate_ollama(session, prompt, endpoint, model):
    url = endpoint.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        async with session.post(url, json=payload, timeout=120) as r:
            j = await r.json()
            return j.get("response", "")
    except Exception as e:
        return f"__ERROR__: {str(e)}"

async def process_batch(items, endpoint, model, batch_size=6):
    async with aiohttp.ClientSession() as session:
        tasks = []

        for item in items:
            tasks.append(async_generate_ollama(
                session,
                item["prompt"],
                endpoint,
                model
            ))

        # batched parallel execution
        results = []
        for i in range(0, len(tasks), batch_size):
            sub = tasks[i:i+batch_size]
            out = await asyncio.gather(*sub)
            results.extend(out)

        return results

def auto_output_folder(exp_id, timestamp):
    folder = f"outputGen/Gen{timestamp}/{exp_id}_{timestamp}"
    ensure_dir(folder)
    return folder



# ===============================================================
# 2. LOAD PDF / DOCX
# ===============================================================

def load_pdf_texts(pdf_dir):
    import pypdf
    from docx import Document

    base = Path(pdf_dir)
    pdfs = list(base.glob("*.pdf"))
    docs = list(base.glob("*.docx"))

    if not pdfs and not docs:
        raise ValueError(f"Tidak ada file PDF/DOCX di {pdf_dir}")

    texts = []

    for p in pdfs:
        reader = pypdf.PdfReader(str(p))
        t = " ".join(page.extract_text() or "" for page in reader.pages)
        texts.append(t)

    for d in docs:
        doc = Document(str(d))
        lines = [p.text for p in doc.paragraphs if p.text]
        texts.append("\n".join(lines))

    return "\n".join(texts)


# ===============================================================
# 3. LOAD PREBUILT FAISS INDEX
# ===============================================================

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

def load_embedder(model_name, device="cpu"):
    model = SentenceTransformer(model_name)
    model.to(device)
    return model


# ===============================================================
# 4. LLM GENERATION
# ===============================================================

def generate_with_ollama(prompt, endpoint, model):
    url = endpoint.rstrip("/") + "/api/generate"
    r = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    }, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "")


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


# ===============================================================
# 5. LOAD DATASET
# ===============================================================

def load_eval_dataset(path):
    df = pd.read_excel(path)
    
    # Ambil 5 teratas dan 5 terbawah
    # df_top = df.head(5)
    # df_bottom = df.tail(1)
    # df = pd.concat([df_top, df_bottom], ignore_index=True)

    df = df[df["id"] == 2].copy()


    df.columns = df.columns.str.lower()

    required = ["id", "context", "question", "answer","bloomlevel"]
    for c in required:
        if c not in df:
            raise ValueError(f"Kolom {c} tidak ada di dataset evaluasi")

    data = []
    for _, r in df.iterrows():
        data.append({
            "id": r["id"],
            "context": r["context"],
            "question": r["question"],
            "groundtruth_answer": r["answer"],
            "bloomlevel": r["bloomlevel"]
        })


    print(f"Dataset evaluasi: {len(data)} item")
    return data, df

def to_numpy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)

# ===============================================================
# 6. RETRIEVAL
# ===============================================================

def faiss_retrieve(query, embedder, index, chunks, top_k=5):
    q_vec = embedder.encode([query], normalize_embeddings=True)
    q_np = to_numpy(q_vec)
    scores, idx = index.search(q_np, top_k)
    return [chunks[i] for i in idx[0]]



# ===============================================================
# 7. RAGAS + DIAGNOSTICS + EXCEL EXPORT
# ===============================================================

def save_excel_report(
    exp_id, out_dir, timestamp,
    df_source,
    questions,
    predictions,
    references,
    contexts_used,
    ragas_scores_item,
    ragas_agg,
    chunks,
    chunk_ids,
    retrieval_hits,
    semantic_scores,
    embedder,
    gt_chunk_ids,
    gt_chunk_retrieved,
    gt_chunk_rank,
    gt_chunk_rr,
    gt_topk_sims,
    faiss_queries   # <=== baru

):


    excel_path = f"{out_dir}/{exp_id}_{timestamp}.xlsx"


    df_items = df_source.copy()

    chunk_to_id = {chunks[i]: chunk_ids[i] for i in range(len(chunks))}


    # -------------------------
    # Format contexts
    # -------------------------
    formatted = []
    retrieved_ids_list = []
    for retrieved in contexts_used:
        ids = [chunk_to_id[c] for c in retrieved]
        retrieved_ids_list.append(ids)
        combined = "\n".join([f"[{cid}] {txt}" for cid, txt in zip(ids, retrieved)])
        formatted.append(combined)
    df_items["faiss_query"] = faiss_queries

    df_items["contexts_retrieval"] = formatted
    df_items["retrieved_chunk_ids"] = retrieved_ids_list
    df_items["retrieved_similarity"] = semantic_scores

    # -------------------------
    # Answers
    # -------------------------
    df_items["llm_answer"] = predictions

    df_items["context_precision"] = [x["context_precision"] for x in ragas_scores_item]
    df_items["context_recall"] = [x["context_recall"] for x in ragas_scores_item]
    df_items["answer_relevancy"] = [x["answer_relevancy"] for x in ragas_scores_item]
    df_items["faithfulness"] = [x["faithfulness"] for x in ragas_scores_item]
    df_items["answer_correctness"] = [x["answer_correctness"] for x in ragas_scores_item]

    df_items["retrieval_hit"] = retrieval_hits
    df_items["gt_chunk_id"] = gt_chunk_ids
    df_items["gt_chunk_retrieved"] = gt_chunk_retrieved
    df_items["gt_chunk_rank"] = gt_chunk_rank
    df_items["gt_chunk_rr"] = gt_chunk_rr

    # -------------------------
    # Summary
    # -------------------------
    df_sum = pd.DataFrame({
        "metric": list(ragas_agg.keys()),
        "score": list(ragas_agg.values())
    })

    # -------------------------
    # CHUNKS SHEET
    # -------------------------
    chunk_embs = to_numpy(embedder.encode(chunks, normalize_embeddings=True))

    context_embs = embedder.encode(df_source["context"].tolist(), normalize_embeddings=True)
    sim_mat = util.cos_sim(chunk_embs, context_embs).cpu().numpy()

    df_chunks = pd.DataFrame({
        "chunk_id": range(1, len(chunks)+1),
        "chunk_text": chunks,
        "chars": [len(c) for c in chunks],
        "tokens": [len(c.split()) for c in chunks],
        "top_groundtruth_similarity": sim_mat.max(axis=1)
    })

    # -------------------------
    # Write
    # -------------------------
    with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
        df_items.to_excel(w, index=False, sheet_name="results")
        df_sum.to_excel(w, index=False, sheet_name="summary")
        df_chunks.to_excel(w, index=False, sheet_name="chunks")

        # ----------------------------------------
        # APPLY FORMATTING (wrap, align, auto width)
        # ----------------------------------------
        wb = w.book
        from openpyxl.styles import Alignment

        for sheet_name in ["results", "summary", "chunks"]:
            ws = wb[sheet_name]
                # === Freeze Top Row ===
            ws.freeze_panes = "A2"
            # wrap + alignment
            for row in ws.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(
                        wrap_text=True,
                        vertical="top",
                        horizontal="left"
                    )

            # auto width per column
            for col in ws.columns:
                max_length = 0
                col_letter = col[0].column_letter

                for cell in col:
                    val = str(cell.value) if cell.value is not None else ""
                    if len(val) > max_length:
                        max_length = len(val)

                ws.column_dimensions[col_letter].width = min(max_length + 3, 80)



    print(f"✔ Excel tersimpan: {excel_path}")
    return excel_path

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

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


# ===============================================================
# 8. RAG PIPELINE (FULL GENERATION MODE)
# ===============================================================

def run_generation(exp, master):
    print(f"\n=== Menjalankan {exp['id']} ===")

    # -----------------------------
    # Load FAISS index dan chunks
    # -----------------------------
    index_path = master["faiss_index_dir"] + "/index.faiss"
    index = load_faiss_index(index_path)
    if hasattr(index, "gpu_resources"):
        index = faiss.index_gpu_to_cpu(index)
    raw_chunks = load_faiss_chunks(index_path)
    chunks     = [c["text"] for c in raw_chunks]
    chunk_ids  = [c["chunk_id"] for c in raw_chunks]

    print(f"✔ FAISS index loaded: {index_path}")
    try:
        print(f"✔ Chunks loaded: {len(chunks)} chunks available")
    except Exception:
        pass

    # -----------------------------
    # Load embedder (harus sama dengan saat buat FAISS)
    # -----------------------------
    embedder = SentenceTransformer(
        "intfloat/multilingual-e5-base",
        device="cpu"   # WAJIB untuk GPU kecil
    )
    print(f"✔ Embedder ready: intfloat/multilingual-e5-base (device=cpu)")


    # -----------------------------
    # Load dataset evaluasi
    # -----------------------------
    data, df_source = load_eval_dataset(master["dataset"])
    print(f"✔ Dataset loaded from: {master['dataset']} (items={len(data)})")

    # -----------------------------
    # LLM setup
    # -----------------------------
    provider   = exp["llm"]["provider"]
    model_name = exp["llm"]["model"]
    endpoint   = master.get("endpoint", "http://localhost:11434")

    if model_name.startswith("gpt") or "4o" in model_name:
        llm = ChatOpenAI(
            model=model_name,
            temperature=exp["llm"]["temperature"],
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        llm = None  # berarti pakai Ollama

    mode = "Ollama (local)" if llm is None else "OpenAI/ChatOpenAI"
    print(f"✔ LLM setup -> provider: {provider}, model: {model_name}, endpoint: {endpoint}, mode: {mode}")

    def llm_generate_single(ctx, q, bloom):
        prompt = build_prompt(ctx, q, bloom)
        if llm is not None:
            return llm.invoke(prompt).content
        return generate_with_ollama(prompt, endpoint, model_name)


    # -----------------------------
    # STORAGE
    # -----------------------------
    questions = []
    predictions = []
    references = []
    contexts_used = []

    retrieval_hits = []
    semantic_scores = []

    gt_chunk_ids = []
    gt_chunk_retrieved = []
    gt_chunk_rank = []
    gt_chunk_rr = []
    gt_topk_sims = []

    # Pre encode semua chunks untuk diagnostics
    from sentence_transformers import util
    print("• Pre-encoding all chunks for diagnostics...")
    chunk_embs = embedder.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=8,           # aman untuk GPU 10 GB
        convert_to_numpy=True
    )
    print("✔ Pre-encoding done")


    total_items = len(data)
    print(f"Mulai proses {total_items} item...\n")
    faiss_queries = []

    # Penjelasan singkat: LOOP 1 melakukan retrieval (FAISS -> rerank),
    # menghitung diagnostic similarity, dan menyimpan konteks untuk tiap item.
    print("Penjelasan: LOOP 1 = retrieval + diagnostics (tidak ada pemanggilan LLM).")

    # =====================================================
    # LOOP 1: hanya retrieval dan diagnostics, TANPA LLM
    # =====================================================
    for idx, item in enumerate(data):
        #if idx % 10 == 0:
        print(f"  → Memproses item {idx}/{total_items}")

        q      = item["question"]
        gt     = item["groundtruth_answer"]
        gt_ctx = find_gt_chunk(gt, chunks, embedder)

        faiss_queries.append(q)   # log query yang dikirim ke faiss


        # 1) retrieval
        print("  → Retrieval detail:")
        log_var("question", q)
        log_var("groundtruth_answer", gt)
        log_var("groundtruth_context", gt_ctx)
        
        # retrieved = sorted(
        #     faiss_retrieve(q, embedder, index, chunks, top_k=master["top_k"]),
        #     key=lambda c: util.cos_sim(embedder.encode(c), embedder.encode(q)),
        #     reverse=True
        # )

        # Step 1: FAISS retrieval top10 dulu
        faiss_top10 = faiss_retrieve(q, embedder, index, chunks, top_k=10)
        log_var("faiss_top10", faiss_top10)

        # Step 2: Rerank dengan BGE atau Jina
        retrieved = rerank_bge(q, faiss_top10, top_k=2)
        log_var("retrieved_after_rerank", retrieved)
        # atau
        # retrieved = rerank_jina(q, faiss_top10, top_k=3)



        contexts_used.append(retrieved)
        # additional per-item retrieval info
        try:
            retrieved_ids = [chunk_ids[chunks.index(c)] for c in retrieved]
        except Exception:
            retrieved_ids = []
        log_var("retrieved_ids", retrieved_ids)
        questions.append(q)
        references.append(gt)

        # 2) retrieval diagnostics
        retrieved_embs = embedder.encode(
            retrieved,
            normalize_embeddings=True,
            batch_size=8
        )
        log_var("retrieved_embs", retrieved_embs)

        gt_emb = embedder.encode(
            gt_ctx,
            normalize_embeddings=True,
            batch_size=8
        )
        log_var("gt_emb", gt_emb)


        sims = util.cos_sim(
            torch.tensor(gt_emb),
            torch.tensor(retrieved_embs)
        )[0].tolist()

        log_var("sims", sims)

        semantic_scores.append(sims)
        log_var("semantic_scores_current", semantic_scores[-1])
        retrieval_hits.append(1 if max(sims) >= 0.3 else 0)
        gt_topk_sims.append(sims)

        # full corpus match
        sims_all = util.cos_sim(gt_emb, chunk_embs)[0].tolist()
        best_idx = int(np.argmax(sims_all))
        cid = chunk_ids[best_idx]
        gt_chunk_ids.append(cid)

        retrieved_ids = []
        for c in retrieved:
            rid = chunks.index(c)
            retrieved_ids.append(chunk_ids[rid])

        gt_chunk_retrieved.append(1 if cid in retrieved_ids else 0)

        sorted_rank = np.argsort(sims_all)[::-1]
        # posisi best_idx dalam ranking
        rank = int(np.where(sorted_rank == best_idx)[0][0]) + 1
        gt_chunk_rank.append(rank)
        gt_chunk_rr.append(round(1.0 / rank, 4))

    # =====================================================
    # LOOP 2: LLM generation (paralel untuk Ollama)
    # =====================================================
    print("\nMulai tahap LLM generation...")
    print("Penjelasan: LOOP 2 = membangun prompt dari konteks yang sudah dikumpulkan dan memanggil LLM untuk tiap item (paralel untuk Ollama).")

    if llm is None:
        # mode Ollama → paralel
        batch_items = []
        for i in range(total_items):
            print(f"  → LLM Memproses item {i}/{total_items}")
            retrieved = contexts_used[i]
            q         = questions[i]

            safe_ctx  = safe_join_context(retrieved, max_tokens=6000)
            bloom = data[i]["bloomlevel"]
            prompt = build_prompt(safe_ctx, q, bloom)

            # logging per-prompt
            print(f"    * prompt_length={len(prompt)} chars")
            print(f"    * prompt_preview={prompt[:300].replace('\n',' ') }")

            batch_items.append({
                "id": data[i]["id"],
                "prompt": prompt,
            })

        # jalankan paralel
        print(f"  → Menjalankan batch async ke endpoint {endpoint} (model={model_name}), batch_size={exp['llm'].get('batch_size', 6)})")
        answers = asyncio.run(
            process_batch(
                batch_items,
                endpoint=endpoint,
                model=model_name,
                batch_size=exp["llm"].get("batch_size", 6)
            )
        )
        predictions = answers
        log_var("predictions_sample", predictions[:3])
    else:
        # mode GPT → tetap sequential (atau bisa kamu paralelkan dengan mekanisme lain)
        for i in range(total_items):
            retrieved = contexts_used[i]
            q         = questions[i]
            #safe_ctx  = safe_join_context(retrieved, max_tokens=6000)
            safe_ctx = "\n".join(retrieved)
            #contexts_used.append(retrieved)


#            print(f"  → GPT mode generate item {i}/{total_items}: prompt_len={len(safe_ctx)}")
            ans = llm_generate_single(safe_ctx, q, data[i]["bloomlevel"])

            log_var("prediction_preview", str(ans)[:300])
            predictions.append(ans)

    # =====================================================
    # RAGAS
    # =====================================================
    from utils.eval_ragas import run_ragas_evaluation

    print("\n• Menjalankan RAGAS evaluation...")

    ragas_item, ragas_agg = run_ragas_evaluation(
        questions=questions,
        predictions=predictions,
        contexts_used=contexts_used,
        references=references,
        openai_key=os.getenv("OPENAI_API_KEY")
    )

    import gc
    

    gc.collect()
    torch.cuda.empty_cache()


    return (
        df_source,
        questions, predictions, references,
        contexts_used,
        ragas_item, ragas_agg,
        chunks,
        chunk_ids,
        retrieval_hits, semantic_scores,
        embedder,
        gt_chunk_ids, gt_chunk_retrieved, gt_chunk_rank, gt_chunk_rr, gt_topk_sims,
        faiss_queries   # <=== tambahan
    )
def save_global_summary(summary_dict, timestamp):
    """
    Membuat 1 Excel summary keseluruhan eksperimen.
    summary_dict adalah dictionary berikut:

    summary[exp_id] = {
        "status": "ok",
        "output_dir": "...",
        "excel": "....xlsx",
        "scores": { ... ragas_agg ... }
    }
    """
    rows = []

    for exp_id, info in summary_dict.items():
        scores = info["scores"]

        rows.append({
            "experiment": exp_id,
            "context_precision": scores.get("context_precision", 0),
            "context_recall": scores.get("context_recall", 0),
            "answer_relevancy": scores.get("answer_relevancy", 0),
            "faithfulness": scores.get("faithfulness", 0),
            "answer_correctness": scores.get("answer_correctness", 0),
            "output_dir": info["output_dir"],
            "excel_file": info["excel"]
        })

    df = pd.DataFrame(rows)

    out_path = f"outputGen/Gen{timestamp}/summary_all_{timestamp}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="summary_all")

        # formatting
        ws = w.book["summary_all"]
        from openpyxl.styles import Alignment

        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(
                    wrap_text=True,
                    vertical="top",
                    horizontal="left"
                )

        # auto width
        for col in ws.columns:
            max_length = 0
            col_letter = col[0].column_letter
            for cell in col:
                val = str(cell.value) if cell.value is not None else ""
                if len(val) > max_length:
                    max_length = len(val)
            ws.column_dimensions[col_letter].width = min(max_length + 3, 80)

    print(f"✔ Global summary Excel tersimpan: {out_path}")
    return out_path

# ===============================================================
# 9. MAIN
# ===============================================================
def main():
    cfg = load_yaml("configs/generation.yaml")
    master = cfg["master"]

    summary = {}

    # timestamp global untuk seluruh eksperimen
    global_timestamp = get_timestamp()

    # buat folder utama
    ensure_dir(f"outputGen/Gen{global_timestamp}")

    for exp in cfg["experiments"]:
        if not exp.get("active", False):
            continue

        exp_id = exp["id"]
        timestamp = global_timestamp   # PENTING: satu timestamp saja

        result = run_generation(exp, master)

        (
            df_source,
            questions, predictions, references,
            contexts_used,
            ragas_item, ragas_agg,
            chunks, chunk_ids,
            retrieval_hits, semantic_scores,
            embedder,
            gt_chunk_ids, gt_chunk_retrieved, gt_chunk_rank,
            gt_chunk_rr, gt_topk_sims,
            faiss_queries
        ) = result

        out_dir = auto_output_folder(exp_id, timestamp)

        excel = save_excel_report(
            exp_id, out_dir, timestamp,
            df_source,
            questions, predictions, references,
            contexts_used,
            ragas_item, ragas_agg,
            chunks,
            chunk_ids,
            retrieval_hits, semantic_scores,
            embedder,
            gt_chunk_ids, gt_chunk_retrieved,
            gt_chunk_rank, gt_chunk_rr,
            gt_topk_sims,
            faiss_queries          
        )

        summary[exp_id] = {
            "status": "ok",
            "output_dir": out_dir,
            "excel": excel,
            "scores": ragas_agg
        }

    # summary JSON
    with open(f"outputGen/Gen{global_timestamp}/summary_{global_timestamp}.json",
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # summary excel
    global_summary_path = save_global_summary(summary, global_timestamp)

    print("\n=== Semua eksperimen selesai ===")
    print(f"Global summary: {global_summary_path}")



if __name__ == "__main__":
    main()
