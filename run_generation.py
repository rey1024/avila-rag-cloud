# ===============================================================
# RAG GENERATION PIPELINE (VERSION 2)
# Menggunakan FAISS index yang sudah ada
# Evaluasi: RAGAS + retrieval diagnostics
# ===============================================================

import os
import logging
import json
import yaml
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from openpyxl import load_workbook



from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import requests
import openpyxl

from utils_generation.io_utils import *
from utils_generation.rag_utils import *
from utils_generation.llm_utils import *

import gc



# ===============================================================
# 0. LOAD ENV
# ===============================================================
from dotenv import load_dotenv
load_dotenv()
print("✔ Env loaded from .env")

def run_generation(exp, master, timestamp):
     # Setup logging

    # logging.basicConfig(
    #     filename=log_file,
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S'
    # )
    setup_logging(exp,timestamp)

    logging.info(f"================= Menjalankan {exp['id']} ============================")
    log_var("experiment_config", exp)

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

    logging.info(f"✔ FAISS index loaded: {index_path}")
    try:
        logging.info(f"✔ Chunks from FAISS loaded: {len(chunks)} chunks available")
    except Exception:
        pass


    # -----------------------------
    # Load dataset evaluasi
    # -----------------------------
    data, df_source = load_eval_dataset(master["dataset"])
    logging.info(f"✔ Dataset loaded from: {master['dataset']} (items={len(data)})")

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
    logging.info(f"✔ LLM setup -> provider: {provider}, model: {model_name}, endpoint: {endpoint}, mode: {mode}")

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
    logging.info("• Pre-encoding all chunks for diagnostics...")
    if os.path.exists("cache/chunk_embs.npy"):
        chunk_embs = np.load("cache/chunk_embs.npy")
        logging.info("✔ Loaded chunk_embs dari cache/chunk_embs.npy")
    else:
        logging.info("  → Tidak ditemukan cache, melakukan encoding ulang...")
        chunk_embs = embedder.encode(
            chunks,
            normalize_embeddings=True,
            batch_size=8,           # aman untuk GPU 10 GB
            convert_to_numpy=True
        )
    logging.info("✔ Pre-encoding done")
    log_var("chunk_embs_shape", chunk_embs.shape)


    total_items = len(data)
    logging.info(f"Mulai proses {total_items} item...")
    faiss_queries = []

    # Penjelasan singkat: LOOP 1 melakukan retrieval (FAISS -> rerank),
    # menghitung diagnostic similarity, dan menyimpan konteks untuk tiap item.
    logging.info("Penjelasan: LOOP 1 = retrieval + diagnostics (tidak ada pemanggilan LLM).")
    logging.info(f"Columns: {df_source.columns.tolist()}")
    logging.info(f"sample data item: {data[0]}")

    # =====================================================
    # LOOP 1: hanya retrieval dan diagnostics, TANPA LLM
    # =====================================================
    for idx, item in enumerate(data):
        #if idx % 10 == 0:
        logging.info(f"\n############## Memproses item {idx}/{total_items}")

        q      = item["question"]
        gt     = item["answer_groundtruth"]
        #gt_ctx = find_gt_chunk(gt, chunks, embedder)

        gt_ctx = item["context"]


        faiss_queries.append(q)   # log query yang dikirim ke faiss


        # 1) retrieval
        logging.info("  → Retrieval detail:")
        log_var("###question", q)
        log_var("###groundtruth_answer", gt)
        log_var("###groundtruth_context", gt_ctx)
        
   
        # Step 1: FAISS retrieval top10 dulu
        faiss_top10 = faiss_retrieve(q, embedder, index, chunks, top_k=10)
        log_var("faiss_top10", faiss_top10)

        # Step 2: Rerank dengan BGE atau Jina
        retrieved = rerank_bge(q, faiss_top10, top_k=2)
        log_var("retrieved_after_rerank", retrieved)
        if len(retrieved) == 0:
            logging.info(f"[RETRIEVAL] Item {idx} retrieved chunk = 0")
            logging.info(f"Q: {q}")
            logging.info(f"FAISS top10: {faiss_top10}")

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

        log_var("sims chunk grountruth vs chunk retrieved", sims)

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
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # =====================================================
    # LOOP 2: LLM generation (paralel untuk Ollama)
    # =====================================================
    logging.info("\nMulai tahap LLM generation...")
    logging.info("Penjelasan: LOOP 2 = Menciptakan jawaban berdasarkan pertanyaan dari konteks yang sudah dikumpulkan dan memanggil LLM untuk tiap item.")

    if llm is None:
        # mode Ollama → paralel
        batch_items = []
        for i in range(total_items):
            logging.info(f"  → LLM Memproses item {i}/{total_items}")
            retrieved = contexts_used[i]
            q         = questions[i]

            safe_ctx  = safe_join_context(retrieved, max_tokens=6000)
            bloom = data[i]["bloom_level"]
            prompt = build_prompt(safe_ctx, q, bloom)

            # logging per-prompt
            logging.info(f"    * prompt_length={len(prompt)} chars")
            logging.info(f"    * prompt_preview={prompt[:300].replace('\n',' ') }")

            batch_items.append({
                "id": data[i]["id"],
                "prompt": prompt,
            })

        # jalankan paralel
        logging.info(f"  → Menjalankan batch async ke endpoint {endpoint} (model={model_name}), batch_size={exp['llm'].get('batch_size', 6)})")
        answers = asyncio.run(
            process_batch(
                batch_items,
                endpoint=endpoint,
                model=model_name,
                batch_size=exp["llm"].get("batch_size", 6)
            )
        )
        predictions = answers
        log_var("predictions_sample", predictions[:300])
    else:
        # mode GPT → tetap sequential (atau bisa kamu paralelkan dengan mekanisme lain)
        for i in range(total_items):
            retrieved = contexts_used[i]
            q         = questions[i]
            #safe_ctx  = safe_join_context(retrieved, max_tokens=6000)
            safe_ctx = "\n".join(retrieved)
            #contexts_used.append(retrieved)


            print(f"  → GPT mode generate item {i}/{total_items}: prompt_len={len(safe_ctx)}")
            ans = llm_generate_single(safe_ctx, q, data[i]["bloom_level"])

            log_var("prediction_preview", str(ans)[:300])
            predictions.append(ans)
    # =====================================================
    #  LLM generation selesai → SIMPAN DULU KE EXCEL
    # =====================================================

    df_llm = pd.DataFrame({
        "id": [d["id"] for d in data],
        "question": questions,
        "llm_answer": predictions,
        "reference": references,
        "contexts_used": [ "\n".join(ctx) for ctx in contexts_used ],
        "bloom_level": [d["bloom_level"] for d in data]
    })

    filename_v1 = f"outputGen/Gen{timestamp}/{exp['id']}-v1-llmAnswer.xlsx"
    os.makedirs(f"outputGen/Gen{timestamp}", exist_ok=True)

    df_llm.to_excel(filename_v1, index=False)
    format_excel(filename_v1)
    logging.info(f"✔ Saved intermediate LLM answers to {filename_v1}")
    print(f"✔ Saved LLM-only results: {filename_v1}")

    # =====================================================
    # RAGAS
    # =====================================================
    from utils_generation.eval_ragas_async import run_ragas_evaluation

    logging.info("Menjalankan RAGAS evaluation...")
    contexts_used_ragas = [
        ctx
        for ctx in contexts_used
    ]


    ragas_item, ragas_agg = run_ragas_evaluation(
        questions=questions,
        predictions=predictions,
        contexts_used=contexts_used_ragas,
        references=references,
        openai_key=os.getenv("OPENAI_API_KEY")
    )

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    logging.info("..Selesai...\n\n")
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
        faiss_queries,
        chunk_embs
    )

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

        result = run_generation(exp, master, timestamp)

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
            faiss_queries,
            chunk_embs
        ) = result

        out_dir = auto_output_folder(exp_id, timestamp)
        embedder = embedder.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

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
            faiss_queries,
            chunk_embs,          
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