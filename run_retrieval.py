import os
import yaml
import faiss
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import openpyxl
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# ================================================
# LOAD PDF/DOCX
# ================================================
def load_pdf_docx(folder):
    import pypdf
    from docx import Document
    path = Path(folder)

    pdfs = list(path.glob("*.pdf"))
    docs = list(path.glob("*.docx"))

    texts = []

    for p in pdfs:
        reader = pypdf.PdfReader(str(p))
        t = " ".join(page.extract_text() or "" for page in reader.pages)
        texts.append(t)

    for d in docs:
        doc = Document(str(d))
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        texts.append("\n".join(paragraphs))

    if not texts:
        raise ValueError("Tidak ada PDF/DOCX ditemukan")

    return "\n".join(texts)

def format_decimal(ws):
    for row in ws.iter_rows():
        for cell in row:
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.00"


# ================================================
# YAML UTILS
# ================================================
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def auto_out(exp_id):
   
    path = f"outputsRetrieval/{stamp}/{exp_id}_{stamp}"
    ensure_dir(path)
    return path


# ================================================
# CHUNKING
# ================================================
def fixed_chunk(text, size, overlap):
    tokens = text.split()
    step = size - overlap
    chunks = []
    for i in range(0, len(tokens), step):
        seg = tokens[i:i + size]
        if seg:
            chunks.append(" ".join(seg))
    return chunks


def semantic_chunk_langchain(text, embedder, 
                             min_chunk_size=150, 
                             max_chunk_size=450,
                             similarity_threshold=0.75):

    # 1. Tokenisasi kalimat
    sents = sent_tokenize(text)
    if len(sents) == 0:
        return []

    # 2. Embedding semua kalimat
    sent_embs = embedder.encode(sents, normalize_embeddings=True, batch_size=4)

    chunks = []
    current_chunk = []
    current_len = 0

    for i in range(len(sents)):
        sent = sents[i]
        sent_len = len(sent.split())

        # Jika chunk masih kosong, mulai chunk baru
        if not current_chunk:
            current_chunk.append(sent)
            current_len += sent_len
            continue

        # 3. Hitung similarity ke kalimat sebelumnya
        sim = np.dot(sent_embs[i], sent_embs[i - 1])

        # Jika similarity turun → boundary potensial
        boundary = sim < similarity_threshold

        # Jika ukuran chunk sudah besar → wajib boundary
        too_big = current_len + sent_len > max_chunk_size

        # Jika boundary atau chunk terlalu panjang → simpan chunk
        if boundary or too_big:
            # Jika chunk terlalu kecil → jangan dipotong
            if current_len < min_chunk_size and not too_big:
                current_chunk.append(sent)
                current_len += sent_len
                continue

            # Simpan chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = sent_len
        else:
            # Tambahkan kalimat
            current_chunk.append(sent)
            current_len += sent_len

    # Sisa chunk terakhir
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_text(text, cfg, embedder=None):
    if cfg["type"] == "fixed":
        return fixed_chunk(text, cfg["size"], cfg["overlap"])

    if cfg["type"] == "sentence":
        return fixed_chunk(text, cfg["window"], cfg["overlap"])

    if cfg["type"] == "semantic":
        return semantic_chunk_langchain(
            text,
            embedder,
            min_chunk_size=cfg.get("min_chunk_size", 150),
            max_chunk_size=cfg.get("max_chunk_size", 450),
            similarity_threshold=cfg.get("threshold", 0.75),
        )


    raise ValueError("Unknown chunking method")

# ======================================================
# BOOSTED GROUNDTRUTH MATCHING UTILITIES
# ======================================================

def token_overlap(a, b):
    sa = set(a.split())
    sb = set(b.split())
    if len(sa) == 0 or len(sb) == 0:
        return 0
    return len(sa & sb) / len(sa | sb)


def map_groundtruth_chunk(context_text, chunks, chunk_embs, gt_emb):
    # -------- METHOD 1: Exact containment --------
    contain_candidates = []
    for i, ch in enumerate(chunks):
        if context_text.strip() in ch:
            contain_candidates.append(i)

    if contain_candidates:
        return contain_candidates[0]

    # -------- METHOD 2: Token Overlap --------
    overlaps = [token_overlap(context_text, c) for c in chunks]
    overlap_best = int(np.argmax(overlaps))
    overlap_score = overlaps[overlap_best]

    # -------- METHOD 3: Semantic Similarity --------
    sims = np.dot(chunk_embs, gt_emb)
    sem_best = int(np.argmax(sims))
    sem_score = sims[sem_best]

    # -------- RULE-BASED DECISION --------
    if overlap_score > 0.15:
        return overlap_best

    return sem_best

def detect_misalignment(row):
    hit = row["hit_at_k"]
    cluster = row["cluster_hit@k"]
    text_hit = row["text_hit@k"]

    # Tidak ditemukan sama sekali
    if text_hit == 0:
        return "NOT_FOUND"

    # ID match sempurna
    if hit == 1:
        return "ALIGNED"

    # Cluster dekat (±1)
    if cluster == 1:
        return "SHIFTED_NEAR"

    # Teks match tetapi ID tidak ditemukan sama sekali → shifted jauh
    if hit == 0 and text_hit == 1:
        return "SHIFTED"

    return "UNKNOWN"

def cluster_neighbors(idx, total, radius=1):
    neighbors = set()
    for r in range(-radius, radius + 1):
        j = idx + r
        if 0 <= j < total:
            neighbors.add(j)
    return neighbors


def ndcg_at_k(retrieved_ids, gt_id, k):
    gains = [1 if rid == gt_id else 0 for rid in retrieved_ids[:k]]
    dcg = 0

    if gains:
        dcg += gains[0]
        for i, g in enumerate(gains[1:], start=2):
            dcg += g / np.log2(i)

    ideal = 1  # because only one relevant item exists
    return dcg / ideal

# ======================================================
# TEXT-BASED HIT MATCHING (exact, overlap, semantic)
# ======================================================
def text_hit_match(gt_text, retrieved_chunks, chunk_embs, gt_emb, 
                   overlap_threshold=0.25, 
                   semantic_threshold=0.86):

    best_exact = 0
    best_overlap = 0
    best_sim = 0

    for rid, chunk_text in retrieved_chunks:

        # Exact
        if gt_text.strip() in chunk_text or chunk_text in gt_text:
            best_exact = 1

        # Overlap
        ov = token_overlap(gt_text, chunk_text)
        best_overlap = max(best_overlap, ov)

        # Semantic sim
        sim = float(np.dot(chunk_embs[rid], gt_emb))
        best_sim = max(best_sim, sim)

    # Majority vote
    votes = int(best_exact == 1) + \
            int(best_overlap >= overlap_threshold) + \
            int(best_sim >= semantic_threshold)

    hit = 1 if votes >= 2 else 0

    return hit, best_exact, best_overlap, best_sim



# ================================================
# RETRIEVAL PIPELINE
# ================================================
# ======================================================
# RETRIEVAL PIPELINE (FINAL VERSION)
# ======================================================

def run_retrieval(exp, master, df):
    print(f"\n=== Running experiment {exp['id']} ===")

    embedder = SentenceTransformer(
        exp["embedding"]["model"],
        device="cuda",
        trust_remote_code=True
    )

    corpus_text = load_pdf_docx(master["pdf_dir"])
    chunks = chunk_text(corpus_text, exp["chunking"], embedder)
    print("Chunks:", len(chunks))

    chunk_embs = embedder.encode(chunks, normalize_embeddings=True, batch_size=4)

    q_embs = embedder.encode(df["question"].tolist(), normalize_embeddings=True, batch_size=4)
    gt_embs = embedder.encode(df["context"].tolist(), normalize_embeddings=True, batch_size=4)

    dim = chunk_embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_embs)

    top_k = master.get("top_k", 5)
    rows = []

    full_sim_matrix = np.dot(chunk_embs, q_embs.T)

    for i, row in df.iterrows():
        q_emb = q_embs[i]
        gt_emb = gt_embs[i]
        gt_text = row["context"]

        # ------------------------------------------------
        # ROBUST GROUNDTRUTH CHUNK MATCHING
        # ------------------------------------------------
        gt_chunk = map_groundtruth_chunk(gt_text, chunks, chunk_embs, gt_emb)

        # ------------------------------------------------
        # RETRIEVAL
        # ------------------------------------------------
        scores, idx = index.search(q_emb.reshape(1, -1), top_k)
        retrieved = idx[0]
        # ------------------------------------------------
        # TEXT HIT K (exact + overlap + semantic similarities)
        # ------------------------------------------------
        # Pair: (chunk_id, chunk_text) untuk setiap retrieved
        retrieved_pairs = [(rid, chunks[rid]) for rid in retrieved]

        text_hit, exact_score, overlap_score, semantic_score = text_hit_match(
            gt_text=gt_text,
            retrieved_chunks=retrieved_pairs,
            chunk_embs=chunk_embs,
            gt_emb=gt_emb,
            overlap_threshold=0.25,
            semantic_threshold=0.86
        )



        # ------------------------------------------------
        # RANKING
        # ------------------------------------------------
        sims_q = np.dot(chunk_embs, q_emb)
        rank_order = np.argsort(sims_q)[::-1]
        rank = int(np.where(rank_order == gt_chunk)[0][0]) + 1
        rr = 1 / rank

        # ------------------------------------------------
        # CLUSTER-BASED HIT (CHUNK ±1)
        # ------------------------------------------------
        cluster = cluster_neighbors(gt_chunk, len(chunks), radius=1)
        cluster_hit = int(any([rid in cluster for rid in retrieved]))

        # ------------------------------------------------
        # nDCG@K
        # ------------------------------------------------
        ndcg = ndcg_at_k(retrieved, gt_chunk, top_k)

        # ------------------------------------------------
        # SIMILARITY STATS
        # ------------------------------------------------
        topk_embs = chunk_embs[retrieved]
        topk_sims = np.dot(topk_embs, q_emb)

        rows.append({
            "id": row["id"],
            "question": row["question"],
            "gt_chunk_id": gt_chunk + 1,
            "retrieved_ids": [r + 1 for r in retrieved],
            "hit_at_k": int(gt_chunk in retrieved),
            "cluster_hit@k": cluster_hit,
            "text_hit@k": text_hit,

            # New detailed scores
            "exact_score": exact_score,
            "overlap_score": overlap_score,
            "semantic_score": semantic_score,

            "rank": rank,
            "rr": rr,
            "ndcg@k": ndcg,
            "topk_sim_max": float(np.max(topk_sims)),
            "topk_sim_avg": float(np.mean(topk_sims)),
        })


    df_res = pd.DataFrame(rows)
    df_res["alignment_status"] = df_res.apply(detect_misalignment, axis=1)
    misaligned_mask = df_res["alignment_status"].isin(["SHIFTED", "SHIFTED_NEAR"])
    misalignment_rate = misaligned_mask.mean()

    summary = {
        "MRR": df_res["rr"].mean(),
        "Recall@K": df_res["hit_at_k"].mean(),
        "ClusterRecall@K": df_res["cluster_hit@k"].mean(),
        "RankAvg": df_res["rank"].mean(),
        "nDCG@K": df_res["ndcg@k"].mean(),
        "TopKSimMaxAvg": df_res["topk_sim_max"].mean(),
        "TopKSimAvg": df_res["topk_sim_avg"].mean(),
        "TextHit@K": df_res["text_hit@k"].mean(),
        "MisalignmentRate": misalignment_rate,
        "TrueRetrievalSuccess": df_res["text_hit@k"].mean(),
        "ExactScoreAvg": df_res["exact_score"].mean(),
        "OverlapScoreAvg": df_res["overlap_score"].mean(),
        "SemanticScoreAvg": df_res["semantic_score"].mean(),
        }



    # ============================================================
    # BLOOM LEVEL ANALYSIS
    # ============================================================
    if "BloomLevel" in df.columns:
        df_res["BloomLevel"] = df["BloomLevel"]

        bloom_summary = (
            df_res.groupby("BloomLevel")
            .agg({
                "rr": "mean",
                "hit_at_k": "mean",
                "text_hit@k": "mean",
                "cluster_hit@k": "mean",
                "rank": "mean",
                "ndcg@k": "mean",
                "topk_sim_max": "mean",
                "topk_sim_avg": "mean",
                "exact_score": "mean",
                "overlap_score": "mean",
                "semantic_score": "mean",
            })
            .rename(columns={
                "rr": "MRR",
                "hit_at_k": "Recall@K",
                "text_hit@k": "TextHit@K",
                "cluster_hit@k": "ClusterRecall@K",
                "rank": "RankAvg",
                "ndcg@k": "nDCG@K",
                "topk_sim_max": "TopKSimMaxAvg",
                "topk_sim_avg": "TopKSimAvg",
                "exact_score": "ExactScoreAvg",
                "overlap_score": "OverlapScoreAvg",
                "semantic_score": "SemanticScoreAvg",
            })
            .reset_index()
        )



    else:
        bloom_summary = pd.DataFrame()  # fallback if missing

    #return df_res, chunks, summary, full_sim_matrix
    return df_res, chunks, summary, bloom_summary, full_sim_matrix



# ================================================
# VISUALIZATION
# ================================================
def plot_heatmap(sim_matrix, out_dir, exp_id):
    plt.figure(figsize=(10, 6))
    sns.heatmap(sim_matrix, cmap="viridis")
    plt.title(f"Similarity Heatmap: {exp_id}")
    path = f"{out_dir}/heatmap.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_scatter_rank_similarity(df_res, out_dir):
    plt.figure(figsize=(6, 5))
    plt.scatter(df_res["rank"], df_res["topk_sim_max"], alpha=0.7)
    plt.xlabel("Rank")
    plt.ylabel("TopK Similarity Max")
    plt.title("Rank vs Similarity")
    plt.grid(True)

    path = f"{out_dir}/scatter_rank_similarity.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


# ================================================
# XLSX EXPORT
# ================================================
def save_excel(exp_id, out_dir, df_res, chunks, summary, bloom_summary, heatmap_path, scatter_path):

    excel_path = f"{out_dir}/{exp_id}.xlsx"

    df_summary = pd.DataFrame({
        "metric": summary.keys(),
        "value": summary.values()
    })

    df_chunks = pd.DataFrame({
        "chunk_id": range(1, len(chunks)+1),
        "chunk_text": chunks,
        "tokens": [len(c.split()) for c in chunks],
        "chars": [len(c) for c in chunks]
    })

    with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
        df_res.to_excel(w, index=False, sheet_name="retrieval")
        ws = w.sheets["retrieval"]
        format_decimal(ws)

        df_summary.to_excel(w, index=False, sheet_name="summary")
        ws2 = w.sheets["summary"]
        format_decimal(ws2)

        # ====================================================
        # Bloom Level Analysis Sheet
        # ====================================================
        if bloom_summary is not None and len(bloom_summary) > 0:
            bloom_summary.to_excel(w, index=False, sheet_name="bloom_analysis")
            ws3 = w.sheets["bloom_analysis"]
            format_decimal(ws3)

        df_chunks.to_excel(w, index=False, sheet_name="chunks")
        ws4 = w.sheets["chunks"]
        format_decimal(ws4)

        wb = w.book

        ws_h = wb.create_sheet("heatmap")
        img_h = openpyxl.drawing.image.Image(heatmap_path)
        img_h.anchor = "A1"
        ws_h.add_image(img_h)

        ws_s = wb.create_sheet("rank_similarity")
        img_s = openpyxl.drawing.image.Image(scatter_path)
        img_s.anchor = "A1"
        ws_s.add_image(img_s)

    print("✔ Excel saved:", excel_path)



# ================================================
# MAIN
# ================================================
def main():
    cfg = load_yaml("configs/retrieval.yaml")
    master = cfg["master"]
    experiments = cfg["experiments"]
    all_bloom = []

    df = pd.read_excel(master["dataset"])
    ensure_dir("outputsRetrieval")

    global_summary = {}

    for exp in experiments:
        if not exp.get("active", False):
            continue

        exp_id = exp["id"]
        out_dir = auto_out(exp_id)

        df_res, chunks, summary, bloom_summary, sim_matrix = run_retrieval(exp, master, df)
        global_summary[exp_id] = summary
        bloom_summary["experiment_id"] = exp_id
        all_bloom.append(bloom_summary)
        heatmap_path = plot_heatmap(sim_matrix, out_dir, exp_id)
        scatter_path = plot_scatter_rank_similarity(df_res, out_dir)
    

        save_excel(exp_id, out_dir, df_res, chunks, summary, bloom_summary, heatmap_path, scatter_path)



        # SAVE GLOBAL COMPARISON
    summary_dir = f"outputsRetrieval/{stamp}"
    ensure_dir(summary_dir)
    df_bloom_global = pd.concat(all_bloom, ignore_index=True)

    df_bloom_global.to_excel(
        f"{summary_dir}/summary_bloom_all_experiments_{stamp}.xlsx",
        index=False
    )
    df_global = pd.DataFrame.from_dict(global_summary, orient="index")
    df_global.reset_index(inplace=True)
    df_global.rename(columns={"index": "experiment_id"}, inplace=True)



    df_global.to_excel(
        f"{summary_dir}/summary_all_experiments_{stamp}.xlsx",
        index=False
    )




    print("\n=== GLOBAL SUMMARY SAVED ===")

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
