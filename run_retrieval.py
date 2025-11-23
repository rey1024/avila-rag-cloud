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


def semantic_chunk(text, threshold, embedder):
    sents = sent_tokenize(text)

    sent_embs = embedder.encode(sents, normalize_embeddings=True)

    chunks = []
    current_chunk = [sents[0]]

    for i in range(1, len(sents)):
        sim = np.dot(sent_embs[i], sent_embs[i - 1])
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        current_chunk.append(sents[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_text(text, cfg, embedder=None):
    if cfg["type"] == "fixed":
        return fixed_chunk(text, cfg["size"], cfg["overlap"])

    if cfg["type"] == "sentence":
        return fixed_chunk(text, cfg["window"], cfg["overlap"])

    if cfg["type"] == "semantic":
        return semantic_chunk(text, cfg["threshold"], embedder)

    raise ValueError("Unknown chunking method")


# ================================================
# RETRIEVAL PIPELINE
# ================================================
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

    chunk_embs = embedder.encode(chunks, normalize_embeddings=True)

    dim = chunk_embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_embs)

    q_embs = embedder.encode(df["question"].tolist(), normalize_embeddings=True)
    gt_embs = embedder.encode(df["context"].tolist(), normalize_embeddings=True)

    top_k = master.get("top_k", 5)
    rows = []

    full_sim_matrix = np.dot(chunk_embs, gt_embs.T)

    for i, row in df.iterrows():
        q_emb = q_embs[i]
        gt_emb = gt_embs[i]

        scores, idx = index.search(q_emb.reshape(1, -1), top_k)
        top_ids = idx[0]

        sims_all = np.dot(chunk_embs, gt_emb)
        gt_chunk_id = int(np.argmax(sims_all)) + 1

        rank_order = np.argsort(sims_all)[::-1]
        rank = int(np.where(rank_order == (gt_chunk_id - 1))[0][0]) + 1
        rr = 1 / rank

        topk_embs = chunk_embs[top_ids]
        topk_sims = np.dot(topk_embs, gt_emb)

        rows.append({
            "id": row["id"],
            "question": row["question"],
            "gt_chunk_id": gt_chunk_id,
            "retrieved_ids": [i + 1 for i in top_ids],
            "hit_at_k": int(gt_chunk_id in (top_ids + 1)),
            "rank": rank,
            "rr": rr,
            "topk_sim_max": float(np.max(topk_sims)),
            "topk_sim_avg": float(np.mean(topk_sims)),
            "gt_sim_to_all": sims_all.tolist()
        })

    df_res = pd.DataFrame(rows)

    summary = {
        "MRR": df_res["rr"].mean(),
        "Recall@K": df_res["hit_at_k"].mean(),
        "RankAvg": df_res["rank"].mean(),
        "TopKSimMaxAvg": df_res["topk_sim_max"].mean(),
        "TopKSimAvg": df_res["topk_sim_avg"].mean(),
    }

    return df_res, chunks, summary, full_sim_matrix


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
def save_excel(exp_id, out_dir, df_res, chunks, summary, heatmap_path, scatter_path):
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
        df_summary.to_excel(w, index=False, sheet_name="summary")
        df_chunks.to_excel(w, index=False, sheet_name="chunks")

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

    df = pd.read_excel(master["dataset"])
    ensure_dir("outputsRetrieval")

    global_summary = {}

    for exp in experiments:
        if not exp.get("active", False):
            continue

        exp_id = exp["id"]
        out_dir = auto_out(exp_id)

        df_res, chunks, summary, sim_matrix = run_retrieval(exp, master, df)
        global_summary[exp_id] = summary

        heatmap_path = plot_heatmap(sim_matrix, out_dir, exp_id)
        scatter_path = plot_scatter_rank_similarity(df_res, out_dir)

        save_excel(exp_id, out_dir, df_res, chunks, summary, heatmap_path, scatter_path)


        # SAVE GLOBAL COMPARISON
    df_global = pd.DataFrame.from_dict(global_summary, orient="index")
    df_global.reset_index(inplace=True)
    df_global.rename(columns={"index": "experiment_id"}, inplace=True)

    df_global.to_excel("outputsRetrieval/summary_all_experiments_{datetime}.xlsx", index=False)

    print("\n=== GLOBAL SUMMARY SAVED ===")

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
