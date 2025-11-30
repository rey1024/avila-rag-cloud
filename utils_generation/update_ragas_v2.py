import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from openpyxl import load_workbook

from utils_generation.eval_ragas import run_ragas_evaluation
from utils_generation.io_utils import save_excel_report


logger = logging.getLogger(__name__)


# ===============================================================
#  Main function: FULL regenerate after updating missing rows
# ===============================================================
def update_ragas_missing_v2(
    original_excel_path,
    out_dir,
    exp_id,
    timestamp,
    openai_key,
    # all original pipeline inputs required for full regenerate
    df_source,
    questions, predictions, references,
    contexts_used,
    chunks,
    chunk_ids,
    retrieval_hits, semantic_scores,
    embedder,
    gt_chunk_ids, gt_chunk_retrieved, gt_chunk_rank, gt_chunk_rr,
    gt_topk_sims,
    faiss_queries,
    chunk_embs
):
    """
    Update skor RAGAS yang kosong atau 0, lalu regenerate seluruh Excel
    menggunakan pipeline asli save_excel_report().
    """

    logger.info("======================================================")
    logger.info("🔄 Memulai update_ragas_missing_v2()")
    logger.info(f"📘 File original: {original_excel_path}")
    logger.info("======================================================")

    df_results = pd.read_excel(original_excel_path, sheet_name="results")

    metric_cols = [
        "context_precision",
        "context_recall",
        "answer_relevancy",
        "faithfulness",
        "answer_correctness"
    ]

    # ---------------------------------------------
    # 1. Cari baris yang harus diupdate
    # ---------------------------------------------
    missing_mask = df_results[metric_cols].isnull().any(axis=1)
    zero_mask = (df_results[metric_cols] == 0).any(axis=1)

    recalc_mask = missing_mask | zero_mask
    recalc_indices = df_results[recalc_mask].index.tolist()

    logger.info(f"🔍 Total rows perlu re-evaluasi: {len(recalc_indices)}")

    # Jika tidak ada yang perlu diupdate → tetap regenerate grafik
    if len(recalc_indices) == 0:
        logger.info("✔ Tidak ada row perlu evaluasi ulang. Hanya regenerate excel baru.")
        updated_ragas_item = None  # signal untuk memakai yang lama
    else:
        logger.info("✔ Menjalankan ulang RAGAS untuk row kosong / bernilai 0...")

        q_list = []
        pred_list = []
        ctx_list = []
        gt_list = []
        row_map = []

        for idx in recalc_indices:
            row = df_results.iloc[idx]

            reason = []
            if missing_mask.iloc[idx]:
                reason.append("missing")
            if zero_mask.iloc[idx]:
                reason.append("zero")

            logger.info(f"• Row {idx} → RE-EVAL (alasan: {', '.join(reason)})")
            logger.info(f"  Question: {row['question'][:80]}...")
            
            q_list.append(row["question"])
            pred_list.append(row["llm_answer"])
            ctx_list.append(row["contexts_retrieval"].split("\n"))
            gt_list.append(row["answer"])
            row_map.append(idx)

        # jalankan RAGAS
        ragas_item_new, _ = run_ragas_evaluation(
            q_list,
            pred_list,
            ctx_list,
            gt_list,
            openai_key
        )

        # update hanya row yang dipilih
        for local_i, excel_idx in enumerate(row_map):
            item = ragas_item_new[local_i]

            df_results.loc[excel_idx, "context_precision"] = item["context_precision"]
            df_results.loc[excel_idx, "context_recall"] = item["context_recall"]
            df_results.loc[excel_idx, "answer_relevancy"] = item["answer_relevancy"]
            df_results.loc[excel_idx, "faithfulness"] = item["faithfulness"]
            df_results.loc[excel_idx, "answer_correctness"] = item["answer_correctness"]

            logger.info(
                f"✔ Row updated {excel_idx}: "
                f"CP={item['context_precision']}, "
                f"CR={item['context_recall']}, "
                f"AR={item['answer_relevancy']}, "
                f"F={item['faithfulness']}, "
                f"AC={item['answer_correctness']}"
            )

        # simpan ulang "ragas_item" komprehensif untuk regenerate Excel
        updated_ragas_item = []
        for i in range(len(df_results)):
            updated_ragas_item.append({
                "context_precision": df_results.loc[i, "context_precision"],
                "context_recall": df_results.loc[i, "context_recall"],
                "answer_relevancy": df_results.loc[i, "answer_relevancy"],
                "faithfulness": df_results.loc[i, "faithfulness"],
                "answer_correctness": df_results.loc[i, "answer_correctness"],
            })

    # ---------------------------------------------
    # 2. Hitung ulang RAGAS Aggregate
    # ---------------------------------------------
    ragas_agg_updated = {
        "context_precision": df_results["context_precision"].mean(),
        "context_recall": df_results["context_recall"].mean(),
        "answer_relevancy": df_results["answer_relevancy"].mean(),
        "faithfulness": df_results["faithfulness"].mean(),
        "answer_correctness": df_results["answer_correctness"].mean(),
    }

    logger.info("📊 Aggregate baru:")
    for k, v in ragas_agg_updated.items():
        logger.info(f"  {k:20s}: {v}")

    # ---------------------------------------------
    # 3. Generate file baru lengkap (grafik)
    # ---------------------------------------------
    timestamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_excel_path = os.path.join(out_dir, f"{exp_id}_v-update_{timestamp2}.xlsx")
    log_path = os.path.join(out_dir, f"{exp_id}_v-update_{timestamp2}.log")

    # tambahkan file logger
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("======================================================")
    logger.info(f"📄 Menyimpan regenerate Excel → {new_excel_path}")
    logger.info(f"📝 Log → {log_path}")
    logger.info("======================================================")

    # panggil save_excel_report untuk rebuild grafik
    save_excel_report(
        exp_id,
        out_dir,
        timestamp2,
        df_source,
        questions,
        predictions,
        references,
        contexts_used,
        updated_ragas_item,
        ragas_agg_updated,
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
        faiss_queries,
        chunk_embs
    )

    logger.info("✔ update_ragas_missing_v2 selesai.")
    return new_excel_path, log_path
