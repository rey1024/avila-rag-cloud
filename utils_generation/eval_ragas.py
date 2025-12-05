# utils/eval_ragas.py

import numpy as np
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness, AnswerRelevancy,
    AnswerCorrectness, ContextPrecision, ContextRecall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

def fmt(v, d=3):
    try:
        return round(float(v), d)
    except:
        return 0.0


def run_ragas_evaluation(questions, predictions, contexts_used, references, openai_key):
    """
    Evaluasi RAGAS dengan retry otomatis jika ditemukan nilai NaN atau 0.
    Maksimal retry = 3.
    """

    logger.info("\n==============================")
    logger.info("📊  RAGAS Evaluation Started - with retry mechanism")
    logger.info("==============================\n")

    # ---------------------------------
    # SETUP LLM DAN EMBEDDER
    # ---------------------------------
    ragas_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        openai_api_key=openai_key
    )
    ragas_embed = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_key
    )
    ragas_embed.client.batch_size = 128

    # ---------------------------------
    # DATASET RAGAS
    # ---------------------------------
    ds = Dataset.from_dict({
        "question": questions,
        "answer": predictions,
        "contexts": contexts_used,
        "ground_truth": references
    })

    metrics = [
        ContextPrecision(),
        ContextRecall(),
        AnswerRelevancy(),
        Faithfulness(),
        AnswerCorrectness()
    ]

    # ==============================================================
    # Retry RAGAS evaluation
    # ==============================================================

    def evaluate_once():
        """1x evaluasi RAGAS"""
        return evaluate(ds, metrics, ragas_llm, ragas_embed)

    max_retry = 3
    delay = 3

    for attempt in range(1, max_retry + 1):
        logger.info(f"\n🔄 RAGAS attempt {attempt}/{max_retry} ...")
        scores = evaluate_once()

        # cek apakah ada nilai error
        failed = False
        for key in ["context_precision", "context_recall",
                    "answer_relevancy", "faithfulness", "answer_correctness"]:

            arr = scores[key]
            if any(v is None for v in arr):
                failed = True
            # if any((v is not None and float(v) <= 0.3) for v in arr):
            #     failed = True

        if not failed:
            logger.info("✔ RAGAS evaluation successful")
            break

        logger.warning(f"⚠ RAGAS result contains NaN/0 → retrying after {delay}s...")
        time.sleep(delay)

    else:
        logger.error("❌ RAGAS failed after maximum retries. Using last computed scores.")

    # ========== lanjut proses normal ============
    total = len(questions)
    ragas_item = []

    logger.info("\n🔍 Detail skor per item:")

    for i in range(total):

        item_scores = {
            "context_precision": fmt(scores["context_precision"][i]),
            "context_recall": fmt(scores["context_recall"][i]),
            "answer_relevancy": fmt(scores["answer_relevancy"][i]),
            "faithfulness": fmt(scores["faithfulness"][i]),
            "answer_correctness": fmt(scores["answer_correctness"][i]),
        }
        ragas_item.append(item_scores)

        logger.info("\n==============================")
        logger.info(f"Item {i+1}/{total}")
        logger.info("==============================")
        logger.info("--- Skor ---")
        for k, v in item_scores.items():
            logger.info(f"  {k:20s}: {v}")

    # Aggregate
    ragas_agg = {
        "context_precision": fmt(np.mean(scores["context_precision"])),
        "context_recall": fmt(np.mean(scores["context_recall"])),
        "answer_relevancy": fmt(np.mean(scores["answer_relevancy"])),
        "faithfulness": fmt(np.mean(scores["faithfulness"])),
        "answer_correctness": fmt(np.mean(scores["answer_correctness"])),
    }

    logger.info("\n==============================")
    logger.info("📈  RAGAS Aggregate Scores")
    logger.info("==============================\n")

    for k, v in ragas_agg.items():
        logger.info(f"  {k:20s}: {v}")

    logger.info("\n✔ Evaluasi RAGAS selesai.\n")

    return ragas_item, ragas_agg

def run_ragas_evaluation(*args, **kwargs): 
    return asyncio.run(run_ragas_evaluation_async(*args, **kwargs))