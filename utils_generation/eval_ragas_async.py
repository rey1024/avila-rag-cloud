import asyncio
import numpy as np
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from ragas.metrics.collections import (
    ContextPrecision,
    AnswerRelevancy,
    Faithfulness
)

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall
from ragas.metrics import LLMContextRecall

from langchain_openai import ChatOpenAI
import logging
import threading
import time


logger = logging.getLogger(__name__)


# =========================================================
# Heartbeat
# =========================================================
def start_heartbeat():
    def hb():
        while True:
            logger.info("[HEARTBEAT] RAGAS masih berjalan...")
            time.sleep(60)
    threading.Thread(target=hb, daemon=True).start()


def fmt(v, d=3):
    try:
        return round(float(v), d)
    except:
        return 0.0


# =========================================================
# Evaluasi satu item
# =========================================================
async def run_single_sample(ds, i, llm_ragas, embeddings, evaluator_llm):

    q  = ds["question"][i]
    r  = ds["answer"][i]
    gt = ds["ground_truth"][i]
    ctx = ds["contexts"][i]

    

    sample = SingleTurnSample(
        user_input=q,
        response=r,
        reference=gt,
        retrieved_contexts=ctx
    )

    try:
        CP = await ContextPrecision(llm=llm_ragas).ascore(
            user_input=q, reference=gt, retrieved_contexts=ctx
        )

        AR = await AnswerRelevancy(llm=llm_ragas, embeddings=embeddings).ascore(
            user_input=q, response=r
        )

        F = await Faithfulness(llm=llm_ragas).ascore(
            user_input=q, response=r, retrieved_contexts=ctx
        )

        CR = await LLMContextRecall(llm=evaluator_llm).single_turn_ascore(sample)
        #CR=10
    except Exception as e:
        raise RuntimeError(f"Metric error: {e}")

    return {
        "context_precision": float(CP.value),
        "context_recall": float(CR),
        "answer_relevancy": float(AR.value),
        "faithfulness": float(F.value),
    }


# =========================================================
# Evaluasi SEQUENTIAL
# =========================================================
async def run_ragas_evaluation_async(questions, predictions, contexts_used, references, openai_key):

    start_heartbeat()

    logger.info("\n==============================")
    logger.info("📊  RAGAS Evaluation Started (Sequential Mode)")
    logger.info("==============================\n")

    client = AsyncOpenAI(api_key=openai_key)

    llm_ragas = llm_factory("gpt-4.1-mini", client=client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    evaluator_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        openai_api_key=openai_key
    )

    ds = {
        "question": questions,
        "answer": predictions,
        "ground_truth": references,
        "contexts": contexts_used
    }

    total = len(questions)
    results = []

    for i in range(total):

        q = ds["question"][i]
        r = ds["answer"][i]
        ctx = ds["contexts"][i]

        # fallback prevent error
        if not ctx or len(ctx) == 0:
            logger.warning(f"[RAGAS] Context kosong item {i+1}, menggunakan fallback")
            ctx = ["No context retrieved"]
            ds["contexts"][i] = ctx

        logger.info("----------------------------------------------")
        logger.info(f"[RAGAS] ▶ Evaluasi item {i+1}/{total}")
        logger.info("----------------------------------------------")
        logger.info(f"  • Pertanyaan     : {q[:80]}...")
        logger.info(f"  • Jawaban LLM    : {r[:80]}...")
        logger.info(f"  • Jumlah konteks : {len(ctx)}")

                # ===========================================================
        # Tambahkan log konteks (preview)
        # ===========================================================
        if isinstance(ctx, list):
            if len(ctx) == 1:
                preview = ctx[0][:150].replace("\n", " ")
                logger.info(f"  • Context Sample : {preview}...")
            else:
                p1 = ctx[0][:150].replace("\n", " ")
                logger.info(f"  • Context 1      : {p1}...")

                if len(ctx) > 1:
                    p2 = ctx[1][:150].replace("\n", " ")
                    logger.info(f"  • Context 2      : {p2}...")
        else:
            preview = str(ctx)[:150].replace("\n", " ")
            logger.info(f"  • Context Sample : {preview}...")

        # ==========================================================

        t0 = time.time()

        try:
            metric = await run_single_sample(ds, i, llm_ragas, embeddings, evaluator_llm)
        except Exception as e:
            logger.error(f"[RAGAS] ❌ Error item {i+1}: {e}")
            metric = {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
            }

        dt = time.time() - t0

        logger.info(f"[RAGAS] ✔ Selesai item {i+1} dalam {dt:.2f} s")
        logger.info(f"        Context Precision : {fmt(metric['context_precision'])}")
        logger.info(f"        Context Recall    : {fmt(metric['context_recall'])}")
        logger.info(f"        Answer Relevancy  : {fmt(metric['answer_relevancy'])}")
        logger.info(f"        Faithfulness      : {fmt(metric['faithfulness'])}")

        results.append(metric)

    # =====================================================
    # AGGREGATE
    # =====================================================
    ragas_agg = {
        k: fmt(np.mean([r[k] for r in results]))
        for k in results[0].keys()
    }

    logger.info("\n==============================")
    logger.info("📈  RAGAS Aggregate Scores")
    logger.info("==============================")
    for k, v in ragas_agg.items():
        logger.info(f"{k:20s}: {v}")

    return results, ragas_agg


# =========================================================
# Wrapper sync
# =========================================================
def run_ragas_evaluation(*args, **kwargs):
    return asyncio.run(run_ragas_evaluation_async(*args, **kwargs))
