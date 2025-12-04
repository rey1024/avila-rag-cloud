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
from ragas.metrics import NonLLMContextRecall   # opsional
from ragas.metrics import LLMContextRecall       # sesuai contoh

from langchain_openai import ChatOpenAI
import logging
import threading
import time


logger = logging.getLogger(__name__)


# Heartbeat
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


# ============================================================
# 1. Evaluasi satu sample (mengikuti EXACT contoh Anda)
# ============================================================
async def run_single_sample(ds, i, llm_ragas, embeddings, evaluator_llm):

    q  = ds["question"][i]
    r  = ds["answer"][i]
    gt = ds["ground_truth"][i]
    ctx = ds["contexts"][i]

    # pastikan konteks list[string]
    if isinstance(ctx, list):
        if len(ctx) > 0 and isinstance(ctx[0], dict):
            ctx = [c["text"] for c in ctx]
    else:
        ctx = [str(ctx)]

    # SingleTurnSample mengikuti spesifikasi contoh
    sample = SingleTurnSample(
        user_input=q,
        response=r,
        reference=gt,
        retrieved_contexts=ctx
    )

    # METRIC sesuai contoh Anda
    CP = await ContextPrecision(llm=llm_ragas).ascore(
        user_input=q,
        reference=gt,
        retrieved_contexts=ctx
    )

    AR = await AnswerRelevancy(llm=llm_ragas, embeddings=embeddings).ascore(
        user_input=q,
        response=r
    )

    F = await Faithfulness(llm=llm_ragas).ascore(
        user_input=q,
        response=r,
        retrieved_contexts=ctx
    )

    #CR = await LLMContextRecall(llm=evaluator_llm).single_turn_ascore(sample)
    CR = await NonLLMContextRecall().single_turn_ascore(sample)
    

    return {
        "context_precision": float(CP.value),
        "context_recall": float(CR),
        "answer_relevancy": float(AR.value),
        "faithfulness": float(F.value),
    }


# ============================================================
# 2. Evaluasi seluruh dataset (async)
# ============================================================
async def run_ragas_evaluation_async(questions, predictions, contexts_used, references, openai_key):

    start_heartbeat()

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

    # ======================================================
    # ✔ SEMAPHORE: batas maksimal 5 task paralel
    # ======================================================
    sema = asyncio.Semaphore(5)

    logger.info(f"[RAGAS] Parallel evaluation dimulai. Limit concurrency = 5")

    tasks = []
    for i in range(total):

        # flatten context
        raw_ctx = ds["contexts"][i]
        if isinstance(raw_ctx, list):
            if len(raw_ctx) > 0 and isinstance(raw_ctx[0], list):
                raw_ctx = [c for sub in raw_ctx for c in sub]
        else:
            raw_ctx = [str(raw_ctx)]
        ds["contexts"][i] = raw_ctx

        # ======================================================
        # ✔ SETIAP TASK DIBUNGKUS DENGAN semaphore
        # ======================================================
        async def task_wrapper(i=i):
            async with sema:
                logger.info(f"[RAGAS] ▶ Memulai task {i+1}/{total} (slot semaphore aktif)")
                return await run_single_sample(ds, i, llm_ragas, embeddings, evaluator_llm)

        tasks.append(task_wrapper())

    # jalankan seluruh tasks paralel
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # post processing
    results = []
    for i, item in enumerate(raw_results):
        if isinstance(item, Exception):
            logger.error(f"[RAGAS] ❌ Error item {i+1}: {item}")
            results.append({
                "context_precision": 0,
                "context_recall": 0,
                "answer_relevancy": 0,
                "faithfulness": 0,
            })
        else:
            logger.info(f"[RAGAS] ✔ Item {i+1} selesai parallel")
            results.append(item)

    # aggregate
    ragas_agg = {
        k: fmt(np.mean([r[k] for r in results]))
        for k in results[0].keys()
    }

    logger.info("\n==============================")
    logger.info("📈  RAGAS Aggregate Scores")
    logger.info("==============================\n")
    for k, v in ragas_agg.items():
        logger.info(f"{k:20s}: {v}")

    return results, ragas_agg

# wrapper synchronous
def run_ragas_evaluation(questions, predictions, contexts_used, references, openai_key):
    return asyncio.run(
        run_ragas_evaluation_async(
            questions,
            predictions,
            contexts_used,
            references,
            openai_key
        )
    )
