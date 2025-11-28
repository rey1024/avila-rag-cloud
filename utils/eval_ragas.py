# utils/eval_ragas.py

import asyncio
import numpy as np
from datasets import Dataset

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from ragas.metrics.collections import (
    ContextPrecision,
    AnswerRelevancy,
    Faithfulness,
    AnswerCorrectness
)

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import LLMContextRecall
from langchain_openai import ChatOpenAI
import logging
logging.basicConfig(level=logging.INFO)

def fmt(v, d=3):
    try:
        return round(float(v), d)
    except:
        return 0.0


async def run_single_sample(ds, i, llm_ragas, embeddings, evaluator_llm):

    q  = ds["question"][i]
    r  = ds["answer"][i]
    gt = ds["ground_truth"][i]
    ctx = ds["contexts"][i]

    # Metrik
    CP = await ContextPrecision(llm=llm_ragas).ascore(
        user_input=q, reference=gt, retrieved_contexts=ctx
    )

    F = await Faithfulness(llm=llm_ragas).ascore(
        user_input=q, response=r, retrieved_contexts=ctx
    )

    AR = await AnswerRelevancy(llm=llm_ragas, embeddings=embeddings).ascore(
        user_input=q, response=r
    )

    AC = await AnswerCorrectness(llm=llm_ragas, embeddings=embeddings).ascore(
        user_input=q, response=r, reference=gt
    )

    sample = SingleTurnSample(
        user_input=q, response=r, reference=gt, retrieved_contexts=ctx
    )
    CR = await LLMContextRecall(llm=evaluator_llm).single_turn_ascore(sample)

    return {
        "context_precision": float(CP.value),
        "context_recall": float(CR),
        "answer_relevancy": float(AR.value),
        "faithfulness": float(F.value),
        "answer_correctness": float(AC.value),
    }


async def run_ragas_evaluation_async(questions, predictions, contexts_used, references, openai_key):

    # Single OpenAI client untuk semua komponen
    client = AsyncOpenAI(api_key=openai_key)

    # Factory sesuai dokumentasi RAGAS terbaru
    llm_ragas = llm_factory(model="gpt-4o-mini", client=client)

    embeddings = embedding_factory(
        provider="openai",
        model="text-embedding-3-large",
        client=client
    )

    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_key
    )

    # Dataset
    ds = Dataset.from_dict({
        "question": questions,
        "answer": predictions,
        "contexts": contexts_used,
        "ground_truth": references
    })

    # Run semua sample
    tasks = [
        run_single_sample(ds, i, llm_ragas, embeddings, evaluator_llm)
        for i in range(len(ds))
    ]

    results = await asyncio.gather(*tasks)

    # Aggregate
    ragas_agg = {
        k: fmt(np.mean([res[k] for res in results]))
        for k in results[0].keys()
    }
    print("\n==============================")
    print("📈  RAGAS Aggregate Scores")
    print("==============================\n")

    for k, v in ragas_agg.items():
        print(f"  {k:20s}: {v}")

    print("\n✔ Evaluasi RAGAS selesai.\n")
    return results, ragas_agg


def run_ragas_evaluation(*args, **kwargs):
    return asyncio.run(run_ragas_evaluation_async(*args, **kwargs))
