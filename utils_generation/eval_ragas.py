# utils/eval_ragas.py

import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness, AnswerRelevancy,
    AnswerCorrectness, ContextPrecision, ContextRecall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import logging
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fmt(v, d=3):
    try:
        return round(float(v), d)
    except:
        return 0.0


def run_ragas_evaluation(questions, predictions, contexts_used, references, openai_key):
    """
    Menjalankan evaluasi RAGAS:
      - context precision
      - context recall
      - answer relevancy
      - faithfulness
      - answer correctness

    Return:
      ragas_item (list per item)
      ragas_agg (aggregate)
    """

    logger.info("\n==============================")
    logger.info("📊  RAGAS Evaluation Started -v1")
    logger.info("==============================\n")

    # ---------------------------------
    # SETUP LLM DAN EMBEDDER
    # ---------------------------------
    ragas_llm = ChatOpenAI(
        model="gpt-4o-mini",
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
    import logging
    logging.getLogger("ragas").setLevel(logging.DEBUG)
    logging.getLogger("ragas.metrics").setLevel(logging.DEBUG)
    logging.getLogger("ragas.llms").setLevel(logging.DEBUG)
    logging.getLogger("ragas.prompt").setLevel(logging.DEBUG)



    scores = evaluate(ds, metrics, ragas_llm, ragas_embed)

    total = len(questions)
    ragas_item = []


    logger.info("🔍 Detail skor per item:\n")

    for i in range(total):

        q = ds[i]["question"]
        a = ds[i]["answer"]
        ctx = ds[i]["contexts"]
        gt = ds[i]["ground_truth"]



        item_scores = {
            "context_precision": fmt(scores["context_precision"][i]),
            "context_recall": fmt(scores["context_recall"][i]),
            "answer_relevancy": fmt(scores["answer_relevancy"][i]),
            "faithfulness": fmt(scores["faithfulness"][i]),
            "answer_correctness": fmt(scores["answer_correctness"][i]),
        }
        ragas_item.append(item_scores)

        logger.info(f"\n==============================")
        logger.info(f"Item {i+1}/{total}")
        logger.info("==============================")
        logger.info(f"Question       : {q}")
        logger.info(f"Answer         : {a}")
        logger.info(f"Ground Truth   : {gt}")
        logger.info(f"Contexts       : {ctx}")
        logger.info("--- Skor ---")
        logger.info(f"  context_precision : {item_scores['context_precision']}")
        logger.info(f"  context_recall    : {item_scores['context_recall']}")
        logger.info(f"  answer_relevancy  : {item_scores['answer_relevancy']}")
        logger.info(f"  faithfulness      : {item_scores['faithfulness']}")
        logger.info(f"  answer_correctness: {item_scores['answer_correctness']}")
        logger.info("")

    # ---------------------------------
    # AGGREGATE
    # ---------------------------------
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
        print(f"  {k:20s}: {v}")

    logger.info("\n✔ Evaluasi RAGAS selesai.\n")

    return ragas_item, ragas_agg
