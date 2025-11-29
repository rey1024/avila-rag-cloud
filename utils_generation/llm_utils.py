import os
import logging
import json
import yaml
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import asyncio
import aiohttp
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
from utils_generation.io_utils import *

from dotenv import load_dotenv
load_dotenv()
print("✔ Env loaded from .env")




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
import logging
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
async def process_batch(items, endpoint, model, batch_size=6):
    async with aiohttp.ClientSession() as session:
        tasks = []

        for item in items:
            #logger.info(f"[OLLAMA] Memproses prompt: {item['prompt'][:50]}...")
            tasks.append(async_generate_ollama(
                session,
                item["prompt"],
                endpoint,
                model
            ))

        # batched parallel execution
        results = []
        for i in range(0, len(tasks), batch_size):
            logger.info(f"[OLLAMA] Memproses batch {i//batch_size + 1}...")
            sub = tasks[i:i+batch_size]
            out = await asyncio.gather(*sub)
            results.extend(out)

        return results
    
def generate_with_ollama(prompt, endpoint, model):
    url = endpoint.rstrip("/") + "/api/generate"
    r = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    }, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "")


