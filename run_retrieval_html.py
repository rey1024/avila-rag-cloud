import os
import yaml
import faiss
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import openpyxl
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
from dotenv import load_dotenv
load_dotenv()
import os
import sys
from sentence_transformers import CrossEncoder

# memastikan folder project menjadi root import
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils_common.utils_new import *
import logging
OUTPUT_DIR=f"results/retrieval/ret_{stamp}"
setup_html_logging(OUTPUT_DIR)
logger = logging.getLogger(__name__)
    close_html_logging()
