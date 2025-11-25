"""
CHUNKING PIPELINE FINAL VERSION
===========================================
Mencakup 4 jenis chunking:

1. Paragraph Chunking
2. Fixed Token Chunking (400 tokens)
3. Semantic Chunking (LangChain-style)
4. Fixed Sentence-aware Chunking (400 tokens, no mid-sentence cut)

Output:
- Satu file XLSX
- 4 sheet sesuai jenis chunk
- wrap text + align top-left
- kolom chunk_text auto width
"""

import re
from pathlib import Path
from docx import Document
import pypdf
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
from datetime import datetime
import openpyxl

nltk.download("punkt")


# ============================================================
# HELPERS FOR CLEANING
# ============================================================

def clean_text(t):
    if not t:
        return ""
    t = t.replace("\xa0", " ")
    t = t.replace("\t", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ============================================================
# LOAD DOCUMENTS (PDF + DOCX)
# ============================================================

def extract_paragraphs_from_pdf(pdf_path):
    reader = pypdf.PdfReader(str(pdf_path))
    paragraphs = []

    for page in reader.pages:
        raw = page.extract_text() or ""
        blocks = re.split(r"\n{2,}", raw)

        for block in blocks:
            block = clean_text(block)
            if len(block) > 30:
                paragraphs.append(block)

    return paragraphs


def extract_paragraphs_from_docx(docx_path):
    doc = Document(str(docx_path))
    paragraphs = []

    for p in doc.paragraphs:
        t = clean_text(p.text)
        if len(t) > 30:
            paragraphs.append(t)

    return paragraphs


def load_documents(folder):
    folder = Path(folder)
    pdfs = sorted(folder.glob("*.pdf"))
    docxs = sorted(folder.glob("*.docx"))

    docs = []
    doc_id = 1

    for p in pdfs:
        paras = extract_paragraphs_from_pdf(p)
        docs.append({
            "doc_id": doc_id,
            "filename": p.name,
            "paragraphs": paras
        })
        print(f"✔ Loaded PDF: {p.name} ({len(paras)} paragraphs)")
        doc_id += 1

    for d in docxs:
        paras = extract_paragraphs_from_docx(d)
        docs.append({
            "doc_id": doc_id,
            "filename": d.name,
            "paragraphs": paras
        })
        print(f"✔ Loaded DOCX: {d.name} ({len(paras)} paragraphs)")
        doc_id += 1

    if not docs:
        raise ValueError("❌ No PDF/DOCX found in folder.")

    return docs


# ============================================================
# 1. PARAGRAPH CHUNKING
# ============================================================

def paragraph_chunking(docs):
    chunks = []
    cid = 1

    for d in docs:
        for para in d["paragraphs"]:
            chunks.append({
                "chunk_id": cid,
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "chunk_text": para,
                "tokens": len(para.split())
            })
            cid += 1

    print(f"📘 Paragraph chunks: {len(chunks)}")
    return pd.DataFrame(chunks)


# ============================================================
# 2. FIXED-SIZE TOKEN CHUNKING (old baseline)
# ============================================================

def fixed_size_chunking(docs, size=400, overlap=80):
    chunks = []
    cid = 1

    for d in docs:
        text = " ".join(d["paragraphs"])
        tokens = text.split()

        step = size - overlap

        for i in range(0, len(tokens), step):
            seg = tokens[i:i + size]
            if not seg:
                continue

            chunk = " ".join(seg)

            chunks.append({
                "chunk_id": cid,
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "chunk_text": chunk,
                "tokens": len(seg)
            })
            cid += 1

    print(f"📗 Fixed-size token chunks: {len(chunks)}")
    return pd.DataFrame(chunks)



# ============================================================
# 3. SEMANTIC CHUNKING (LangChain-style grouping)
# ============================================================

def semantic_chunking(docs, embedder, threshold=0.80):
    chunks = []
    cid = 1

    for d in docs:
        paras = d["paragraphs"]
        para_embs = embedder.encode(paras, normalize_embeddings=True)

        current_group = [paras[0]]

        for i in range(1, len(paras)):
            sim = float(np.dot(para_embs[i], para_embs[i - 1]))

            if sim < threshold:
                chunks.append({
                    "chunk_id": cid,
                    "doc_id": d["doc_id"],
                    "filename": d["filename"],
                    "chunk_text": " ".join(current_group),
                    "tokens": len(" ".join(current_group).split())
                })
                cid += 1
                current_group = []

            current_group.append(paras[i])

        if current_group:
            chunks.append({
                "chunk_id": cid,
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "chunk_text": " ".join(current_group),
                "tokens": len(" ".join(current_group).split())
            })
            cid += 1

    print(f"📙 Semantic chunks: {len(chunks)}")
    return pd.DataFrame(chunks)


# ============================================================
# 4. FIXED SENTENCE-AWARE CHUNKING (400 tokens, no mid-sentence split)
# ============================================================

def fixed_sentence_chunking(text,
                            target_tokens=400,
                            max_tokens=450,
                            overlap_tokens=80):

    sentences = sent_tokenize(text)
    chunks = []

    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())

        if not current_chunk:
            current_chunk.append(sent)
            current_len = sent_len
            continue

        if current_len + sent_len <= target_tokens:
            current_chunk.append(sent)
            current_len += sent_len
            continue

        if current_len + sent_len > max_tokens:
            chunks.append(" ".join(current_chunk))

            overlap_sents = []
            run = 0
            for s in reversed(current_chunk):
                w = len(s.split())
                if run + w > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                run += w

            current_chunk = overlap_sents + [sent]
            current_len = sum(len(s.split()) for s in current_chunk)

        else:
            current_chunk.append(sent)
            current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"📕 Fixed sentence-aware chunks: {len(chunks)}")
    return pd.DataFrame({
        "chunk_id": range(1, len(chunks)+1),
        "chunk_text": chunks,
        "tokens": [len(c.split()) for c in chunks]
    })


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("BAAI/bge-m3")

    print("\n📂 Loading documents from docs/ ...")
    docs = load_documents("docs/")

    print("\n=== 1. PARAGRAPH CHUNKING ===")
    df_paragraph = paragraph_chunking(docs)

    print("\n=== 2. FIXED TOKEN CHUNKING ===")
    df_fixed = fixed_size_chunking(docs)

    print("\n=== 3. SEMANTIC CHUNKING ===")
    df_semantic = semantic_chunking(docs, embedder)

    print("\n=== 4. FIXED SENTENCE-AWARE CHUNKING ===")
    full_text = " ".join(para for d in docs for para in d["paragraphs"])
    df_fixed_sentence = fixed_sentence_chunking(full_text)

    # Save to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"chunks_{timestamp}.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        df_paragraph.to_excel(writer, index=False, sheet_name="paragraph_chunks")
        df_fixed.to_excel(writer, index=False, sheet_name="fixed_chunks")
        df_semantic.to_excel(writer, index=False, sheet_name="semantic_chunks")
        df_fixed_sentence.to_excel(writer, index=False, sheet_name="fixed_sentence_chunks")

        # Formatting
        wb = writer.book
        for sheet in ["paragraph_chunks", "fixed_chunks", "semantic_chunks", "fixed_sentence_chunks"]:
            ws = wb[sheet]

            for row in ws.iter_rows():
                for cell in row:
                    cell.alignment = openpyxl.styles.Alignment(
                        wrap_text=True,
                        vertical="top",
                        horizontal="left"
                    )

            ws.column_dimensions['C'].width = 80  # chunk_text column width

    print("\n🎉 All chunk types saved!")
    print(f"📁 Output File: {out_file}")
