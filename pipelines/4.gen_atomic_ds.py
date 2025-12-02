import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import json
import traceback
from dotenv import load_dotenv
load_dotenv()
import sys
import os
import time 
# memastikan folder project menjadi root import
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils_common.utils import *

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


from openai import APIConnectionError, APITimeoutError
import time

from openai import APIConnectionError, APITimeoutError, RateLimitError
import unicodedata

def normalize_u(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def robust_contains(ctx, atm):
    ctx_n = normalize_u(ctx)
    atm_n = normalize_u(atm)
    return atm_n in ctx_n

def safe_llm_call(prompt, retries=3, timeout=20):
    """
    Wrapper pemanggil LLM dengan retry.
    Mengembalikan string content mentah dari LLM atau None jika benar benar gagal.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content

        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            print(f"[LLM timeout or connection error] Retry {attempt+1}/{retries}: {e}")
            time.sleep(1)

        except Exception as e:
            print(f"[LLM generic error] Retry {attempt+1}/{retries}: {e}")
            time.sleep(1)

    print("❌ LLM failed after max retries")
    return None

# ===========================
# LLM GUIDED ATOMIC EXTRACTION
# ===========================
def validate_contiguous_span(context, span):
    idx = context.find(span)
    if idx == -1:
        return 0

    start = idx
    end = idx + len(span)

    # cek boundary kiri: tidak memotong di tengah kalimat
    if start > 0 and context[start - 1] not in ".!? \n":
        return 0

    # cek boundary kanan: tidak memotong di tengah kalimat
    if end < len(context) and context[end] not in ".!? \n":
        return 0

    return 1

import re

def split_into_sentences(text):
    """
    Pemecah kalimat sederhana berbasis tanda titik seru tanya.
    Tidak sempurna tapi cukup untuk fallback.
    """
    # ganti newline dengan spasi
    text = text.replace("\n", " ")
    # pisah kasar
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # bersihkan
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences

def heuristic_fallback_span(context, answer):
    """
    Fallback tanpa LLM.
    Prinsip:
    - Cari kalimat yang memuat kata kunci dari jawaban.
    - Ambil 1–3 kalimat yang berurutan sebagai span.
    - Hanya mengembalikan UNK jika tidak ada satu pun overlap.
    """

    ctx_sentences = split_into_sentences(context)
    if not ctx_sentences:
        return "UNK", 0.0

    # token sederhana dari jawaban
    answer_tokens = [t.lower() for t in re.findall(r"\w+", answer) if len(t) > 3]
    answer_tokens = list(set(answer_tokens))  # unik

    if not answer_tokens:
        # kalau tidak ada token berarti, ambil kalimat pertama saja
        span = ctx_sentences[0]
        return span, 0.3

    # skor untuk tiap kalimat: berapa banyak token jawaban muncul
    scores = []
    for idx, sent in enumerate(ctx_sentences):
        s_low = sent.lower()
        hit = sum(1 for tok in answer_tokens if tok in s_low)
        scores.append((hit, idx))

    # kalimat dengan hit terbanyak
    scores.sort(reverse=True)  # urut desc by hit
    best_hit, best_idx = scores[0]

    # Jika tidak ada satu pun token yang match, baru kita boleh UNK
    if best_hit == 0:
        print("[INFO] Heuristic found no lexical overlap. Returning UNK.")
        return "UNK", 0.0

    # Ambil jendela kecil: kalimat best_idx dan tetangganya
    start = max(0, best_idx - 1)
    end = min(len(ctx_sentences), best_idx + 2)  # sampai best_idx+1

    span = " ".join(ctx_sentences[start:end]).strip()
    return span, 0.4 + min(0.5, best_hit / max(1, len(answer_tokens)))
import re

def split_into_sentences(text):
    text = text.replace("\n", " ")
    # pecah berbasis ., ?, !
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences
def extract_atomic_span_by_index(context, question, answer):
    sentences = split_into_sentences(context)

    # siapkan daftar bernomor untuk LLM
    numbered = ""
    for i, sent in enumerate(sentences):
        numbered += f"[{i}] {sent}\n"

    prompt = f"""
Kamu adalah extraction engine.

Tugas:
Pilih rentang kalimat berurutan (contiguous) yang mendukung jawaban.
Kamu TIDAK BOLEH menyalin teks atau memparafrase.
Kamu hanya boleh mengembalikan indeks start dan end.

Output HARUS JSON murni:
{{
  "start": 0,
  "end": 0,
  "confidence": 0.0
}}

=== QUESTION ===
{question}

=== ANSWER ===
{answer}

=== CONTEXT SENTENCES ===
{numbered}
"""
    print("Prompt untuk LLM:\n\n", prompt)
    raw = safe_llm_call(prompt)
    if raw is None:
        return "UNK", 0.0

    raw = raw.strip().replace("```json","").replace("```","")

    try:
        js = json.loads(raw)
    except:
        try:
            js = robust_json_parse(raw)
        except:
            return "UNK", 0.0

    start = int(js.get("start", -1))
    end = int(js.get("end", -1))
    conf = float(js.get("confidence", 0))

    # validasi index
    if start < 0 or end < 0 or start >= len(sentences) or end >= len(sentences) or end < start:
        return "UNK", 0.0

    # ✨ INI BAGIAN PENTING — EXACT SPAN DARI PYTHON
    span = " ".join(sentences[start:end+1]).strip()
    print("Extracted span:\n", span)
    print("=====\n\n\n\n")
    return span, conf

def extract_atomic_span(context, question, answer):
    prompt = f"""
Kamu adalah extraction engine.

Tugas:
Temukan blok kalimat berurutan (contiguous) dalam konteks
yang secara langsung mendukung jawaban.

PENTING:
- Kamu TIDAK BOLEH menyalin teks.
- Kamu TIDAK BOLEH memparafrase.
- Kamu hanya boleh memberikan indeks kalimat.
- Output HARUS berupa rentang kalimat dalam konteks.
- Jika informasi pendukung terletak pada kalimat 3 dan 5,
  kamu harus mengembalikan:
      "start_sentence": 3
      "end_sentence": 5
  dan kami yang akan mengambil substring exact.

Output JSON murni:
{
  "start_sentence": 0,
  "end_sentence": 0,
  "confidence": 0.0
}

== QUESTION ==
{question}

== ANSWER ==
{answer}

== CONTEXT ==
(Konteks dipecah oleh kamu menjadi daftar kalimat berurut)

"""
    print("Prompt untuk LLM:\n", prompt)
    print("----\n\n")
    try:
        raw = safe_llm_call(prompt)

        # Jika LLM total gagal, langsung ke fallback heuristik
        if raw is None:
            print("[WARN] LLM returned None, using heuristic fallback")
            return heuristic_fallback_span(context, answer)

        # Bersihkan block code jika ada
        raw = raw.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        print("Raw LLM output:\n", raw)

        # Tahap 1: coba parse JSON langsung
        try:
            js = json.loads(raw)
        except Exception:
            # Tahap 2: coba parser robust
            try:
                js = robust_json_parse(raw)
            except Exception as e:
                print("[WARN] robust_json_parse failed:", e)
                # Tahap 3: fallback heuristik
                return heuristic_fallback_span(context, answer)

        span = js.get("span", "").strip()
        conf = float(js.get("confidence", 0))

        # Jika span terlalu pendek atau kosong, pakai fallback
        if len(span) < 3:
            print("[WARN] Span too short from LLM, using heuristic fallback")
            return heuristic_fallback_span(context, answer)

        # Pastikan span benar benar ada di dalam context, kalau tidak, fallback
        if span not in context:
            print("[WARN] LLM span not found in context, using heuristic fallback")
            return heuristic_fallback_span(context, answer)
        
        print("=====\n\n\n\n")
        return span, conf

    except Exception as e:
        print("LLM extraction error:", e)
        traceback.print_exc()
        return heuristic_fallback_span(context, answer)

# ===========================
# VALIDATOR
# ===========================

def validate_atomic_subset(df):
    """
    Cek apakah atomic_context_groundtruth merupakan subset context asli.
    Tambahkan kolom:
    - atomic_is_valid
    - atomic_match_chars
    - atomic_match_ratio
    """

    is_valid = []
    match_chars = []
    match_ratio = []

    print("\nValidating atomic spans...")

    for i, row in tqdm(df.iterrows(), total=len(df)):

        context = str(row["context"])
        atomic = str(row["atomic_context_groundtruth"])

        # Normalisasi ringan
        ctx = " ".join(context.split())
        atm = " ".join(atomic.split())

        if atm == "UNK":
            is_valid.append(0)
            match_chars.append(0)
            match_ratio.append(0)
            continue

        # Cek substring exact
        # Cek substring exact dengan robust unicode
        if robust_contains(context, atomic):
            # Gunakan normalized versi untuk boundary check
            if validate_contiguous_span(normalize_u(context), normalize_u(atomic)) == 1:
                is_valid.append(1)
                match_chars.append(len(atomic))
                match_ratio.append(len(atomic) / len(context))
            else:
                is_valid.append(1)   # masih valid meski boundary tidak perfect
                match_chars.append(len(atomic))
                match_ratio.append(len(atomic) / len(context))
        else:
            is_valid.append(0)
            match_chars.append(0)
            match_ratio.append(0)



    df["atomic_is_valid"] = is_valid
    df["atomic_match_chars"] = match_chars
    df["atomic_match_ratio"] = match_ratio

    return df
import re

def robust_json_parse(raw):
    raw = raw.strip()

    # Remove markdown
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Extract first JSON block
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("No JSON object detected")

    candidate = match.group(0)

    # Clean common LLM errors
    candidate = candidate.replace("\n", "\\n")
    candidate = re.sub(r",\s*}", "}", candidate)  # remove trailing comma
    candidate = re.sub(r",\s*]", "]", candidate)

    # Fix unescaped quotes in value strings
    def escape_value(m):
        key, val = m.group(1), m.group(2)
        val = val.replace('"', '\\"')
        return f'"{key}": "{val}"'

    candidate = re.sub(r'"([^"]*)":\s*"([^"]*)"', escape_value, candidate)

    # Parse JSON
    return json.loads(candidate)


def export_invalid_cases(df, output_path):
    """Simpan baris yang atomic-nya tidak valid."""
    invalid_df = df[df["atomic_is_valid"] == 0]

    if len(invalid_df) > 0:
        error_path = output_path.replace(".xlsx", "-errors.xlsx")
        invalid_df.to_excel(error_path, index=False)
        format_excel(error_path)

        print(f"❗ Found {len(invalid_df)} invalid atomic spans. Saved to: {error_path}")
    else:
        print("✔ Semua atomic spans valid. Tidak ada error.")


# ===========================
# GENERATOR
# ===========================

def generate_atomic_dataset(input_path, output_path):
    print("Loading dataset:", input_path)
    df = pd.read_excel(input_path)
    #df=df.head(5)
    print("Loaded:", df.shape)

    atomic_spans = []
    atomic_conf = []
    atomic_sent_count = []

    print("Generating atomic context ground truth ...\n")

    for i, row in tqdm(df.iterrows(), total=len(df)):

        context = str(row["context"])
        question = str(row["question"])
        answer = str(row["answer_groundtruth"])

        #span, conf = extract_atomic_span(context, question, answer)
        span, conf = extract_atomic_span_by_index(context, question, answer)


        atomic_spans.append(span)
        atomic_conf.append(conf)

        sent_count = span.count(".") + span.count("!") + span.count("?")
        atomic_sent_count.append(max(sent_count, 1))

    df["atomic_context_groundtruth"] = atomic_spans
    df["atomic_extraction_confidence"] = atomic_conf
    df["atomic_sentence_count"] = atomic_sent_count

    # Jalankan validator
    df = validate_atomic_subset(df)

    # Simpan dataset utama
    print("Saving output:", output_path)
    df.to_excel(output_path, index=False)
    format_excel(output_path)

    # export error cases
    export_invalid_cases(df, output_path)

    print("\n✔ DONE. Atomic dataset successfully generated & validated.\n")

def regenerate_invalid_atomic(input_path, output_path):
    print("Loading dataset:", input_path)
    df = pd.read_excel(input_path)
    print("Loaded:", df.shape)

    new_atomic_spans = []
    new_atomic_conf = []
    new_atomic_sent_count = []

    print("\nRegenerating only invalid atomic spans...\n")

    for i, row in tqdm(df.iterrows(), total=len(df)):

        context = str(row["context"])
        question = str(row["question"])
        answer = str(row["answer_groundtruth"])

        prev_valid = row.get("atomic_is_valid", 0)

        if prev_valid == 1:
            # ⬆ baris valid → copy saja
            new_atomic_spans.append(row["atomic_context_groundtruth"])
            new_atomic_conf.append(row["atomic_extraction_confidence"])
            new_atomic_sent_count.append(row["atomic_sentence_count"])
            continue

        # ⬇ baris invalid → generate ulang
        span, conf = extract_atomic_span(context, question, answer)

        new_atomic_spans.append(span)
        new_atomic_conf.append(conf)

        sent_count = span.count(".") + span.count("!") + span.count("?")
        new_atomic_sent_count.append(max(sent_count, 1))

    # masukkan hasil baru
    df["atomic_context_groundtruth"] = new_atomic_spans
    df["atomic_extraction_confidence"] = new_atomic_conf
    df["atomic_sentence_count"] = new_atomic_sent_count

    # validasi ulang
    df = validate_atomic_subset(df)

    print("\nSaving output:", output_path)
    df.to_excel(output_path, index=False)
    format_excel(output_path)

    # simpan error
    export_invalid_cases(df, output_path)

    print("\n✔ DONE. Only invalid spans regenerated.\n")

def regenerate_atomic_smart(df):
    """
    Hanya mengulang generation + validation untuk baris yang:
    - atomic_context_groundtruth == 'UNK'
    - atau atomic_is_valid_final == 0
    """

    new_spans = []
    new_conf = []
    new_sent = []
    generated_flag = []   # <--- NEW

    print("\n🚀 Smart regeneration started...\n")

    for i, row in tqdm(df.iterrows(), total=len(df)):

        context = str(row["context"])
        question = str(row["question"])
        answer = str(row["answer_groundtruth"])

        atomic_old = str(row.get("atomic_context_groundtruth", "UNK"))
        valid_final = int(row.get("atomic_is_valid_final", 0))

        # =====================================================
        # CASE 1 & 2 → regenerate
        # =====================================================
        if atomic_old == "UNK" or valid_final == 0:

            span, conf = extract_atomic_span_by_index(context, question, answer)

            sent_count = span.count(".") + span.count("!") + span.count("?")
            sent_count = max(sent_count, 1)

            new_spans.append(span)
            new_conf.append(conf)
            new_sent.append(sent_count)

            generated_flag.append(1)    # <--- NEW FLAG

        # =====================================================
        # CASE 3 → copy
        # =====================================================
        else:
            new_spans.append(atomic_old)
            new_conf.append(row["atomic_extraction_confidence"])
            new_sent.append(row["atomic_sentence_count"])

            generated_flag.append(0)    # <--- NEW FLAG

    # Update dataframe
    df["atomic_context_groundtruth"] = new_spans
    df["atomic_extraction_confidence"] = new_conf
    df["atomic_sentence_count"] = new_sent
    df["atomic_generated_new"] = generated_flag   # <--- NEW COLUMN

    # VALIDASI ULANG
    df = validate_atomic_all_levels(df)

    return df

def validator_atomic_substring_only(df):
    """
    Validator paling longgar: hanya cek apakah atomic span merupakan substring context.
    Tidak ada boundary checking.
    Tidak ada normalization strict.

    Hasil:
    - atomic_is_valid_loose
    """

    valid = []
    for i, row in tqdm(df.iterrows(), total=len(df)):

        context = str(row["context"])
        atomic  = str(row["atomic_context_groundtruth"])

        # clean minimal
        ctx = context.replace("\n", " ").strip()
        atm = atomic.replace("\n", " ").strip()

        if atm == "UNK" or len(atm) < 3:
            valid.append(0)
            continue

        if atm in ctx:
            valid.append(1)
        else:
            valid.append(0)

    df["atomic_is_valid_loose"] = valid
    print("✔ Loose validation complete")
    return df

def validate_atomic_all_levels(df):
    """
    Validator terpadu dengan tiga level:
    1. STRICT     : substring + boundary contiguous (paling ketat)
    2. CONTIGUOUS : boundary check saja
    3. LOOSE      : substring saja (tanpa boundary)

    Menghasilkan:
    - atomic_is_valid_strict
    - atomic_is_valid_contiguous
    - atomic_is_valid_loose
    - atomic_is_valid_final   (gabungan paling fair)
    """

    strict_list = []
    contiguous_list = []
    loose_list = []
    final_list = []

    print("\n🔍 Running unified validation (strict + contiguous + loose)...\n")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        ctx_raw = str(row["context"])
        atm_raw = str(row["atomic_context_groundtruth"])

        # minimal cleaning
        ctx = ctx_raw.replace("\n", " ").strip()
        atm = atm_raw.replace("\n", " ").strip()

        # default invalid
        strict_ok = 0
        contiguous_ok = 0
        loose_ok = 0

        if atm != "UNK" and len(atm) >= 3:

            # ---- LEVEL 3: LOOSE (substring saja)
            if atm in ctx:
                loose_ok = 1

            # ---- LEVEL 2: CONTIGUOUS (boundary)
            if loose_ok == 1:
                if validate_contiguous_span(ctx, atm) == 1:
                    contiguous_ok = 1

            # ---- LEVEL 1: STRICT (substring + contiguous)
            if loose_ok == 1 and contiguous_ok == 1:
                strict_ok = 1

        # final logic (fair and inclusive)
        if strict_ok == 1:
            final_ok = 1
        elif loose_ok == 1:
            # meski boundary fail, tetap valid sebagai atomic span
            final_ok = 1
        else:
            final_ok = 0

        strict_list.append(strict_ok)
        contiguous_list.append(contiguous_ok)
        loose_list.append(loose_ok)
        final_list.append(final_ok)

    df["atomic_is_valid_strict"] = strict_list
    df["atomic_is_valid_contiguous"] = contiguous_list
    df["atomic_is_valid_loose"] = loose_list
    df["atomic_is_valid_final"] = final_list

    print("\n✔ Unified validation completed.")
    print(f"  STRICT valid     : {df['atomic_is_valid_strict'].sum()}")
    print(f"  CONTIGUOUS valid : {df['atomic_is_valid_contiguous'].sum()}")
    print(f"  LOOSE valid      : {df['atomic_is_valid_loose'].sum()}")
    print(f"  FINAL valid      : {df['atomic_is_valid_final'].sum()}")

    return df


# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    INPUT = "datasets/validated-atomicv2.xlsx"
    OUTPUT = "datasets/validated-atomicv4-revalidated.xlsx"

    #generate_atomic_dataset(INPUT, OUTPUT)
    #regenerate_invalid_atomic(INPUT, OUTPUT)
    df = pd.read_excel(INPUT)
    #df = df[df["atomic_context_groundtruth"] == "UNK"].reset_index(drop=True)
    #df = df.head(3)  # untuk testing saja

    print("Jumlah baris UNK:", len(df))
    df = validate_atomic_all_levels(df)

    # Langkah 2: smart regeneration (UNK atau final_invalid)
    df = regenerate_atomic_smart(df)

    # Langkah 3: validasi lagi (biar yakin)
    df = validate_atomic_all_levels(df)

    df.to_excel(OUTPUT, index=False)
    format_excel(OUTPUT)

    print("\n✔ DONE. Smart atomic regeneration saved.\n")
