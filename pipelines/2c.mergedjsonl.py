import json
import pandas as pd
import datetime
import os
import sys
# memastikan folder project menjadi root import
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils_common.utils import format_excel
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
# -------------------------------
# KONFIGURASI INPUT DAN OUTPUT
# -------------------------------
jsonl_files = [
    "results/ds_question/ds_1-3_20251130_064157/bloom_1_3_.jsonl",
    "results/ds_question/ds_4_6_20251130_070056/bloom_4_6_20251130_070056.jsonl"
    
]

xlsx_files = [
    "results/ds_question/ds_1-3_20251130_064157/bloom_1_3_.xlsx",
    "results/ds_question/ds_4_6_20251130_070056/bloom_4_6_20251130_070056.xlsx"
    
]

output_jsonl = f"datasets/merged_dataset{get_timestamp()}.jsonl"
output_xlsx = f"datasets/merged_dataset{get_timestamp()}.xlsx"

# -------------------------------
# PEMETAAN BLOOM LEVEL KE ANGKA
# -------------------------------
bloom_map = {
    "Remember": 1,
    "Understand": 2,
    "Apply": 3,
    "Analyze": 4,
    "Evaluate": 5,
    "Create": 6
}


def filter_existing_files(file_list):
    existing = []
    for f in file_list:
        if os.path.exists(f):
            existing.append(f)
        else:
            print(f"⚠️ FILE TIDAK DITEMUKAN → {f}")
    return existing

# ----------------------------------------------------------
# 1. FUNGSI MERGE DAN DEDUPLIKASI JSONL
# ----------------------------------------------------------
def merge_jsonl(in_files, out_file):
    merged = []

    for fpath in in_files:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)

                    # ubah bloom_level teks → angka
                    if "bloom_level" in obj and obj["bloom_level"] in bloom_map:
                        obj["bloom_level"] = bloom_map[obj["bloom_level"]]

                    merged.append(obj)

                except:
                    print(f"Skipped invalid line: {line}")

    # Hapus duplikat berdasarkan pertanyaan
    seen = set()
    unique = []
    for obj in merged:
        q = obj.get("question", "").strip().lower()
        if q not in seen:
            seen.add(q)
            unique.append(obj)

    # Urutkan Bloom Level
    unique_sorted = sorted(unique, key=lambda x: x.get("bloom_level", 999))

    # Tambahkan ID auto increment
    for i, obj in enumerate(unique_sorted, start=1):
        obj["id"] = i

    # Simpan JSONL
    with open(out_file, "w", encoding="utf-8") as out:
        for obj in unique_sorted:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return len(unique_sorted), unique_sorted

# ----------------------------------------------------------
# 2. FUNGSI MERGE DAN DEDUPLIKASI XLSX
# ----------------------------------------------------------
def merge_xlsx(in_files, out_file):
    dfs = []

    for fpath in in_files:
        df = pd.read_excel(fpath)

        # konversi bloom → angka
        if "bloom_level" in df.columns:
            df["bloom_level"] = df["bloom_level"].map(bloom_map).fillna(df["bloom_level"])

        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    # Hapus duplikat berdasarkan pertanyaan
    if "question" in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=["question"], keep="first")

    # Urutkan berdasarkan Bloom Level
    if "bloom_level" in merged_df.columns:
        merged_df = merged_df.sort_values("bloom_level", ascending=True)

    # Tambahkan kolom ID
    merged_df.insert(0, "id", range(1, len(merged_df) + 1))

    # Simpan XLSX
    merged_df.to_excel(out_file, index=False)
    return merged_df


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":

    print("🔍 Memeriksa file JSONL...")
    jsonl_files_checked = filter_existing_files(jsonl_files)

    print("🔍 Memeriksa file XLSX...")
    xlsx_files_checked = filter_existing_files(xlsx_files)

    print("\n🔄 Merge JSONL + dedup...")
    total_jsonl, merged_records = merge_jsonl(jsonl_files_checked, output_jsonl)
    if total_jsonl > 0:
        print(f"✔ JSONL selesai: {total_jsonl} baris unik → {output_jsonl}")

    print("\n🔄 Merge XLSX + dedup...")
    merged_df = merge_xlsx(xlsx_files_checked, output_xlsx)
    if not merged_df.empty:
        print(f"✔ XLSX selesai: {len(merged_df)} baris unik → {output_xlsx}")

    print("\n🎉 SEMUA SELESAI!")
    print("Dataset sudah dicek, digabung, dibersihkan duplikat, dan diurutkan.")

    print("🔄 Merge JSONL + dedup...")
    total_jsonl, merged_records = merge_jsonl(jsonl_files, output_jsonl)
    print(f"✔ JSONL selesai: {total_jsonl} baris unik → {output_jsonl}")

    print("🔄 Merge XLSX + dedup...")
    merged_df = merge_xlsx(xlsx_files, output_xlsx)
    print(f"✔ XLSX selesai: {len(merged_df)} baris unik → {output_xlsx}")
    format_excel(output_xlsx)   # <<--- tambahkan ini

    print("\n🎉 SEMUA BERHASIL!")
    print("Dataset sudah digabung, distandarkan, dibersihkan duplikat, dan diurutkan.")
