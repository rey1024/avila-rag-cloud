import pandas as pd
import json

def excel_to_jsonl(excel_path, jsonl_path):
    # Baca Excel
    df = pd.read_excel(excel_path)

    # Simpan sebagai JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.to_dict()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✔ Konversi selesai → {jsonl_path}")


# Contoh pemanggilan
excel_to_jsonl(
    excel_path="answers_20251130_074721-tambahan.xlsx",
    jsonl_path="answers_20251130_074721-tambahan.jsonl"
)
