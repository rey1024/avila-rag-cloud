from huggingface_hub import snapshot_download

base_dir = "models/embeding"

models = {
    "BAAI/bge-m3": "bge-m3",
    "Snowflake/snowflake-arctic-embed-l-v2.0": "snowflake-arctic-embed-l-v2.0",
    "google/embeddinggemma-300m": "embeddinggemma-300m",
    "Alibaba-NLP/gte-multilingual-base": "gte-multilingual-base",
    "intfloat/multilingual-e5-base": "multilingual-e5-base"
}

for hf_id, folder_name in models.items():
    print(f"⬇ Downloading: {hf_id}")
    
    snapshot_download(
        repo_id=hf_id,
        local_dir=f"{base_dir}/{folder_name}",
        local_dir_use_symlinks=False
    )

print("✅ Semua model sudah berhasil didownload")
