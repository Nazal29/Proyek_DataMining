# Nama file: dataset_tracker.py (Ini adalah file DATASET TRACKING)
import pandas as pd
import hashlib
import json
import os
from datetime import datetime

FILE_DATASET = "dataset_olist_siap_train.csv"
LOG_FILE = "dataset_version_log.json"

def generate_file_hash(filepath):
    """Membuat sidik jari (MD5 Hash) unik dari file CSV"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def track_dataset():
    if not os.path.exists(FILE_DATASET):
        print(f"Error: Dataset {FILE_DATASET} tidak ditemukan!")
        return

    # 1. Baca metadata dari CSV
    df = pd.read_csv(FILE_DATASET)
    total_baris = len(df)
    total_kolom = len(df.columns)
    kolom_list = list(df.columns)
    
    # 2. Buat Hash untuk tracking versi
    file_hash = generate_file_hash(FILE_DATASET)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 3. Susun data tracking
    tracking_data = {
        "timestamp": timestamp,
        "filename": FILE_DATASET,
        "version_hash": file_hash,
        "metadata": {
            "total_rows": total_baris,
            "total_columns": total_kolom,
            "columns": kolom_list
        },
        "description": "Dataset matang hasil preprocessing Olist siap pakai untuk RL"
    }

    # 4. Simpan ke file log JSON
    log_history = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as file:
            try:
                log_history = json.load(file)
            except json.JSONDecodeError:
                log_history = []

    log_history.append(tracking_data)

    with open(LOG_FILE, "w") as file:
        json.dump(log_history, file, indent=4)

    print("-" * 50)
    print("DATASET TRACKING BERHASIL DICATAT!")
    print(f"Versi Hash : {file_hash}")
    print(f"Total Data : {total_baris} baris, {total_kolom} kolom")
    print(f"Log tersimpan di file: {LOG_FILE}")
    print("-" * 50)

if __name__ == "__main__":
    track_dataset()