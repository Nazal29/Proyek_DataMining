# Nama file: siapkan_data.py (Ini adalah file DATA PREPROCESSING)
import pandas as pd
import numpy as np
import os

FOLDER_OLIST = "Brazilian E-Commerce Public Dataset by Olist"

print("--- MEMULAI DATA PREPROCESSING ---")
# 1. INTEGRASI DATA: Baca dan Gabungkan
orders = pd.read_csv(f"{FOLDER_OLIST}/olist_orders_dataset.csv")
items = pd.read_csv(f"{FOLDER_OLIST}/olist_order_items_dataset.csv")
df = pd.merge(items, orders, on='order_id')

# 2. DATA TRANSFORMATION: Format Waktu
df['Date'] = pd.to_datetime(df['order_purchase_timestamp']).dt.date

# 3. AGREGASI DATA: Rangkum per Hari
daily_sales = df.groupby(['Date', 'product_id']).agg(
    Price=('price', 'mean'),
    Quantity_Sold=('order_id', 'count')
).reset_index()

# 4. FEATURE ENGINEERING: Simulasi Harga Kompetitor
np.random.seed(42)
daily_sales['Competitor_Price'] = daily_sales['Price'] * np.random.uniform(0.95, 1.05, len(daily_sales))

# 5. FILTERING: Ambil 1 produk terlaris
top_product = daily_sales.groupby('product_id')['Quantity_Sold'].sum().idxmax()
df_final = daily_sales[daily_sales['product_id'] == top_product]

# 6. EXPORT PROCESSED DATA
NAMA_FILE_BARU = "dataset_olist_siap_train.csv"
df_final.to_csv(NAMA_FILE_BARU, index=False)

print(f"PREPROCESSING SELESAI! File '{NAMA_FILE_BARU}' berhasil dibuat.")