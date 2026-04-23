import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Pastikan file data tersedia
FILE_CSV = "dataset_olist_siap_train.csv"

if not os.path.exists(FILE_CSV):
    print(f"Error: File '{FILE_CSV}' tidak ditemukan. Pastikan sudah menjalankan siapkan_data.py")
    exit()

# 2. Load Data
df = pd.read_csv(FILE_CSV)
df['Date'] = pd.to_datetime(df['Date']) # Pastikan format tanggal benar
produk_id = df['product_id'].iloc[0]

print("Memproses Exploratory Data Analysis (EDA)...")

# 3. Setup Canvas (Gambar berisi 3 grafik sekaligus)
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle(f'Exploratory Data Analysis (EDA)\nProduk Terlaris Olist ID: {produk_id}', fontsize=16, fontweight='bold')

# --- Grafik 1: Tren Penjualan Harian (Time Series) ---
axes[0].plot(df['Date'], df['Quantity_Sold'], color='royalblue', marker='.', linestyle='-', linewidth=1.5)
axes[0].set_title('Tren Kuantitas Penjualan Harian', fontsize=12)
axes[0].set_ylabel('Jumlah Terjual', fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].tick_params(axis='x', rotation=45)

# --- Grafik 2: Distribusi Fluktuasi Harga ---
axes[1].hist(df['Price'], bins=20, color='darkorange', edgecolor='black', alpha=0.7)
axes[1].set_title('Distribusi Harga Penjualan (Histogram)', fontsize=12)
axes[1].set_xlabel('Harga (Price)', fontsize=10)
axes[1].set_ylabel('Frekuensi (Hari)', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.6)

# --- Grafik 3: Scatter Plot (Harga vs Kuantitas Terjual) untuk melihat Elastisitas ---
axes[2].scatter(df['Price'], df['Quantity_Sold'], color='seagreen', alpha=0.6, s=50)
axes[2].set_title('Hubungan Harga vs Kuantitas (Visualisasi Elastisitas)', fontsize=12)
axes[2].set_xlabel('Harga (Price)', fontsize=10)
axes[2].set_ylabel('Jumlah Terjual (Quantity)', fontsize=10)
axes[2].grid(True, linestyle='--', alpha=0.6)

# Rapikan layout dan simpan
plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Menyisakan ruang untuk judul utama
nama_file_output = 'grafik_eda_minggu3.png'
plt.savefig(nama_file_output)
print(f"SELESAI! Grafik EDA berhasil disimpan sebagai '{nama_file_output}'")

# Tampilkan di layar
plt.show()