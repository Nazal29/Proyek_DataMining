import os
import zipfile
import subprocess

def collect_kaggle_data():
    dataset_name = "olistbr/brazilian-ecommerce"
    download_dir = "Brazilian E-Commerce Public Dataset by Olist"
    zip_file_path = os.path.join(download_dir, "brazilian-ecommerce.zip")

    print("="*50)
    print("MEMULAI PROSES DATA COLLECTION (Versi Final API)")
    print("="*50)
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"[*] Folder '{download_dir}' berhasil dibuat.")

    # 1. Injeksi Token Langsung ke Environment
    env_vars = os.environ.copy()
    env_vars["KAGGLE_API_TOKEN"] = "KGAT_194da5d73c8f372d93573efcdc864714"

    print(f"[*] Mengunduh dataset '{dataset_name}' dari Kaggle...")
    try:
        # 2. Eksekusi perintah langsung di Shell Windows
        # Menggunakan format string biasa dan shell=True agar venv langsung mengenali 'kaggle'
        command = f'kaggle datasets download -d {dataset_name} -p "{download_dir}"'
        
        subprocess.run(
            command, 
            env=env_vars,
            check=True,
            shell=True # Ini kunci utamanya biar Windows nggak bingung
        )
        print("[*] Proses download dari server selesai.")
        
    except subprocess.CalledProcessError as e:
        print("\n[ERROR] Download gagal dari server Kaggle.")
        print("Pastikan koneksi internet stabil.")
        return
    except Exception as e:
        print(f"\n[ERROR] Kesalahan sistem internal: {e}")
        return

    # 3. Ekstrak file ZIP
    print("[*] Mengurai (Unzip) data mentah...")
    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        
        # Bersihkan file ZIP biar folder tetap rapi
        os.remove(zip_file_path)
        print("[*] File ZIP berhasil diekstrak dan dibersihkan.")
        
        # 4. Verifikasi hasil
        total_files = len([name for name in os.listdir(download_dir) if name.endswith('.csv')])
        print("\n" + "="*50)
        print(f"DATA COLLECTION SELESAI!")
        print(f"Total {total_files} file CSV mentah berhasil dikumpulkan di folder:")
        print(f"'{download_dir}'")
        print("="*50)
    else:
        print("\n[ERROR] File ZIP tidak ditemukan setelah diunduh. Mungkin proses download terputus.")

if __name__ == "__main__":
    collect_kaggle_data()