import kagglehub
import shutil
import os

dataset_path = kagglehub.dataset_download("thecoderenroute/instagram-posts-dataset")

# Hedef klasör yolu
target_folder = "data"

# Eğer hedef klasör zaten varsa önce temizleyelim:
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)

# Dataset klasörünü hedef klasöre kopyalayalım:
shutil.copytree(dataset_path, target_folder, dirs_exist_ok=True)

print("✅ Veri seti başarıyla indirildi ve 'data' klasörüne aktarıldı.")
