import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image

# Parlaklık hesaplama
def calculate_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[..., 2].mean()

# Keskinlik hesaplama
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

# Dominant renkleri bulma (HIZLANDIRILDI)
def dominant_colors(image, k=2):
    pixels = np.float32(image.reshape(-1, 3))
    _, labels, palette = cv2.kmeans(pixels, k, None, 
                                    (cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2), 
                                    5, cv2.KMEANS_RANDOM_CENTERS)
    dominant = palette[np.argmax(np.unique(labels, return_counts=True)[1])]
    return dominant

# Görsellerin bulunduğu ana klasör
image_folder = 'data'

# Özellikleri saklayacak liste
features_list = []
max_images = 100  # 🔥 İşlem süresini kısaltmak için sadece ilk 100 görseli işle

# Klasör içindeki tüm alt klasörleri gezerek resimleri işle
image_files = []
for root, _, files in os.walk(image_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

image_files = image_files[:max_images]  # 🔥 İlk 100 görselle sınırlandır

for idx, image_path in enumerate(image_files):
    try:
        # 🔥 OpenCV yerine PIL kullanarak açalım
        img = Image.open(image_path).convert("RGB")
        img = img.resize((200, 200))  # Görseli küçült
        image = np.array(img)  # NumPy array'e çevir

        brightness = calculate_brightness(image)
        sharpness = calculate_sharpness(image)
        dominant = dominant_colors(image)

        # Listeye ekle
        features_list.append({
            'image_name': os.path.basename(image_path),
            'brightness': brightness,
            'sharpness': sharpness,
            'color_r': dominant[2],
            'color_g': dominant[1],
            'color_b': dominant[0]
        })

        print(f"[{idx+1}/{len(image_files)}] İşlendi: {os.path.basename(image_path)}")

    except Exception as e:
        print(f"⚠️ Hata: {image_path} işlenemedi - {e}")
        continue

# Özellikleri CSV'ye kaydet
if features_list:
    df = pd.DataFrame(features_list)
    df.to_csv('data/features.csv', index=False)
    print("✅ Özellik çıkarımı tamamlandı, 'data/features.csv' oluşturuldu.")
else:
    print("❌ Hiç özellik çıkarılamadı, CSV boş.")
