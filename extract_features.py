import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
from mtcnn import MTCNN
from skimage.measure import shannon_entropy

# Parlaklık hesaplama
def calculate_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[..., 2].mean()

# Keskinlik hesaplama
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

# Bulanıklık tespiti (düşük Laplacian değeri = bulanık)
def calculate_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Kontrast hesaplama
def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

# Entropi hesaplama
def calculate_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(gray)

# Renk canlılığı hesaplama
def calculate_colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

# Ortalama renk yoğunluğu
def calculate_average_color(image):
    avg_color = image.mean(axis=0).mean(axis=0)
    return avg_color

# MTCNN ile yüz sayısı (dinamik oluşturma)
def detect_faces(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    results = detector.detect_faces(rgb)
    return len(results)

# En-boy oranı
def calculate_aspect_ratio(image):
    h, w = image.shape[:2]
    return w / h

# Kenar yoğunluğu
def edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size

# Dominant renk
def dominant_colors(image, k=2):
    pixels = np.float32(image.reshape(-1, 3))
    _, labels, palette = cv2.kmeans(pixels, k, None,
                                    (cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2),
                                    5, cv2.KMEANS_RANDOM_CENTERS)
    dominant = palette[np.argmax(np.unique(labels, return_counts=True)[1])]
    return dominant
"""
# Klasör ayarı
image_folder = 'data'

# Veri listesi
features_list = []

# Görsel toplama
image_files = []
for root, _, files in os.walk(image_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))


# Görselleri işle
for idx, image_path in enumerate(image_files):
    try:
        folder_name = os.path.basename(os.path.dirname(image_path))
        likes = int(folder_name.split('_')[-2])
        followers = int(folder_name.split('_')[1])
        if likes <= 0:
            continue
        if followers <= 0:
            continue
        img = Image.open(image_path).convert("RGB")
        img = img.resize((200, 200))
        image = np.array(img)

        brightness = calculate_brightness(image)
        sharpness = calculate_sharpness(image)
        blur_score = calculate_blur_score(image)
        contrast = calculate_contrast(image)
        entropy = calculate_entropy(image)
        colorfulness = calculate_colorfulness(image)
        faces = detect_faces(image)
        aspect_ratio = calculate_aspect_ratio(image)
        edge = edge_density(image)
        dominant = dominant_colors(image)
        avg_color = calculate_average_color(image)

        features_list.append({
            'image_name': os.path.basename(image_path),
            'brightness': brightness,
            'sharpness': sharpness,
            'blur_score': blur_score,
            'contrast': contrast,
            'entropy': entropy,
            'colorfulness': colorfulness,
            'faces': faces,
            'aspect_ratio': aspect_ratio,
            'edge': edge,
            'color_r': dominant[2],
            'color_g': dominant[1],
            'color_b': dominant[0],
            'avg_r': avg_color[2],
            'avg_g': avg_color[1],
            'avg_b': avg_color[0],
            'likes': likes,
            'followers': followers
        })
        print(f"[{idx+1}/{len(image_files)}] İşlendi: {os.path.basename(image_path)} | Likes: {likes} | Followers: {followers}")

    except Exception as e:
        print(f"⚠️ Hata: {image_path} işlenemedi - {e}")
        continue

# CSV'ye kaydet
if features_list:
    df = pd.DataFrame(features_list)
    df.to_csv('data/features.csv', index=False)
    print("✅ Özellik çıkarımı tamamlandı, 'data/features.csv' oluşturuldu.")
else:
    print("❌ Hiç özellik çıkarılamadı, CSV boş.")
"""