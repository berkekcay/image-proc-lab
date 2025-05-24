import streamlit as st
import joblib
import numpy as np
import cv2
from mtcnn import MTCNN
from PIL import Image
from extract_features import (
    calculate_brightness,
    calculate_sharpness,
    dominant_colors,
    calculate_contrast,
    calculate_entropy,
    calculate_colorfulness,
    detect_faces,
    calculate_aspect_ratio,
    edge_density
)
import time

# 🎯 Model yükleniyor
model = joblib.load("data/engagement_model.pkl")

# 🌐 Sayfa ayarları
st.set_page_config(page_title="AI Görsel Etkileşim Analizi", layout="wide")

# ✨ Genel stil
st.markdown("""
    <style>
        .center-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 30px;
            padding: 30px 0;
        }

        .custom-box {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            width: 100%;
            max-width: 700px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ✨ Başlık ve yükleme kutuları
with st.container():
    st.markdown("""
        <div class="center-wrapper">
            <div class="custom-box">
                <h1 style='margin-bottom: 0.5rem; color: black;'>📸 AI GÖRSEL ETKİLEŞİM ANALİZİ</h1>
                <p style='font-size: 17px; color: black;'>Yüklediğiniz görseli analiz ederek tahmini beğeni sayısını ve içerik kalitesi hakkında öneriler sunar.</p>
            </div>

    """, unsafe_allow_html=True)

    st.markdown("""           
    <div class="custom-box">
        <h4 style='color: black;'>🔍 Görsel Yükleyin 🔍</h4>
        <p style='color: gray; font-size: 14px;'>JPG, PNG , JPEG — Max  200MB</p>
    </div>
""", unsafe_allow_html=True)

    st.markdown("<p style='font-size:18px; font-weight:bold;'>📁 Görsel seçin veya sürükleyin (.jpg, .png)</p>", unsafe_allow_html=True)

uploaded_image = st.file_uploader(
    label="",  # artık boş
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="🖼️ Yüklenen Görsel", use_container_width=True)

    if st.button("🔍 Analizi Başlat"):
        with st.spinner("🧠 Görsel analiz ediliyor..."):
            progress = st.progress(0)
            for i in range(1, 6):
                time.sleep(0.2)
                progress.progress(i * 20)

            image_path = "temp.jpg"
            image.save(image_path)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (200, 200))

            contrast = calculate_contrast(img)
            entropy = calculate_entropy(img)
            colorfulness = calculate_colorfulness(img)
            faces = detect_faces(img)
            aspect_ratio = calculate_aspect_ratio(img)
            edges = edge_density(img)
            brightness = calculate_brightness(img)
            sharpness = calculate_sharpness(img)
            dominant = dominant_colors(img)
            features = np.array([[brightness, sharpness, dominant[2], dominant[1], dominant[0]]])
            predicted_likes = model.predict(features)[0]

        with st.container():
            st.markdown("""
                <div style="
        display: flex;
        justify-content: center;
        margin-top: 2rem;
    ">
        <div class="custom-box" style="width: 100%; max-width: 700px;">
                    <h3 style='color: black;'>📊 Analiz Sonuçları 📊</h3>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)

            c1.metric("🌟 Parlaklık", f"{brightness:.2f}")
            c2.metric("🔍 Netlik", f"{sharpness:.2f}")
            c3.metric("❤️ Beğeni Tahmini", f"{int(predicted_likes)}")

            rgb_hex = f"#{int(dominant[2]):02x}{int(dominant[1]):02x}{int(dominant[0]):02x}"
            st.markdown("### 🧪 Ekstra Özellikler")
            st.write(f"📏 Kontrast: {contrast:.2f}")
            st.write(f"🔀 Entropi: {entropy:.2f}")
            st.write(f"🌈 Renk Canlılığı: {colorfulness:.2f}")
            st.write(f"👤 Yüz Sayısı (MTCNN): {faces}")
            st.write(f"📐 En-Boy Oranı: {aspect_ratio:.2f}")
            st.write(f"🧩 Kenar Yoğunluğu: {edges:.4f}")
            st.markdown("### 🎨 Dominant Renk")
            st.color_picker("Dominant RGB", value=rgb_hex, label_visibility="collapsed")
            st.write(f"RGB: ({int(dominant[2])}, {int(dominant[1])}, {int(dominant[0])})")

            st.markdown("### 💡 Optimizasyon Önerileri")

            if brightness < 100:
                st.warning("📢 Görsel karanlık. Daha parlak bir versiyon deneyin.")
            elif brightness > 200:
                st.warning("📢 Görsel çok parlak. Kontrastı azaltabilirsiniz.")

            if sharpness < 200:
                st.warning("📢 Görsel net değil. Daha yüksek çözünürlüklü bir görsel kullanabilirsiniz.")
            elif sharpness > 1000:
                st.warning("📢 Görsel fazla keskin olabilir. Doğal görünüme dikkat edin.")

            if predicted_likes < 500:
                st.info("📌 Etkileşim düşük olabilir. Renk ve kompozisyonu gözden geçirin.")
            else:
                st.success("🎉 Görseliniz yüksek etkileşim alabilir!")

            if colorfulness < 20:
                st.info("🌈 Görselde renk canlılığı düşük. Daha canlı renkler kullanmayı düşünebilirsiniz.")
            elif colorfulness > 60:
                st.warning("🌈 Görsel çok renkli olabilir. Göz yorgunluğu yaratabilir.")

            if contrast < 30:
                st.info("🌓 Kontrast düşük. Görsel daha dikkat çekici olabilir.")
            elif contrast > 100:
                st.warning("🌓 Aşırı kontrast var. Doğal görünüm için dengeli kontrast önerilir.")

            if faces == 0:
                st.info("👤 Görselde yüz bulunamadı. İnsan yüzü içeren görseller daha fazla etkileşim alabilir.")
            else:
                st.success(f"👤 {faces} adet yüz algılandı. Bu etkileşimi artırabilir!")

            if edges < 0.01:
                st.info("🧩 Görsel çok sade. Dikkat çekmek için biraz detay eklenebilir.")
            elif edges > 0.10:
                st.warning("🧩 Görsel çok karmaşık. Basitlik her zaman iyidir!")

            if entropy < 4.0:
                st.info("🔀 Görselde bilgi yoğunluğu düşük. Daha zengin içerikli görseller denenebilir.")
            elif entropy > 7.0:
                st.warning("🔀 Görsel çok bilgi içeriyor olabilir. Dikkat dağıtıcı olabilir.")

            if aspect_ratio > 2:
                st.info("📐 Görsel çok yatay. Mobil görünümde bozulabilir.")
            elif aspect_ratio < 0.5:
                st.info("📐 Görsel çok dikey. Daha dengeli bir oran denenebilir.")
