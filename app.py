import streamlit as st
import joblib
import numpy as np
import cv2
import altair as alt
import pandas as pd
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
    edge_density,
    calculate_blur_score
)
import time


# 🎯 Model yükleniyor
model = joblib.load("data/engagement_model.pkl")

# 🌐 Sayfa ayarları
st.set_page_config(page_title="AI Görsel Etkileşim Analizi", layout="wide")


# ✨ Genel stil
st.markdown("""
    <style>
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            background-color: #121212 !important;
            color: #e0e0e0 !important;
        }

        .center-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 30px;
            padding: 30px 0;
        }

        .custom-box {
            background: rgba(30, 30, 30, 0.95);
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 700px;
            text-align: center;
            backdrop-filter: blur(6px);
            color: #ffffff !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMarkdownContainer"] {
            color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)


# ✨ Başlık ve yükleme kutuları
with st.container():
    st.markdown("""
        <div class="center-wrapper">
            <div class="custom-box">
                <h1 style='margin-bottom: 0.5rem; color: black;'> AI GÖRSEL ETKİLEŞİM ANALİZİ</h1>
                <p style='font-size: 17px; color: black;'>Yüklediğiniz görseli analiz ederek tahmini beğeni sayısını ve içerik kalitesi hakkında öneriler sunar.</p>
            </div>

    """, unsafe_allow_html=True)


st.markdown("""
<div style='display: flex; justify-content: center;'>
    <p style='font-size:18px; font-weight:bold;'>📁 Görsel seçin veya sürükleyin (.jpg, .png)</p>
</div>
""", unsafe_allow_html=True)


followers = st.number_input("👥 Takipçi Sayınızı Girin:", min_value=1, value=1000, step=100)


uploaded_image = st.file_uploader(
    label="",
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
            blur_score = calculate_blur_score(img)
            edge = edge_density(img)
            features = np.array([[brightness, blur_score, faces, contrast, edge, colorfulness, entropy, sharpness, dominant[2], dominant[1], dominant[0], followers]])
            predicted_likes = model.predict(features)[0]

            # Grafik bloğu burada ekleniyor
            metrics_df = pd.DataFrame({
                'Ozellik': ['Parlaklık', 'Netlik', 'Kontrast', 'Entropi', 'Renk Canlılığı'],
                'Deger': [brightness, sharpness, contrast, entropy, colorfulness]
            })

            chart = alt.Chart(metrics_df).mark_bar().encode(
                x=alt.X('Ozellik', sort=None),
                y='Deger',
                color=alt.Color('Ozellik', legend=None),
                tooltip=[alt.Tooltip('Ozellik', title='Özellik'), alt.Tooltip('Deger', title='Değer')]
            ).properties(
                width=600,
                height=300,
                title='📊 Görsel Özellikler Grafiği'
)


            st.altair_chart(chart, use_container_width=True)

        with st.container():
            st.markdown("""
    <div style="
        display: flex;
        justify-content: center;
        margin-top: 4rem;
    "> 
    <div class="custom-box" style="width: 100%; max-width: 700px;">
        <h3 style='color: black; margin-top: 1rem; margin-bottom: 1.5rem;'>📊 Analiz Sonuçları 📊</h3>
""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("🌟 Parlaklık", f"{brightness:.2f}")
            c2.metric("🔍 Netlik", f"{sharpness:.2f}")
            c3.metric("❤️ Beğeni Tahmini", f"{int(predicted_likes)}")

            rgb_hex = f"#{int(dominant[2]):02x}{int(dominant[1]):02x}{int(dominant[0]):02x}"
            st.markdown(f"<div style='font-size:18px; font-weight:600;'> EKSTRA ÖZELLİKLER: {contrast:.2f}</div>", unsafe_allow_html=True)         
            st.markdown(f"<div style='font-size:18px; font-weight:600;'>🔷 Kontrast: {contrast:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:18px; font-weight:600;'>🔀 Entropi: {entropy:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:18px; font-weight:600;'>🌈 Renk Canlılığı: {colorfulness:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:18px; font-weight:600;'>👤 Yüz Sayısı (MTCNN): {faces}</div>", unsafe_allow_html=True)
            #st.markdown(f"<div style='font-size:18px; font-weight:600;'>📀 En-Boy Oranı: {aspect_ratio:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:18px; font-weight:600;'>🧩 Kenar Yoğunluğu: {edges:.4f}</div>", unsafe_allow_html=True)
            st.markdown("### 🎨 Dominant Renk", unsafe_allow_html=True)
            st.color_picker("Dominant RGB", value=rgb_hex, label_visibility="collapsed")
            st.markdown(f"<div style='font-size:18px; font-weight:600;'>RGB: ({int(dominant[2])}, {int(dominant[1])}, {int(dominant[0])})</div>", unsafe_allow_html=True)


            with st.expander("📊 Özellik Skalaları | Değerlerin Anlamı"):
                st.markdown("#### 🔆 Parlaklık (brightness)")
                st.markdown("- 🔴 0–80: Karanlık\n- 🟡 80–220: **İdeal**\n- 🔴 220+: Aşırı parlak")

                st.markdown("#### 🔍 Netlik (sharpness)")
                st.markdown("- 🔴 <150: Bulanık\n- 🟡 150–1200: **İdeal**\n- 🔵 >1200: Aşırı net / yapay görünüm")

                st.markdown("#### 🧪 Bulanıklık (blur_score)")
                st.markdown("- 🔴 <50: Yüksek bulanıklık (net değil)\n- 🟡 50–300: Orta düzey\n- 🔵 >300: Net görüntü")

                st.markdown("#### 🔷 Kontrast (contrast)")
                st.markdown("- 🔴 <30: Zayıf kontrast\n- 🟡 30–100: **İdeal**\n- 🔴 >100: Aşırı kontrast")

                st.markdown("#### 🧩 Kenar Yoğunluğu (edge_density)")
                st.markdown("- 🔵 <0.01: Çok sade\n- 🟡 0.01–0.10: **Dengeli**\n- 🔴 >0.10: Fazla detay / karışık")

                st.markdown("#### 🌈 Renk Canlılığı (colorfulness)")
                st.markdown("- 🔴 <20: Soluk renkler\n- 🟡 20–60: **Doğal canlılık**\n- 🔴 >60: Aşırı canlı / yapay görünüm")

                st.markdown("#### 🔀 Entropi (entropy)")
                st.markdown("- 🔵 <4: Sade\n- 🟡 4–7: **Dengeli bilgi yoğunluğu**\n- 🔴 >7: Aşırı karmaşık / dikkat dağıtıcı")


            st.markdown("## 💡 Detaylı Optimizasyon Önerileri")

            # Parlaklık
            if brightness < 80:
                st.warning("🌙 **Görsel karanlık görünüyor.**\n\nDaha fazla ışık, daha canlı bir görüntü sunabilir. Arka plan aydınlatmasını artırmayı veya pozlamayı düzenlemeyi düşünebilirsiniz.")
            elif brightness > 220:
                st.warning("☀️ **Görsel çok parlak olabilir.**\n\nDetay kaybı yaşanabilir. Parlaklığı azaltmak ya da kontrastı dengelemek faydalı olabilir.")
            else:
                st.success("✅ Parlaklık seviyesi ideal aralıkta.")

            # Netlik (Sharpness)
            if sharpness < 150:
                st.warning("🔍 **Görsel net değil.**\n\nOdak sorunları olabilir veya düşük çözünürlüklü görsel yüklenmiş olabilir. Daha net bir versiyon tercih edin.")
            elif sharpness > 1200:
                st.info("⚠️ **Aşırı netlik algılandı.**\n\nBu, keskinleştirme filtrelerinden kaynaklanıyor olabilir. Görsel doğallığını kaybedebilir.")
            else:
                st.success("✅ Netlik seviyesi iyi.")

            # Renk Canlılığı
            if colorfulness < 20:
                st.info("🎨 **Renkler soluk görünüyor.**\n\nRenk kontrastını artırmak veya daha sıcak tonlar tercih etmek daha dikkat çekici olabilir.")
            elif colorfulness > 60:
                st.warning("🌈 **Renk aşırı canlı olabilir.**\n\nBu durum yapay görünüm yaratabilir. Tonları dengelemekte fayda var.")
            else:
                st.success("✅ Renk canlılığı dengeli.")

            # Kontrast
            if contrast < 30:
                st.info("🌓 **Kontrast düşük.**\n\nKatmanlar veya nesneler yeterince ayrışmıyor olabilir. Hafif kontrast artırımı önerilir.")
            elif contrast > 100:
                st.warning("🌗 **Kontrast çok yüksek.**\n\nBu, detay kaybına ve rahatsız edici bir görünüme neden olabilir.")
            else:
                st.success("✅ Kontrast seviyesi iyi.")

            # Entropi (Bilgi Yoğunluğu)
            if entropy < 4.0:
                st.info("🔀 **Görsel basit yapıda.**\n\nDaha fazla detay eklemek, kullanıcıyı daha çok etkileyebilir.")
            elif entropy > 7.0:
                st.warning("🔀 **Görsel çok karmaşık.**\n\nİzleyicinin odaklanması zorlaşabilir. Odak nesnesini belirginleştirin.")
            else:
                st.success("✅ Görsel dengeli bilgi yoğunluğuna sahip.")

            """
            # En-boy oranı
            if aspect_ratio > 2:
                st.warning("📐 **Görsel çok yatay.**\n\nMobilde görüntüleme sorunları yaşanabilir. Daha dengeli bir oran tercih edilebilir.")
            elif aspect_ratio < 0.5:
                st.warning("📐 **Görsel çok dikey.**\n\nKullanıcı deneyimi açısından yatay oranlar daha etkilidir.")
            else:
                st.success("✅ En-boy oranı kullanıcı dostu.")
            """
            # Kenar yoğunluğu
            if edges < 0.01:
                st.info("🧩 **Detay az.**\n\nGörsel fazla sade olabilir. Ufak dokular ya da arka plan detayları eklenebilir.")
            elif edges > 0.10:
                st.warning("🧩 **Detay fazlalığı.**\n\nGörsel karmaşık görünebilir. Odak noktası belirgin olmalı.")
            else:
                st.success("✅ Kenar yoğunluğu ideal.")