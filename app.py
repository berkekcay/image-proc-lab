import streamlit as st
import joblib
import numpy as np
import cv2
from extract_features import calculate_brightness, calculate_sharpness, dominant_colors

# Eğitilmiş modeli yükle
model = joblib.load("data/engagement_model.pkl")

# Streamlit başlıkları
st.title("📸 AI Sosyal Medya İçerik Optimizasyonu")
st.subheader("AI ile görsellerinizi analiz edin ve en iyi etkileşim için öneriler alın!")

# Görsel yükleme alanı
uploaded_image = st.file_uploader("📤 Bir görsel yükleyin", type=["jpg", "png", "jpeg"])

if uploaded_image and st.button("🔍 Analiz Başlat"):
    # Yüklenen dosyayı kaydet
    image_path = "temp.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    
    # Görseli oku ve küçült
    image = cv2.imread(image_path)
    image = cv2.resize(image, (200, 200))

    # Özellikleri çıkar
    brightness = calculate_brightness(image)
    sharpness = calculate_sharpness(image)
    dominant = dominant_colors(image)

    # Model için özellik vektörü oluştur
    features = np.array([[brightness, sharpness, dominant[2], dominant[1], dominant[0]]])
    predicted_likes = model.predict(features)[0]

    # Görseli göster
    st.image("temp.jpg", caption="Yüklenen Görsel", use_container_width=True)

    # Sonuçları göster
    st.subheader("📊 Analiz Sonuçları")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌟 Parlaklık", f"{brightness:.2f}")
    col2.metric("🔍 Netlik", f"{sharpness:.2f}")
    col3.metric("❤️ Tahmini Beğeni", f"{int(predicted_likes)}")

    # 🎨 Dominant Renk
    st.subheader("🎨 Dominant Renk")
    st.write(f"RGB({int(dominant[2])}, {int(dominant[1])}, {int(dominant[0])})")

    # 💡 **Optimizasyon Önerileri**
    st.subheader("💡 Optimizasyon Önerileri")

    if brightness < 100:
        st.warning("📢 Görseliniz biraz karanlık. Daha parlak bir resim kullanabilirsiniz!")
    elif brightness > 200:
        st.warning("📢 Görsel çok parlak olabilir. Kontrastı biraz düşürebilirsiniz.")

    if sharpness < 200:
        st.warning("📢 Görseliniz net değil! Daha yüksek çözünürlüklü bir görsel kullanabilirsiniz.")
    elif sharpness > 1000:
        st.warning("📢 Görsel fazla keskin olabilir. Doğal görünüme dikkat edin.")

    if predicted_likes < 500:
        st.info("📌 Görselinizin tahmini etkileşimi düşük, daha renkli veya net görseller tercih edin.")
    else:
        st.success("🎉 Görseliniz yüksek etkileşim alabilir! Harika seçim!")