# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import google.generativeai as genai

# --- Page Config ---
st.set_page_config(
    page_title="Plant Disease & Fungus Detector",
    page_icon="Leaf",
    layout="centered"
)

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Leaf Plant Disease & Fungus Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a leaf photo → Get disease name + Hindi/English treatment</p>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/emoji/344/leaf.png", width=80)
    st.header("How to Use")
    st.write("1. Take a clear photo of the **leaf** (front side).")
    st.write("2. Upload below.")
    st.write("3. Click Detect Disease .")
    st.info("Supports 38 diseases: Tomato, Apple, Potato, Corn, etc.")

# --- Load Model (Cached) ---
@st.cache_resource
def load_keras_model():
    return load_model('ai_crops_model.keras')

model = load_keras_model()

# --- 38 Class Names ---
class_names = {
    0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy',
    4: 'Blueberry___healthy', 5: 'Cherry_(including_sour)___Powdery_mildew', 6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight', 10: 'Corn_(maize)___healthy', 11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)', 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Peach___Bacterial_spot', 17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy', 20: 'Potato___Early_blight',
    21: 'Potato___Late_blight', 22: 'Potato___healthy', 23: 'Raspberry___healthy', 24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew', 26: 'Strawberry___Leaf_scorch', 27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot', 29: 'Tomato___Early_blight', 30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold', 32: 'Tomato___Septoria_leaf_spot', 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot', 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# --- Gemini Setup ---
@st.cache_resource
def get_gemini_model():
    # FIXED: Use secrets.toml
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel('gemini-2.5-flash')

# --- File Upload (Only One!) ---
uploaded_file = st.file_uploader("Upload Leaf Photo", type=['png', 'jpg', 'jpeg'], key="leaf_uploader")

if uploaded_file:
    # Show image
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption="Your Leaf", use_container_width=True)  # FIXED: use_container_width

    if st.button("Detect Disease & Fungus", type="primary"):
        with st.spinner("Analyzing leaf..."):
            # Preprocess
            img_array = img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)[0]
            conf = np.max(preds) * 100
            idx = np.argmax(preds)
            disease = class_names.get(idx, "Unknown")

            # Gemini AI
            gemini = get_gemini_model()
            if 'healthy' in disease.lower():
                prompt = f"Healthy plant: {disease.replace('___', ' ')}. Give short advice in English + Hindi: 1. Crop care 2. Fertilizer 3. Prevention"
            else:
                prompt = f"Plant disease: {disease.replace('___', ' ')}. Give short cure in English + Hindi: 1. Disease info 2. Fertilizer 3. Treatment & prevention"

            response = gemini.generate_content(prompt)
            advice = response.text

        # --- Results ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Confidence", f"{conf:.1f}%")
            color = "green" if 'healthy' in disease.lower() else "red"
            st.markdown(f"<h2 style='color:{color};'>{disease.replace('___', ' → ')}</h2>", unsafe_allow_html=True)
            st.image(image, width=200)

        with col2:
            st.subheader("Treatment Advice")
            st.markdown(advice)

        st.success("Analysis Complete!")