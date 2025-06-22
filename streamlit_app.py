
import streamlit as st
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import gc
import cv2
from sklearn.cluster import KMeans

st.set_page_config(page_title="💄 Nuanță Ruj - Modelul Tău", layout="centered")
st.title("💋 Detectare Nuanță Ruj cu Modelul Tău Roboflow")

@st.cache_resource
def load_lipstick_db():
    df = pd.read_csv("avon_lipsticks.csv")
    df["rgb"] = df["hex"].apply(lambda x: tuple(int(x.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)))
    return df

def get_dominant_color(pixels):
    kmeans = KMeans(n_clusters=1, random_state=0).fit(pixels)
    return tuple(map(int, kmeans.cluster_centers_[0]))

def classify_rgb(rgb):
    h, s, v = list(cv2.cvtColor(np.uint8([[list(rgb)]]), cv2.COLOR_RGB2HSV)[0][0])
    if s < 64 and v > 200:
        return "nude"
    elif h < 10 or h > 170:
        return "roșu"
    elif 10 <= h <= 25:
        return "corai"
    elif 26 <= h <= 35:
        return "piersică"
    elif 36 <= h <= 70:
        return "auriu / galben"
    elif 71 <= h <= 160:
        return "mov / prună"
    elif 161 <= h <= 169:
        return "roz / fucsia"
    else:
        return "neclasificat"

def find_closest_color(rgb, df):
    return df.iloc[((df["rgb"].apply(lambda ref: np.linalg.norm(np.array(rgb) - np.array(ref))))).idxmin()]

def segment_lips_with_custom_roboflow(image, api_key, model_url):
    image.save("/tmp/temp_image.jpg")
    with open("/tmp/temp_image.jpg", "rb") as image_file:
        response = requests.post(
            model_url,
            files={"file": image_file},
            headers={"Authorization": f"Bearer {api_key}"}
        )
    if response.status_code == 200:
        mask_url = response.json()["predictions"][0].get("mask")
        if mask_url:
            return Image.open(BytesIO(requests.get(mask_url).content)).convert("RGB")
    return None

df_lipsticks = load_lipstick_db()
MODEL_URL = "https://infer.roboflow.com/lips-detection-8jbnb/1"
API_KEY = st.secrets["exqyo8vHkgd72GHfEtOa"]

uploaded_files = st.file_uploader("📸 Încarcă imagini pentru analiză", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files and API_KEY:
    for i, img_file in enumerate(uploaded_files):
        with st.expander(f"📷 Imagine #{i+1}: {img_file.name}", expanded=True):
            img = Image.open(img_file).convert("RGB").resize((512, 512))
            st.image(img, caption="Imagine originală", use_container_width=True)

            mask = segment_lips_with_custom_roboflow(img, API_KEY, MODEL_URL)

            if mask:
                mask_np = np.array(mask)
                lips_pixels = mask_np.reshape(-1, 3)
                lips_pixels = lips_pixels[(lips_pixels != [0, 0, 0]).any(axis=1)]

                if len(lips_pixels) > 0:
                    dominant = get_dominant_color(lips_pixels)
                    nuanta_text = classify_rgb(dominant)
                    ruj = find_closest_color(dominant, df_lipsticks)

                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.image(np.full((60, 60, 3), dominant, dtype=np.uint8), caption="Culoare", width=70)
                    with col2:
                        st.markdown(f"🎯 **Nuanță detectată**: `{nuanta_text}`")
                        st.markdown(f"💄 **Ruj sugerat**: `{ruj['name']}` ({ruj['label']})")
                else:
                    st.warning("⚠️ Nu s-au detectat pixeli colorați.")
            else:
                st.error("❌ Roboflow nu a returnat mască validă.")

            del img, mask
            gc.collect()
