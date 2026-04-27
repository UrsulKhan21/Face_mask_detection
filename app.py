import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


MODEL_PATH = "models/face_mask_model.h5"
IMG_SIZE = 128


st.set_page_config(page_title="Face Mask Detection", layout="centered")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def show_prediction(img):
    model = get_model()
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img, verbose=0)[0][0]

    if prediction < 0.5:
        st.success(f"Prediction: With Mask ({(1 - prediction):.2%} confidence)")
    else:
        st.error(f"Prediction: Without Mask ({prediction:.2%} confidence)")


st.title("Face Mask Detection")
st.write("Upload a face image or take a webcam photo to check whether the person is wearing a mask.")

input_mode = st.radio("Choose input type", ["Upload Image", "Webcam Photo"], horizontal=True)

uploaded_file = None
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a webcam photo")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Selected Image", use_container_width=True)

    try:
        show_prediction(image)
    except OSError:
        st.warning("Model file not found. Train the model first with: python train.py")
