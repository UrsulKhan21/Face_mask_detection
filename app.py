import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


MODEL_PATH = "models/face_mask_model.h5"
IMG_SIZE = 128


st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 1040px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.6rem 1.8rem;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #ffffff;
            margin-bottom: 1.4rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.25rem;
            line-height: 1.15;
            color: #111827;
        }

        .hero p {
            margin: 0.55rem 0 0;
            color: #4b5563;
            font-size: 1rem;
        }

        .result-box {
            padding: 1rem 1.15rem;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #f9fafb;
            margin-top: 1rem;
        }

        .metric-label {
            color: #6b7280;
            font-size: 0.9rem;
            margin-bottom: 0.15rem;
        }

        .metric-value {
            color: #111827;
            font-size: 1.4rem;
            font-weight: 700;
        }

        .small-note {
            color: #6b7280;
            font-size: 0.92rem;
        }

        div[data-testid="stCameraInput"] {
            border: 1px dashed #cbd5e1;
            border-radius: 8px;
            padding: 0.75rem;
            background: #f8fafc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        label = "With Mask"
        confidence = 1 - prediction
        st.success("Mask detected")
    else:
        label = "Without Mask"
        confidence = prediction
        st.error("No mask detected")

    st.markdown(
        f"""
        <div class="result-box">
            <div class="metric-label">Prediction</div>
            <div class="metric-value">{label}</div>
            <div class="small-note">Confidence: {confidence:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <h1>Face Mask Detection</h1>
        <p>Upload an image or take a browser webcam photo to check whether a face mask is present.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.subheader("Choose Input")
    upload_tab, camera_tab = st.tabs(["Upload Image", "Webcam Photo"])

    uploaded_file = None
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Select a JPG or PNG image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

    with camera_tab:
        st.caption("Use this from the browser opened by Streamlit. Allow camera permission when asked.")
        uploaded_file = st.camera_input("Take a webcam photo", label_visibility="collapsed")

with right_col:
    st.subheader("Preview & Result")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    with right_col:
        st.image(image, caption="Selected image", use_container_width=True)

        try:
            show_prediction(image)
        except OSError:
            st.warning("Model file not found. Train the model first with: python train.py")
else:
    with right_col:
        st.info("Your selected or captured image will appear here.")

with st.expander("Webcam not opening?"):
    st.write(
        "Run the app with `streamlit run app.py`, open the local browser URL, "
        "choose the webcam tab, and allow camera permission."
    )
    st.write(
        "Also check Windows Camera privacy settings and close apps like Zoom, Teams, "
        "or the Camera app if they are already using the webcam."
    )
