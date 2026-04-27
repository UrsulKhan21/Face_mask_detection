import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


MODEL_PATH = "models/face_mask_model.h5"
IMG_SIZE = 128


def load_mask_model():
    return load_model(MODEL_PATH)


def predict_mask(img_path, model=None):
    if model is None:
        model = load_mask_model()

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "With Mask" if prediction < 0.5 else "Without Mask"
    confidence = 1 - prediction if prediction < 0.5 else prediction
    return label, float(confidence)


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ")
    result, score = predict_mask(image_path)
    print(f"Prediction: {result}")
    print(f"Confidence: {score:.2%}")
