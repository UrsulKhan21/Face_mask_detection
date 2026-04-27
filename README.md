# Face Mask Detection using CNN and Streamlit

This project detects whether a person is wearing a face mask or not using a CNN model.

## Features

- Train a CNN model on an image dataset
- Predict mask / no mask from an image path
- Upload an image in a Streamlit app
- Take a webcam photo in the Streamlit app
- Save and load a trained Keras model

## Project Structure

```text
face-mask-detection/
|-- dataset/
|   |-- with_mask/
|   `-- without_mask/
|-- models/
|   `-- face_mask_model.h5
|-- app.py
|-- train.py
|-- predict.py
|-- requirements.txt
|-- labels.txt
|-- README.md
`-- .gitignore
```

## Dataset

Place images in this structure:

```text
dataset/
|-- with_mask/
|   |-- img1.jpg
|   `-- img2.jpg
`-- without_mask/
    |-- img1.jpg
    `-- img2.jpg
```

You can use a public face-mask dataset from Kaggle or your own collected images.

## Install

TensorFlow may not install on the newest Python versions. Python 3.11 or 3.12 is recommended.

On this system, use the included Python 3.11 setup script:

```powershell
.\setup_python311.ps1
```

If PowerShell blocks scripts, run this once in the project folder:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_python311.ps1
```

Manual setup:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train Model

```bash
python train.py
```

After training, the model is saved at:

```text
models/face_mask_model.h5
```

## Run Prediction

```bash
python predict.py dataset/with_mask/sample1.jpg
```

Or run it without an argument and enter the image path when asked:

```bash
python predict.py
```

## Run Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## CNN Architecture

- Input image: 128 x 128 x 3
- Conv2D + ReLU
- MaxPooling
- Conv2D + ReLU
- MaxPooling
- Conv2D + ReLU
- MaxPooling
- Flatten
- Dense + ReLU
- Dropout
- Dense + Sigmoid

## Output

- `0 = with_mask`
- `1 = without_mask`

## Accuracy Tips

- Use more images
- Keep both classes balanced
- Increase epochs
- Use data augmentation
- Try transfer learning with MobileNetV2
