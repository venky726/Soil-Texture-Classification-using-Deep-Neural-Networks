import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile

# Load the trained model
model = load_model("soil_classifier_vgg19.h5")

# Class labels
class_names = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']

# Gabor filter function
def apply_gabor(img):
    gabor_kernels = []
    ksize = 31
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_kernels.append(kernel)
    filtered_imgs = [cv2.filter2D(img, cv2.CV_8UC3, k) for k in gabor_kernels]
    return np.mean(filtered_imgs, axis=0).astype(np.uint8)

# Prediction function
def predict_soil_image(image_path):
    IMG_SIZE = 224
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = apply_gabor(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return class_names[np.argmax(pred)]

# Streamlit UI
st.title("Soil Texture Classification Using Deep Neural Networks")
st.write("Upload a soil image to classify its type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    # Predict and display
    prediction = predict_soil_image(file_path)
    st.image(uploaded_file, caption=f"Predicted Class: {prediction}", use_container_width=True)
    st.success(f"Predicted Soil Type: {prediction}")