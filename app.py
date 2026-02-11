import streamlit as st
from model_helper import predict
from PIL import Image
import os

st.set_page_config(page_title="Vehicle Damage Detection", layout="centered")

st.title("🚗 Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temporarily
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    with st.spinner("Analyzing image..."):
        prediction = predict(temp_path)

    st.success(f"Predicted Class: {prediction}")

    # Optional cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
