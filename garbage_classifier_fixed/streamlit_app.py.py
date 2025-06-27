import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Load model and class names
model = tf.keras.models.load_model("model.keras")
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# Image preprocessing
def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app layout
st.title("♻️ Garbage Classifier")
st.write("Upload an image of garbage to classify it into one of the 6 categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.success(f"Prediction: **{predicted_class}** ({confidence * 100:.2f}%)")

            st.subheader("Class Probabilities:")
            for i, prob in enumerate(prediction):
                st.write(f"{class_names[i]}: {prob*100:.2f}%")
