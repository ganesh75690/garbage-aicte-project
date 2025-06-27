import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import gradio as gr
import pickle

# Load the trained Keras model
model = load_model("MobileNetV3_GarbageClassifier_FIXED.keras")

# Load class names
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# Define prediction function
def predict_image(img):
    # Resize to match training input size
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32)

    # Apply MobileNetV3 preprocessing
    img_array = preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = round(100 * float(np.max(predictions[0])), 2)

    return f"{predicted_label} ({confidence}%)"

# Gradio interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Garbage Classifier (MobileNetV3Large)",
    description="Upload an image of garbage to classify it into one of 6 categories: cardboard, glass, metal, paper, plastic, or trash."
)

# Run the app
interface.launch()

