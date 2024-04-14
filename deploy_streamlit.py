import streamlit as st
import numpy as np
import cv2
import io
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model('image_classifier_model.h5')

def load_and_preprocess_image(file_upload, size=(128, 128)):
    image = Image.open(io.BytesIO(file_upload.getvalue()))
    image = np.array(image.convert('RGB'))
    image = cv2.resize(image, size)
    image = image / 255.0
    return image

def main():
    st.set_page_config(page_title="Image Classifier", page_icon=":guardsman:", layout="wide")

    st.title("Image Classifier")

    file_upload = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

    if file_upload is not None:
        image = load_and_preprocess_image(file_upload)
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)

        if predictions[0][0] > 0.5:
            prediction_text = "Biodegradable"
        else:
            prediction_text = "Non-Biodegradable"

        st.write("**Prediction:**", prediction_text)

    st.write("**Made with Streamlit and TensorFlow**")

if __name__ == "__main__":
    main()
    
    