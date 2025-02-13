import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("model.h5")

st.title("Handwritten digit recognition:")
st.write("Upload a digit image and the model will predict the number")

uploaded_file = st.file_uploader("Choose an image", type = ["jpg", "png", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    image_arr = np.array(image)/255.0
    image_arr = image_arr.reshape(1, 28, 28, 1)

    st.image(image, caption="Uploaded image",  use_container_width=True)


    prediction = model.predict(image_arr)
    predicted_digit = np.argmax(prediction)

    st.write(f"Predicted digit: {predicted_digit}")