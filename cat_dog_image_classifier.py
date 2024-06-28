import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import keras

st.set_page_config(
    page_title="CAT DOG IMAGE CLASSIFIER",
    page_icon=":dog:",
    layout="wide",
)
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.title("CAT DOG Image Classifer")
st.write("check out the [Model](https://www.kaggle.com/code/kalidasvijaybhak/cat-and-dog-classification)")
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0
img_file_buffer = st.file_uploader('Upload a JPG image', type='jpg', key=st.session_state["file_uploader_key"], accept_multiple_files=False)
target_size = (150, 150)

loaded_model = keras.models.load_model('simple_cnn_cat_dog.h5')

# File uploader and image display on the left
col1, col2 = st.columns([3, 2])

with col1:

    if img_file_buffer is not None:
        img1 = Image.open(img_file_buffer)
        img2 = Image.open(img_file_buffer)
        new_image = img2.resize((600, 400))
        st.image(new_image, caption=img_file_buffer.name)
        img = np.array(img1)
        if len(img.shape) != 3:
            raise ValueError("Invalid image shape")
        img = tf.image.resize(img, target_size)
        img = img / 255
        img_array = img.numpy()

# Buttons on the right
with col2:
    if st.button('Refresh Data'):
        st.session_state["file_uploader_key"] += 1
        st.experimental_rerun()
    
    if st.button('Classify'):
        if img_file_buffer is not None:
            y_pred = loaded_model.predict(img_array.reshape(1, 150, 150, 3))

            output = "dog" if y_pred > 0.5 else "cat"
            # st.header("This is a " + output)
            if y_pred[0] > 0.5:
                output = "dog"
                confidence = np.round(( y_pred[0]) * 100, 2)  # Inverse confidence for dog
            else:
                output = "cat"
                confidence = np.round((1-y_pred[0]) * 100, 2)  # Direct confidence for cat

            # Display the result with calculated confidence
            st.header(f"This is a {output} with {str(confidence).strip('[]')}% confidence.")


        elif img_file_buffer is None:
            st.warning('Upload an image first', icon="⚠️")
 
