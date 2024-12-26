import os

from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.sidebar.title("HIBA & NOUR GLSI-5-B")

model = load_model('flower_classifier.h5')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image) / 255.0
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])

    class_index = np.argmax(result)
    class_probability = result[class_index] * 100

    print("Probabilités de toutes les classes :")
    for i, prob in enumerate(result):
        print(f"{flower_names[i]}: {prob * 100:.2f}%")

    outcome = f"The Image belongs to {flower_names[np.argmax(result)]} with a probability of {class_probability:.2f}% "
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width = 200)

    st.markdown(classify_images(uploaded_file))

