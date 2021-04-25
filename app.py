import streamlit as st
import numpy as np
import pandas as pd

import streamlit as st 
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def process_image(image):

#   image = tf.image.decode_image(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size=[224, 224])

  st.write(image.shape)

  return image


BATCH_SIZE = 1

def create_data_batches(x, batch_size=BATCH_SIZE):
  data = np.array([process_image(x)])
  data = tf.data.Dataset.from_tensor_slices(data)
  data_batch = data.batch(BATCH_SIZE)

  st.write(data_batch)

  return data_batch


st.title("Wildlife Conservation")
st.subheader("Classifying wildlife from humans")

safe_html="""  
    <div style="background-color:#F4D03F;padding:10px >
    <h2 style="color:white;text-align:center;"> Animal Detected</h2>
    </div>
"""
warning_html="""  
    <div style="background-color:#F08080;padding:10px >
    <h2 style="color:black ;text-align:center;"> Predator detected</h2>
    </div>
"""
model = tf.keras.models.load_model('mobilenet.h5', custom_objects={"KerasLayer": hub.KerasLayer})

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Predict:"):
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.asarray(image).astype('float32')
        prep_image = create_data_batches(image)
        pred = model.predict(prep_image)

        class_names = ['bald_eagle' 'black_bear' 'bobcat' 'canada_lynx'
 'columbian_black-tailed_deer' 'cougar' 'coyote' 'deer' 'elk' 'gray_fox'
 'gray_wolf' 'mountain_beaver' 'nutria' 'raccoon' 'raven' 'red_fox'
 'ringtail' 'sea_lions' 'seals' 'virginia_opossum']

        st.write(pred)
        st.write(np.argmax(pred, axis=1))
        

    





