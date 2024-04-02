import streamlit as st
import tensorflow as tf

tf.config.run_functions_eagerly(True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('C:/Users/Munaza/OneDrive - University of Westminster/Desktop/New folder/FYP/Dataset/Model/ssal_model_x_5_1.h5', compile=False)
    return model
