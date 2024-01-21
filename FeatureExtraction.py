import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np



from google.colab import drive
drive.mount('/content/drive')

data_dir = "/content/drive/MyDrive/FYP - Munz/Dataset"
categories = ["Shoplifting", "Normal"]



img_size = (224, 224)  # Image size for MobileNetV3Large
batch_size = 32

# Load pre-trained MobileNetV3Large (without top layer)
model = MobileNetV3Large(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))



# Add GlobalAveragePooling2D layer to reduce feature dimensionality
model = tf.keras.Sequential([model, GlobalAveragePooling2D()])  # to reduce ram



# Freeze pre-trained layers
# for layer in model.layers:
#     layer.trainable = False




# Function to process videos in smaller chunks
def process_video_chunks(video_path, chunk_size=50): #to reduce ram usage added chunk
    features = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)  # Resize to match model input
        frame = frame / 255.0  # Normalize pixel values
        feature = model.predict(np.expand_dims(frame, axis=0))  # Add batch dimension
        features.append(feature.flatten())  # Flatten feature vector
        frame_count += 1
        if frame_count % chunk_size == 0:
            yield np.concatenate(features)  # Yield accumulated features
            features = []  # Reset features list
    cap.release()
    if features:  # Yield remaining features if any
        yield np.concatenate(features)

# Process videos in a memory-efficient way
all_features = []
all_labels = []
for category in categories:
    category_dir = os.path.join(data_dir, category)
    video_paths = [os.path.join(category_dir, video_file) for video_file in os.listdir(category_dir)]
    for video_path in video_paths:
        for chunk_features in process_video_chunks(video_path):
            all_features.append(chunk_features)
            all_labels.append(category)




# Function to process videos in smaller chunks
def process_video_chunks(video_path, chunk_size=50): #to reduce ram usage added chunk
    features = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)  # Resize to match model input
        frame = frame / 255.0  # Normalize pixel values
        feature = model.predict(np.expand_dims(frame, axis=0))  # Add batch dimension
        features.append(feature.flatten())  # Flatten feature vector
        frame_count += 1
        if frame_count % chunk_size == 0:
            yield np.concatenate(features)  # Yield accumulated features
            features = []  # Reset features list
    cap.release()
    if features:  # Yield remaining features if any
        yield np.concatenate(features)

# Process videos in a memory-efficient way
all_features = []
all_labels = []
for category in categories:
    category_dir = os.path.join(data_dir, category)
    video_paths = [os.path.join(category_dir, video_file) for video_file in os.listdir(category_dir)]
    for video_path in video_paths:
        for chunk_features in process_video_chunks(video_path):
            all_features.append(chunk_features)
            all_labels.append(category)




# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

# Send feature vectors to your Liquid Neural Network model (example using TensorFlow)
# Assuming your LNN model is defined as `lnn_model`
predictions = lnn_model(X_train)  # Example for training data
