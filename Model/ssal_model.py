# -*- coding: utf-8 -*-
"""ConvLSTM-semi supervised_PSPD.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J6yMmajYmFXHqYGIaIjgdsjCw-Xj4j2j

1. Import Necessary Libraries:
"""

import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from google.colab import drive
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

"""2. Mount Google Drive:"""

drive.mount('/content/drive')

"""3. Define Data Paths:"""

data_dir = '/content/drive/MyDrive/FYP - Munz/Dataset'
normal_videos_dir = os.path.join(data_dir, 'Normal')
shoplifting_videos_dir = os.path.join(data_dir, 'Shoplifting')

"""4. Load and Preprocess Videos:"""

# Function to get paths for labeled and unlabeled videos
def get_video_paths(directory, labeled=True):
    video_paths_labels = []
    video_counter = 0
    for root, _, files in os.walk(directory):
          for file in files:
              if file.endswith(".mp4"):
                video_counter +=1
                if (labeled == True and video_counter <= 75) or (labeled == False and video_counter >75) :
                  video_path = os.path.join(root, file)
                  label = 1 if "Shoplifting" in os.path.basename(root) else 0 if labeled else -1
                  video_paths_labels.append((video_path, label))
    return video_paths_labels


def load_and_preprocess_videos(video_paths_labels):
    """Loads videos, extracts and preprocesses frames, and combines them into sequences.

    Args:
        video_paths (list): List of paths to video files.

    Returns:
        tuple: (video_sequences, labels)
            video_sequences (np.ndarray): 4D array of shape (num_videos, seq_length, height, width, channels)
            labels (np.ndarray): 1D array of labels for each video
    """

    video_sequences = []
    video_labels = []
    for video_path, label in video_paths_labels:
        # Load video using OpenCV
        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame (e.g., resize, convert to grayscale)
            frame = cv2.resize(frame, (128, 128))  # Adjust frame size
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

            frames.append(frame)

        cap.release()

        # Combine frames into sequences of specified length
        seq_length = 10  # Adjust sequence length
        video_sequences.append(np.array(frames[:seq_length]))  # Append frame sequences
        video_labels.append(label)

    # Normalize pixel values to [0, 1]
    video_sequences = np.array(video_sequences).astype('float32') / 255.0

    video_sequences = np.expand_dims(video_sequences, axis=-1)  # Add channel dimension

    flipped_video_sequences = []
    flipped_video_labels = []
    # Flip frames for augmentation
    for i in range(len(video_sequences)):
        flipped_frames = []
        for frame in video_sequences[i]:
            flipped_frame = cv2.flip(frame, 1)  # Flip horizontally
            flipped_frames.append(flipped_frame)

        flipped_frames = np.expand_dims(flipped_frames, axis=-1)  # Add channel dimension
        flipped_video_sequences.append(np.array(flipped_frames))
        flipped_video_labels.append(video_labels[i])  # Assign same label to flipped video

    video_sequences = np.concatenate((video_sequences, flipped_video_sequences), axis=0)
    video_labels = np.concatenate((video_labels, flipped_video_labels), axis=0)
    return np.array(video_sequences), np.array(video_labels)


normal_videos, normal_labels = load_and_preprocess_videos(get_video_paths(normal_videos_dir))
shoplifting_videos, shoplifting_labels = load_and_preprocess_videos(get_video_paths(shoplifting_videos_dir))

unlabeled_videos1, _ = load_and_preprocess_videos(get_video_paths(normal_videos_dir, labeled=False))
unlabeled_videos2, __ = load_and_preprocess_videos(get_video_paths(shoplifting_videos_dir, labeled=False))

"""5. Combine and Split Data:"""

print(len(normal_videos), len(shoplifting_videos), len(normal_labels), len(shoplifting_labels), len(unlabeled_videos1) , len(unlabeled_videos2))

# Concatenate labeled data
X_labeled = np.concatenate((normal_videos, shoplifting_videos))
y_labeled = np.concatenate((normal_labels, shoplifting_labels))

# Split labeled data for training and testing
X_train_labeled, X_test_labeled, y_train_labeled, y_test_labeled = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

# Create a validation set from the test set
X_val, X_test_labeled, y_val, y_test_labeled = train_test_split(X_test_labeled, y_test_labeled, test_size=0.5, random_state=42)

# print(y_labeled.value_counts)
import pandas as pd

y_labeled_series = pd.Series(y_labeled)  # Convert NumPy array to pandas Series
print(y_labeled_series.value_counts())

X_unlabeled = np.concatenate((unlabeled_videos1, unlabeled_videos2))

# Combine labeled and unlabeled data
X_train_combined = np.concatenate((X_train_labeled, X_unlabeled))
y_train_combined = np.concatenate((y_train_labeled, -1 * np.ones(len(X_unlabeled))))  # Use -1 for unlabeled samples

"""6. Design LSTM Architecture:"""

from tensorflow.keras.layers import Conv2D, TimeDistributed, MaxPooling2D, Flatten, LSTM, Dropout, Dense

model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(10, 128, 128, 1)),  # Adjust frame shape
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    # Add auxiliary output for anomaly detection
    # Dense(1, activation='sigmoid', name='auxiliary_output'),
    # Attension(...),
    Dense(1, activation='sigmoid',  kernel_regularizer=tf.keras.regularizers.l2(0.01))  # Main shoplifter detection output ## Add regularization to layers
])

"""7. Compile the Model:"""

#function to calculate comnined loss
# def combined_loss(y_true, y_pred):
#     main_loss = mean_squared_error(y_true[:, 0], y_pred[:, 0])
#     auxiliary_loss = binary_crossentropy(y_true[:, 1], y_pred[:, 1])
#     return main_loss + 0.2 * auxiliary_loss  # Adjust weight as needed

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss=binary_crossentropy, optimizer='adam', metrics=['accuracy'])

"""8. Train the Model:"""

# from uncertainty_metrics import uncertainty_sampling
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

epochs = 30

# Store training and validation loss during each epoch
train_loss = []
val_loss = []

for epoch in range(epochs):
    print(f" * * * * * * * * * epoch {epoch} * * * * * * * * * ")

    # Initial training on labeled data
    history = model.fit(X_train_labeled, y_train_labeled, epochs=1, batch_size=32)
    train_loss.append(history.history['loss'][0])  # Extract loss from history

    # Evaluate on validation data (if available)
    if X_val is not None and y_val is not None:
        val_loss_epoch = model.evaluate(X_val, y_val, verbose=0)[0]  # Get validation loss
        val_loss.append(val_loss_epoch)

    # Generate pseudo-labels for unlabeled data
    unlabeled_predictions = model.predict(X_unlabeled)
    pseudo_labels = np.where(unlabeled_predictions > 0.85, 1, 0)  # Set threshold

    # Retrain with combined data and pseudo-labels
    y_train_combined = np.concatenate((y_train_labeled, pseudo_labels.flatten()))
    model.fit(X_train_combined, y_train_combined, epochs=1, batch_size=32,
              validation_data=(X_val, y_val),
              callbacks=[EarlyStopping(patience=3)])

# Plot training and validation loss
plt.plot(train_loss, label='Training Loss')
if len(val_loss) > 0:
    plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

from scipy.stats import entropy
import matplotlib.pyplot as plt


# Active Learning
def calculate_uncertainty(predictions):
    # Calculate entropy of predictions
    uncertainty = entropy(predictions.T)
    return uncertainty


#Implementation of Active Learning Approach

# # Active learning loop
# num_iterations = 30  # Example: number of active learning iterations
# val_loss = []  # List to store validation loss
# train_loss = []  # List to store training loss
# mean_uncertainty = []  # List to store mean uncertainty
#
# for iteration in range(num_iterations):
#     print(f"Iteration {iteration + 1}")
#
#     # Ensure X_train_combined and y_train_combined have the same number of samples
#     print("len(X_train_combined) = ", len(X_train_combined), " len(y_train_combined) = ", len(y_train_combined))
#     # assert len(X_train_combined) == len(y_train_combined), "Number of samples in X_train_combined and y_train_combined must match"
#
#     # Train the model
#     history = model.fit(X_train_combined, y_train_combined, epochs=1, batch_size=32)
#     train_loss.append(history.history['loss'][0])
#
#     # Manually evaluate on validation set
#     if X_val is not None and y_val is not None:
#         val_loss_epoch = model.evaluate(X_val, y_val, verbose=0)[0]  # Get validation loss
#         val_loss.append(val_loss_epoch)
#
#     # Predict on unlabeled data
#     unlabeled_predictions = model.predict(X_unlabeled)
#
#     # Calculate uncertainty
#     uncertainty = calculate_uncertainty(unlabeled_predictions)
#     mean_uncertainty.append(np.mean(uncertainty))
#
#     # Select top uncertain samples
#     num_samples_to_label = 10  # Example: number of samples to label in each iteration
#     top_uncertain_indices = np.argsort(uncertainty)[-num_samples_to_label:] % len(simulated_labels)
#     samples_to_label = X_unlabeled[top_uncertain_indices]
#
#     # Query labels for selected samples (simulated or from an oracle)
#     simulated_labels = np.random.randint(0, 2, size=num_samples_to_label)  # Simulated labels for demonstration
#
#     # Update labeled dataset
#     X_train_combined = np.concatenate((X_train_combined, samples_to_label))
#     y_train_combined = np.concatenate((y_train_combined, simulated_labels))
#
#     # Remove labeled samples from the unlabeled dataset
#     X_unlabeled = np.delete(X_unlabeled, top_uncertain_indices, axis=0)
#     # y_train_combined = np.delete(y_train_combined, top_uncertain_indices, axis=0)  # Remove corresponding labels
#
#     simulated_labels = np.delete(simulated_labels, top_uncertain_indices, axis=0)
#
# # Plot training and validation loss
# plt.plot(train_loss, label='Training Loss')
# if len(val_loss) > 0:
#     plt.plot(val_loss, label='Validation Loss')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss over Iterations')
# plt.legend()
# plt.show()
#
# # Plot mean uncertainty
# plt.plot(mean_uncertainty, label='Mean Uncertainty')
# plt.xlabel('Iteration')
# plt.ylabel('Mean Uncertainty')
# plt.title('Mean Uncertainty over Iterations')
# plt.legend()
# plt.show()


"""9. Evaluate Performance:"""

# Evaluate performance on a separate validation set of labeled videos
model.evaluate(X_val, y_val)

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

y_pred = model.predict(X_test_labeled)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Calculate precision, recall, f1, and support for each class
precision, recall, f1, support = precision_recall_fscore_support(y_test_labeled, y_pred_binary)

# Print metrics for each class separately
for i in range(len(precision)):
    print(f"Class {i}:")
    print(f"Precision: {precision[i]:.4f}")
    print(f"Recall: {recall[i]:.4f}")
    print(f"F1 score: {f1[i]:.4f}")
    print(f"Support: {support[i]}")

# Calculate AUC-ROC (which is a single value)
auc = roc_auc_score(y_test_labeled, y_pred)
print(f"AUC-ROC: {auc:.4f}")

loss, accuracy = model.evaluate(X_test_labeled, y_test_labeled)
print("Test accuracy:", accuracy)

"""10. Save Model:"""

model.save('/content/drive/MyDrive/FYP - Munz/Model/model.h5')

"""11. Load the Trained Model:"""

model = tf.keras.models.load_model('/content/drive/MyDrive/FYP - Munz/Model/model.h5')

"""12. Prepare a New Video:"""

def preprocess_video(video_paths):
    """Loads videos, extracts and preprocesses frames, and combines them into sequences.

    Args:
        video_paths (list): List of paths to video files.

    Returns:
        tuple: (video_sequences, labels)
            video_sequences (np.ndarray): 4D array of shape (num_videos, seq_length, height, width, channels)
    """

    video_sequences = []
    # for video_path in (video_paths):
        # Load video using OpenCV
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame (e.g., resize, convert to grayscale)
        frame = cv2.resize(frame, (128, 128))  # Adjust frame size
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        frames.append(frame)

    cap.release()

    # Combine frames into sequences of specified length
    seq_length = 10  # Adjust sequence length
    video_sequences.append(np.array(frames[:seq_length]))

    # Normalize pixel values to [0, 1]
    video_sequences = np.array(video_sequences).astype('float32') / 255.0

    video_sequences = np.expand_dims(video_sequences, axis=-1)  # Add channel dimension

    return video_sequences


video_path = '/content/drive/MyDrive/FYP - Munz/Dataset/Test/shoplift_test_video.mp4'
# print("test_video_sequences.shape before reshape:", test_video_sequences.shape)
test_video_sequences = preprocess_video(video_path)
# test_video_sequences = test_video_sequences.reshape((-1, 10, 128, 128, 1))

# print("test_video_sequences.shape after reshape:", test_video_sequences.shape)  # Confirm shape

print(type(test_video_sequences))
print(test_video_sequences)

"""13. Make Predictions:"""

predictions = model.predict(test_video_sequences)
print(predictions)

"""14. Visualize Predictions:"""

import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

def visualize_predictions(video_path, predictions, threshold=0.5,desired_fps=10):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    seq_length = 20

    # Calculate the delay between frames based on desired FPS
    delay = int(1000 / desired_fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # print(predictions)
        if len(predictions) == 1:  # Check for single prediction
            prediction = predictions[0][0]  # Access directly
        else:
            prediction = predictions[frame_index // seq_length][0]  # Access for multiple predictions

        if prediction > threshold:
            cv2.putText(frame, "Shoplifting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2_imshow(frame)
        if cv2.waitKey(delay) == ord('q'):
            break
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

visualize_predictions(video_path, predictions, desired_fps=15)

def preprocess_video(video_paths):
    """Loads videos, extracts and preprocesses frames, and combines them into sequences.

    Args:
        video_paths (list): List of paths to video files.

    Returns:
        tuple: (video_sequences, labels)
            video_sequences (np.ndarray): 4D array of shape (num_videos, seq_length, height, width, channels)
    """

    video_sequences = []
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame (e.g., resize, convert to grayscale)
        frame = cv2.resize(frame, (128, 128))  # Adjust frame size
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        frames.append(frame)

    cap.release()

    # Combine frames into sequences of specified length
    seq_length = 10  # Adjust sequence length
    video_sequences.append(np.array(frames[:seq_length]))

    # Normalize pixel values to [0, 1]
    video_sequences = np.array(video_sequences).astype('float32') / 255.0

    video_sequences = np.expand_dims(video_sequences, axis=-1)  # Add channel dimension

    return video_sequences


video_path = '/content/drive/MyDrive/FYP - Munz/Dataset/Test/normal_test_video.mp4'
# print("test_video_sequences.shape before reshape:", test_video_sequences.shape)
test_video_sequences = preprocess_video(video_path)
# test_video_sequences = test_video_sequences.reshape((-1, 10, 128, 128, 1))

# print("test_video_sequences.shape after reshape:", test_video_sequences.shape)  # Confirm shape

predictions = model.predict(test_video_sequences)
print(predictions)

import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

def visualize_predictions(video_path, predictions, threshold=0.5,desired_fps=10):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    seq_length = 10
    delay = int(1000 / desired_fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(predictions) == 1:  # Check for single prediction
            prediction = predictions[0][0]  # Access directly
        else:
            prediction = predictions[frame_index // seq_length][0]  # Access for multiple predictions

        if prediction > threshold:
            cv2.putText(frame, "Shoplifting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2_imshow(frame)
        if cv2.waitKey(1) == ord('q'):
            break
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

visualize_predictions(video_path, predictions, desired_fps=15)
