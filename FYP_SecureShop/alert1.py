import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tempfile
import os
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import smtplib

# Load the trained model (outside the main loop)
@st.cache_resource  # Cache the loaded model for efficiency
def load_my_model():
    model = tf.keras.models.load_model('C:/Users/Munaza/OneDrive - University of Westminster/Desktop/New folder/FYP/Dataset/Model/ssal_model_x_5_1.h5', compile=False)
    # model = tf.keras.models.load_model(
    #     'C:/Users/Munaza/OneDrive - University of Westminster/Desktop/New folder/FYP/Dataset/Model/ssal_model_x_5_1.h5')
    return model

model = load_my_model()  # Load the model once at the start


def send_email_with_attachment(subject, body, recipient_email, attachment_path):
    sender_email = "your_email@example.com"
    sender_password = "your_password"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach the video file
    with open(attachment_path, "rb") as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % os.path.basename(attachment_path))
        msg.attach(part)

    # Log in to server and send the email
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(msg)
    server.quit()

# Function to save annotated videos with unique names
def save_annotated_video(video_path, predictions, threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"annotated_video_{timestamp_str}.mp4"
    output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    seq_length = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prediction = predictions[min(frame_index // seq_length, len(predictions) - 1)][0]

        text = "Shoplifting" if prediction > threshold else "Normal"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if text == "Shoplifting" else (0, 255, 0), 2)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()

    # Ensure H.264 encoding
    converted_path = output_path.replace(".mp4", "_converted.mp4")
    os.system(f"ffmpeg -i {output_path} -c:v libx264 -crf 23 -y {converted_path}")

    # Delete the original annotated video file
    os.remove(output_path)

    # Create a copy named "display.mp4" outside the annotated directory
    # display_video_path = os.path.join("..", "display.mp4")
    # print("Absolute display video path:", os.path.abspath(display_video_path))
    # os.system(f"cp {converted_path} {display_video_path}")

    # Create a copy named "display.mp4" outside the annotated directory
    display_video_path = "display.mp4"  # Directly in the project folder, or specify another path
    print("Absolute display video path:", os.path.abspath(display_video_path))
    os.system( f"copy /Y {converted_path} {display_video_path}")  # Use copy /Y for Windows to overwrite without prompting

    return converted_path, display_video_path

# Preprocessing function
def preprocess_video(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    video_sequences = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps)

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        frame = cv2.resize(frame, (128, 128))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        frame_count += 1
    cap.release()

    seq_length = 10
    for i in range(0, len(frames), seq_length):
        sequence = frames[i:i + seq_length]
        if len(sequence) == seq_length:
            sequence = np.array(sequence).astype('float32') / 255.0
            sequence = np.expand_dims(sequence, axis=-1)
            video_sequences.append(sequence)

    return np.array(video_sequences)

# Main upload and processing function
def upload_video():
    OUTPUT_DIRECTORY = "annotated_videos"
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])
    if uploaded_file is not None:
        with st.spinner("Preprocessing video..."):
            try:
                test_video_sequences = preprocess_video(uploaded_file)
                predictions = model.predict(test_video_sequences)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    video_path = tmpfile.name

                # Save the annotated video and get the path to the converted video
                converted_path, display_video_path = save_annotated_video(video_path, predictions)

                print(display_video_path)

                shoplifting_detected = any(pred[0] > 0.7 for pred in predictions)  # Assuming binary classification [0, 1]

                if shoplifting_detected:
                    # Send an email alert with the annotated video attached
                    subject = "Shoplifting Detected"
                    body = "Shoplifting activity has been detected. Please find the attached video clip for review."
                    recipient_email = "recipient@example.com"  # Replace with actual recipient email address
                    send_email_with_attachment(subject, body, recipient_email, display_video_path)

                    # Display success message in Streamlit and show the video
                    st.success("Shoplifting detected. An alert email has been sent.")

                else:
                    # Optionally handle the "Normal" case, e.g., display a different message
                    os.remove(converted_path)
                    st.info("No shoplifting activity detected.")

                st.video(display_video_path, format='video/mp4')
                st.success("Video processed and predictions visualized!")
            except Exception as e:
                st.error(f"An error occurred: {e}")


st.title("SecureShop : Shoplifting Detection")
OUTPUT_DIRECTORY = "annotated_videos"  # Output directory
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)  # Create the directory if it doesn't exist
upload_video()
