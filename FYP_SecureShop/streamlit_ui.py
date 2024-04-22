import logging

import streamlit as st
import os
import tempfile
import time
from model_loader import load_model
from email_sender import send_email_with_attachment
from video_processor import preprocess_video, save_annotated_video
from video_processor import OUTPUT_DIRECTORY


def upload_video():
    st.title("SecureShop: Shoplifting Detector")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])
    if uploaded_file is not None:
        with st.spinner("Preprocessing video..."):
            try:
                model = load_model() # Load the model
                model.compile(run_eagerly=True)
                test_video_sequences = preprocess_video(uploaded_file)
                predictions = model.predict(test_video_sequences)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    video_path = tmpfile.name

                # Save the annotated video and get the path to the converted video
                converted_path, display_video_path = save_annotated_video(video_path, predictions)

                print(display_video_path)

                detection_time = time.time()
                shoplifting_detected = any(pred[0] > 0.7 for pred in predictions)  # Assuming binary classification [0, 1]

                if shoplifting_detected:
                    # Send an email alert with the annotated video attached
                    subject = "Shoplifting Detected"
                    body = "Shoplifting activity has been detected. Please find the attached video clip for review."
                    recipient_email = "fathima.20200253@iit.ac.lk"  # Replace with actual recipient email address
                    send_email_with_attachment(subject, body, recipient_email, display_video_path)

                    # Display success message in Streamlit and show the video
                    st.success("Shoplifting detected. An alert email has been sent.")
                    alert_time = time.time()
                    latency = alert_time - detection_time
                    print(f"Latency: {latency} seconds")
                    logging.basicConfig(filename='_upload_performance_log.txt', level=logging.INFO)
                    logging.info(f"Latency : {latency} seconds")

                else:
                    # Optionally handle the "Normal" case, e.g., display a different message
                    os.remove(converted_path)
                    st.info("No shoplifting activity detected.")

                st.video(display_video_path, format='video/mp4')
                st.success("Video processed and predictions visualized!")
            except Exception as e:
                st.error(f"An error occurred: {e}")