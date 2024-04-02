import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import threading
import time

from model_loader import load_model

model = load_model() # Load the model
# Function to detect shoplifting in real-time from CCTV footage
def detect_shoplifting(stop_event):
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    seq_length = 10
    frames = []

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from the camera.")
            break

        frame = cv2.resize(frame, (128, 128))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

        if len(frames) == seq_length:
            frame_sequence = np.array(frames).astype('float32') / 255.0
            frame_sequence = np.expand_dims(frame_sequence, axis=-1)
            frame_sequence = np.expand_dims(frame_sequence, axis=0)

            prediction = model.predict(frame_sequence)[0][0]
            text = "Shoplifting" if prediction > 0.7 else "Normal"
            color = (0, 0, 255) if text == "Shoplifting" else (0, 255, 0)

            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow('Shoplifting Detection', frame)
            frames = []

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or stop_event.is_set():
            break

        elapsed_time = time.time() - start_time
        if elapsed_time < 1:
            time.sleep(1 - elapsed_time)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Added for potential OpenCV cleanup

# Main function to run the detection
def main():
    st.title("SecureShop : Shoplifting Detection")
    stop_event = threading.Event()
    detection_running = False

    if not detection_running:
        if st.button("Start Detection"):
            detection_running = True
            detection_thread = threading.Thread(target=detect_shoplifting, args=(stop_event,), daemon=True)
            detection_thread.start()

    if detection_running:
        if st.button("Stop Detection"):
            stop_event.set()
            detection_thread.join()
            detection_running = False

if __name__ == "__main__":
    main()
