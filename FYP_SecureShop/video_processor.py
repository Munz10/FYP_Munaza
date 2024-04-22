import logging
import shutil
import subprocess
import cv2
import numpy as np
import os
import tempfile
import datetime
import time

OUTPUT_DIRECTORY = "detected_videos"
elapsed_time = 0
def preprocess_video(uploaded_file):
    start_time = time.time()
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
    elapsed_time = time.time() - start_time
    print(f"Processing Time: {elapsed_time} seconds")
    return np.array(video_sequences)

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
    converted_path = output_path.replace(".mp4", "_shoplift.mp4")
    ffmpeg_command = [
        "ffmpeg",
        "-i", output_path,
        "-c:v", "libx264",
        "-crf", "23",
        "-y", converted_path
    ]
    subprocess.run(ffmpeg_command, check=True)  # check=True raises an exception on error

    # Delete the original annotated video file
    os.remove(output_path)

    # Create a copy named "display.mp4" outside the annotated directory
    display_video_path = "display.mp4"  # Directly in the project folder, or specify another path
    print("Absolute display video path:", os.path.abspath(display_video_path))
    shutil.copyfile(converted_path, display_video_path)

    logging.basicConfig(filename='_upload_performance_log.txt', level=logging.INFO)
    logging.info(f"Processing Time: {elapsed_time} seconds")

    return converted_path, display_video_path

