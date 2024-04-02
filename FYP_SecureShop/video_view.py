import os
import streamlit as st

def video_view_page():
    st.title("SecureShop: Shoplifting Detector")
    st.write("Select video from side bar for review")

    video_folder = "C:/Users/Munaza/PycharmProjects/FYP_SecureShop/detected_videos"
    video_files = [f for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]
    video_files.sort()  # Sort the list alphabetically

    st.sidebar.header("Video Files")
    selected_video = st.sidebar.radio("Select a video file", video_files)

    if selected_video:
        st.sidebar.subheader("Selected Video:")
        st.sidebar.write(selected_video)

        video_path = os.path.join(video_folder, selected_video)
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)
