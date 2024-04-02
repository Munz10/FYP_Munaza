# main.py

import streamlit as st
from streamlit_ui import upload_video
from video_view import video_view_page

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_home():
    st.session_state.page = 'home'

def navigate_upload():
    st.session_state.page = 'upload_video'

def navigate_view():
    st.session_state.page = 'video_view'

# Navigation
st.sidebar.button("Home", on_click=navigate_home)
st.sidebar.button("Upload Video", on_click=navigate_upload)
st.sidebar.button("View Videos", on_click=navigate_view)

# Page rendering
if st.session_state.page == 'home':
    st.title("SecureShop: Shoplifting Detector")
    st.header("Welcome to SecureShop")
elif st.session_state.page == 'upload_video':
    upload_video()
elif st.session_state.page == 'video_view':
    video_view_page()
