import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading

# Set page config
st.set_page_config(page_title="Streamlit Object Detection", layout="wide")

# Sidebar
st.sidebar.title("Model Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your .pt model", type=['pt'])

# Caching the model loading to prevent reloading on every run
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model = None
if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        model = load_model(tmp_file_path)
        st.sidebar.success("Model Loaded Successfully")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        # Clean up if invalid
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# Main Area
st.title("Live Object Detection Feed")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()

    def update_model(self, model):
        with self.lock:
            self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Acquire lock to ensure thread safety when accessing the model
        with self.lock:
            if self.model is not None:
                # Run inference
                results = self.model(img)
                # Plot results (returns numpy array)
                for result in results:
                    img = result.plot()
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# STUN server configuration for cloud deployment (Render)
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="object-detection", 
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False}
)

# Update the video processor with the loaded model
if ctx.video_processor:
    ctx.video_processor.update_model(model)

if uploaded_file is None:
    st.info("Please upload a YOLO .pt model in the sidebar to start detection.")
