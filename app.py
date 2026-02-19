import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import gc
import tempfile
import os

# Set page to wide mode to save UI space
st.set_page_config(page_title="Ultra-Light YOLO", layout="wide")

# 1. Optimized Model Loading
@st.cache_resource(ttl=300) # Automatically clear from RAM after 5 mins of inactivity
def load_yolo_model(model_path):
    # Load model and immediately move to CPU
    model = YOLO(model_path)
    return model

st.sidebar.title("📦 Memory Optimizer")
uploaded_file = st.sidebar.file_uploader("Upload .pt model", type=['pt'])

# Initialize session state for the model
if "model" not in st.session_state:
    st.session_state.model = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state.model = load_yolo_model(tmp_file.name)
    st.sidebar.success("Model Active")

# 2. High-Efficiency Video Callback
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    if st.session_state.model is not None:
        # imgsz=160 is the most aggressive RAM-saving setting (Default is 640)
        # stream=True ensures results are processed as a generator, saving RAM
        results = st.session_state.model(
            img, 
            conf=0.3, 
            imgsz=160, 
            verbose=False,
            half=False # Keep False for CPU-only servers like Render
        )
        
        for result in results:
            img = result.plot()
        
        # CRITICAL: Manual memory cleanup after every frame
        del results
        gc.collect() 
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Robust WebRTC Config
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("🚀 Ultra-Light Live Detection")
st.info("Limit: 512MB RAM. Use YOLO Nano models for stability.")

if st.session_state.model is None:
    st.warning("Upload a model in the sidebar to begin.")

webrtc_streamer(
    key="yolo-ultra-light",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # Improves FPS on slow CPUs
)
