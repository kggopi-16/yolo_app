import streamlit as st
import cv2
import tempfile
import gc
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# 1. Memory-Efficient Model Loading
@st.cache_resource(ttl=600)
def load_yolo_model(model_path):
    # Ensure it stays on CPU for Render
    return YOLO(model_path)

st.set_page_config(page_title="YOLO Live Stream", layout="wide")

# Sidebar for Model Upload
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload .pt model", type=['pt'])

# Global model state
if "model" not in st.session_state:
    st.session_state.model = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state.model = load_yolo_model(tmp_file.name)
    st.sidebar.success("Model Ready")

# 2. Simplified Video Processing Callback
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Only run inference if model is loaded
    if st.session_state.model is not None:
        # imgsz=320 is critical for Render's low RAM
        results = st.session_state.model(img, conf=0.25, imgsz=320, verbose=False)
        for result in results:
            img = result.plot()
        del results # Manual cleanup to free RAM
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Robust RTC Config for Cloud (Render/Streamlit Cloud)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]}
)

st.title("Live YOLO Object Detection")

if st.session_state.model is None:
    st.warning("Please upload a YOLO .pt model in the sidebar to start detection.")

webrtc_streamer(
    key="yolo-detection",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
