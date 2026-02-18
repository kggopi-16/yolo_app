import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading

# 1. Page Configuration
st.set_page_config(page_title="YOLO Live Detector", layout="wide", page_icon="🔍")

# 2. Sidebar - Model Upload
st.sidebar.title("🚀 Model Hub")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload YOLO .pt model", type=['pt'])

# Cache the model so it doesn't reload and crash the app
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model = None
if uploaded_file is not None:
    # Save uploaded file to a temporary file because YOLO needs a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        model = load_model(tmp_file_path)
        st.sidebar.success("✅ Model Loaded!")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")

# 3. Video Processing Class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()

    def update_model(self, model):
        with self.lock:
            self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Thread-safe inference
        with self.lock:
            if self.model is not None:
                # Running detection with a hardcoded threshold (0.25)
                results = self.model(img, conf=0.25)
                for result in results:
                    img = result.plot()
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Production WebRTC Configuration (STUN/TURN)
# This bypasses the "Connection taking longer" error on Render
rtc_config = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]}
)

# 5. Main UI
st.title("🎥 Real-Time Object Detection")
st.markdown("Upload your `.pt` model in the sidebar and click **Start** to begin.")

ctx = webrtc_streamer(
    key="yolo-detection-main", 
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True  # Keeps the UI responsive
)

# Connect the loaded model to the background video processor
if ctx.video_processor:
    ctx.video_processor.update_model(model)

if uploaded_file is None:
    st.warning("⚠️ Please upload a YOLO model in the sidebar to activate the camera.")
