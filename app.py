import streamlit as st
import cv2
import tempfile
import os
import gc
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading

# 1. Memory Management: TTL (Time To Live) removes model from RAM if unused
@st.cache_resource(ttl=600)
def load_model(model_path):
    model = YOLO(model_path)
    # Enable half-precision (FP16) to save 50% RAM if hardware supports it
    try:
        model.to('cpu') # Ensure it's on CPU for Render
    except:
        pass
    return model

st.set_page_config(page_title="Slim YOLO Detector", layout="wide")

# Sidebar
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload .pt model", type=['pt'])

model = None
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        model = load_model(tmp_file_path)
        st.sidebar.success("Model Ready")
        # Cleanup temp file path from memory
        del tmp_file_path
        gc.collect() 
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()

    def update_model(self, model):
        with self.lock:
            self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        with self.lock:
            if self.model is not None:
                # OPTIMIZATION: imgsz=320 reduces RAM usage significantly during math
                results = self.model(img, conf=0.25, imgsz=320, verbose=False)
                for result in results:
                    img = result.plot()
                
                # Cleanup frame-specific objects
                del results
        
        # Periodic garbage collection
        if threading.active_count() < 5: # Avoid spamming GC
             gc.collect()
             
        return av.VideoFrame.from_ndarray(img, format="bgr24")

rtc_config = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]}
)

st.title("Low-RAM Object Detection")

ctx = webrtc_streamer(
    key="yolo-slim-v1", 
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

if ctx.video_processor:
    ctx.video_processor.update_model(model)
