import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import tempfile
import os

# 1. Page Config
st.set_page_config(page_title="YOLO Cloud", layout="wide")

# 2. Optimized Model Loading (Cached to prevent reloading on every rerun)
@st.cache_resource
def load_yolo_model(model_path):
    # Load model and set to CPU (Streamlit Cloud doesn't have GPUs)
    model = YOLO(model_path)
    return model

# Sidebar Configuration
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload YOLOv8/v11 .pt model", type=['pt'])

# 3. WebRTC Configuration
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Use a container for the model to avoid streamlit_webrtc serialization issues
class ModelContainer:
    def __init__(self):
        self.model = None

if "model_container" not in st.session_state:
    st.session_state.model_container = ModelContainer()

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    st.session_state.model_container.model = load_yolo_model(tmp_path)
    st.sidebar.success("Model Loaded Successfully!")

# 4. Video Callback Function
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Access model from session state container
    model = st.session_state.model_container.model
    
    if model is not None:
        # Run inference
        # imgsz=320 is a good balance for Cloud CPUs; 160 might be too blurry
        results = model.predict(img, conf=0.25, imgsz=320, verbose=False)
        
        # Plot results on the frame
        annotated_frame = results[0].plot()
        
        # YOLO returns BGR, webrtc_streamer expects BGR for 'bgr24'
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI Layout
st.title("🚀 YOLO Cloud Live Stream")
if st.session_state.model_container.model is None:
    st.warning("Please upload a .pt model in the sidebar (e.g., yolov8n.pt)")
else:
    webrtc_streamer(
        key="yolo-detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
