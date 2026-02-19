import streamlit as st
import tempfile
import os
import threading
import numpy as np

# Page config - must be first Streamlit call
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🎯",
    layout="wide"
)

# ── Lazy imports so the app can at least render even if packages are missing ──
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Guard: show friendly error if core packages failed
# ─────────────────────────────────────────────────────────────────────────────
if cv2 is None or YOLO is None:
    st.error(
        "⚠️ Required packages (opencv or ultralytics) are not installed correctly. "
        "Please check `requirements.txt` and `packages.txt`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – Model upload
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Model Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Upload your YOLO .pt model", type=["pt"]
)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached so it doesn't reload on every Streamlit re-run)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    """Load a YOLO model from a file path and cache it."""
    return YOLO(model_path)


model = None
model_tmp_path = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(uploaded_file.getvalue())
        model_tmp_path = tmp.name

    try:
        model = load_model(model_tmp_path)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        if model_tmp_path and os.path.exists(model_tmp_path):
            os.remove(model_tmp_path)
        model = None

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎯 Live YOLO Object Detection")

if not WEBRTC_AVAILABLE:
    st.error(
        "streamlit-webrtc or av is not available. "
        "Ensure both are listed in requirements.txt."
    )
    st.stop()

if uploaded_file is None:
    st.info("👈 Please upload a YOLO `.pt` model in the sidebar to begin.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# WebRTC Video Processor
# ─────────────────────────────────────────────────────────────────────────────
class VideoProcessor(VideoProcessorBase):
    """Thread-safe video processor that runs YOLO inference on each frame."""

    def __init__(self):
        self._model = None
        self._confidence = 0.5
        self._lock = threading.Lock()

    def update(self, model):
        with self._lock:
            self._model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        with self._lock:
            current_model = self._model

        if current_model is not None:
            try:
                results = current_model(img, verbose=False)
                for result in results:
                    img = result.plot()  # annotated numpy array
            except Exception:
                pass  # silently skip a bad frame

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─────────────────────────────────────────────────────────────────────────────
# ICE / STUN configuration for Streamlit Cloud
# Using multiple public STUN servers for reliability
# ─────────────────────────────────────────────────────────────────────────────
rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
            {"urls": ["stun:stun.services.mozilla.com"]},
        ]
    }
)

ctx = webrtc_streamer(
    key="yolo-object-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,   # non-blocking processing for better performance
)

# Push the latest model & confidence into the running processor
if ctx.video_processor and model is not None:
    ctx.video_processor.update(model)

st.caption(
    "📷 Click **START** above to open your webcam. "
    "Detections will be drawn in real-time using your uploaded model."
)
