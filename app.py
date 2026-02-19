import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from streamlit.runtime.scriptrunner import add_script_run_context

# 1. ADD STUN/TURN SERVERS
# Without a TURN server, Streamlit Cloud often fails to relay video data
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]}
)

st.set_page_config(page_title="YOLO Cloud Fix", layout="wide")

@st.cache_resource
def load_model():
    # Use the smallest model possible for Cloud stability
    return YOLO("yolov8n.pt") 

model = load_model()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # 2. RUN INFERENCE (Reduced size for CPU speed)
    results = model.predict(img, conf=0.25, imgsz=320, verbose=False)
    annotated_frame = results[0].plot()
    
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

st.title("🚀 YOLO Cloud Deployment")

# 3. CONFIGURE STREAMER FOR CLOUD
ctx = webrtc_streamer(
    key="yolo-cloud",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # Drops frames if CPU lags (Prevents freezing)
)

# 4. FIX THE 'missing ScriptRunContext' WARNING
if ctx.video_processor:
    add_script_run_context(ctx.video_processor)
