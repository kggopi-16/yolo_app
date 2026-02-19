# 🎯 YOLO Object Detection App

This application allows users to upload a YOLO model and perform real-time object detection via webcam using `streamlit-webrtc`.

## 🚀 Live App

👉 **[https://yoloapp123.streamlit.app/](https://yoloapp123.streamlit.app/)**

## Features

- **Model Upload**: Upload any `.pt` YOLO model file via the sidebar.
- **Real-time Detection**: Uses WebRTC to stream video from your webcam and apply the YOLO model for object detection.
- **Dynamic Model Loading**: The model is loaded and cached efficiently.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Troubleshooting

If you encounter "Connection is taking longer than expected", the public STUN servers are unable to establish a connection due to your network's firewall or NAT settings. Try accessing the app from a different network (e.g., mobile data) or a less restrictive Wi-Fi.
