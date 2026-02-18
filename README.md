# Streamlit Object Detection App

This application allows users to upload a YOLO model and perform real-time object detection via webcam using `streamlit-webrtc`.

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

## Features

- **Model Upload**: Upload any `.pt` YOLO model file via the sidebar.
- **Real-time Detection**: Uses WebRTC to stream video from your webcam and apply the YOLO model for object detection.
- **Dynamic Model Loading**: The model is loaded and cached efficiently.

## Deployment on Render

This app is configured for deployment on Render.
1. Create a new **Web Service** on Render.
2. Connect your repository.
3. Set the Build Command to `pip install -r requirements.txt`.
4. Set the Start Command to `streamlit run app.py`.
