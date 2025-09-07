# 🚦 Traffic Light Detector

A real-time **Traffic Light Detection System** built with **Python, OpenCV, and NumPy**.  
The project detects and classifies traffic lights as **STOP (Red)**, **SLOW (Yellow)**, or **GO (Green)** in real-time from webcam/video input.  
It also provides a Gradio interface to process uploaded videos and return annotated outputs.

---

## Features
- Real-time webcam detection with bounding boxes  
- HSV color space segmentation (robust color detection)  
- ROI (Region of Interest) support for focused detection  
- Classification into **STOP / SLOW / GO**  
- Gradio interface for video upload & annotated output  
- Deployable on **Hugging Face Spaces**  

---

# Project Structure

- **Traffic-light-detector/**
  - `app.py`: Gradio app for video upload and annotation.
  - `requirements.txt`: Project dependencies.
  - `.gitignore`: Files and folders to be ignored by Git.
  - `README.md`: Project documentation.
  - **src/**
    - `main.py`: Final webcam detector script with ROI and state logic (STOP/SLOW/GO).
    - `step1_webcam.py`: Initial webcam access script.
    - `step2_webcam.py`: Enhanced webcam script.
    - `step3_red_detect.py`: Script for detecting the color red.
    - `step3_object_detect.py`: Script for general object detection.
    - `step4_hsv_tuning.py`: Tool for tuning HSV color ranges.
    - `step5_multi_color_detect.py`: Script for detecting multiple colors (red, yellow, green).
    - `step6_detector_smooth.py`: Script to smooth detection output.
    - `step7_stable_detector.py`: Improved stable detector.
    - `step8_roi_detector.py`: Detector with integrated region of interest.
    - `step9_roi.py`: Core ROI logic script.
    - `step10_multi.py`: Multi-stage detection script.
    - `step10_decision.py`: Traffic light state decision logic.
    - `step11_ui.py`: User interface components.
    - `step11_deploy.py`: Deployment script.

---

## Tech Stack
- **Python**
- **OpenCV**
- **NumPy**
- **HSV Color Space**
- **Gradio**

---

## Setup & Run

### 1. Clone repository
```bash
git clone https://github.com/<your-username>/Traffic-light-detector.git
cd Traffic-light-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run real-time webcam detector
```bash
python src/main.py
```

### 5. Run Gradio video app
```bash
python app.py
```

### To share publicly:
```bash
demo.launch(share=True)
```

---

## Deployment

### Hugging Face Spaces
Go to Hugging Face Spaces - New Space
Upload:
- app.py
- requirements.txt

Select SDK: Gradio, Hardware: CPU
Deploy → Public URL will be generated

## Author
- Akshay Sharma
- SRM Institute of Science and Technology
