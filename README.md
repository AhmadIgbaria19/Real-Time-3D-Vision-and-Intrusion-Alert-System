# Real-Time 3D Vision & Smart Intrusion Detection System 🚨🎥

### Developed by: Igbaria Ahmad  
**University of Haifa**  
**March 2025**

---

## 🔍 Overview
This project implements a real-time computer vision system for 3D reconstruction and intelligent person detection at the entrance of a house using only a mobile camera and Python + OpenCV.

It combines multiple stages:
- Camera calibration
- Image undistortion
- Feature detection & matching
- 3D triangulation
- Intrusion detection using YOLOv8
- Automatic email alert system with snapshot capture

---

## 🧠 Modules and Stages

### 📷 1. Camera Calibration
- Calibrates the camera using a 10×7 checkerboard pattern.
- Calculates intrinsic matrix, distortion coefficients, and reprojection error (MSE).

### 🌀 2. Image Undistortion
- Corrects lens distortions using calibration parameters.
- Produces clean, geometrically accurate images for further processing.

### 🧩 3. Feature Matching
- Uses ORB + BFMatcher to find keypoint matches between consecutive frames.
- Saves `.npy` files and match-visualization images.

### 🧠 4. 3D Triangulation
- Computes 3D coordinates from matched 2D keypoints.
- Filters unreliable matches using pixel distance (Δ).
- Saves 3D plots with statistics + delta histograms.

### 🚨 5. Real-Time Person Detection (Surveillance Mode)
- Uses YOLOv8 to detect a person in front of the door.
- Undistorts the video stream, adds warning overlays.
- Sends an alert email with timestamped image attached.

---


---

## ⚙️ Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- `ultralytics` (for YOLOv8)

## 🧠 YOLOv8 Model

This project uses [YOLOv8](https://github.com/ultralytics/ultralytics) from Ultralytics for person detection.
```bash

- Gmail account (with App Password for SMTP)

Install dependencies:
```bash
pip install opencv-python numpy ultralytics
python main.py

📬 Email Alert System
When a person is detected:

A red warning flashes on the screen

A snapshot is saved to DetectedPerson/

An email is sent to the user with the image attached

⚠️ Make sure to set up a Gmail App Password and update it in the code.

📚 References
OpenCV Calibration Docs

LearnOpenCV – Calibration Guide

GeeksForGeeks – Feature Matching

Triangulation StackOverflow

Igbaria Ahmad
B.Sc. Student in Computer Science
University of Haifa
March 2025

