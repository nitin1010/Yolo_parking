# 🚗 YOLO Parking Management System

AI-powered parking management system with real-time license plate detection and automatic slot assignment.

## Features
- Real-time license plate detection using YOLO11
- Automatic parking slot assignment
- Dual-feed GUI (webcam + parking lot video)
- Comprehensive reporting system
- Computer vision-based vehicle detection

## Quick Start
1. Install dependencies: `pip install ultralytics opencv-python easyocr pillow numpy`
2. Download required model files (see HOW_TO_USE_THIS_CODE.txt)
3. Run: `python demo_gui.py`
4. Generate reports: `python generate_assignment_report.py`

## Documentation
See `HOW_TO_USE_THIS_CODE.txt` for complete setup and usage instructions.

## Tech Stack
- Python 3.8+
- YOLO11 (Ultralytics)
- EasyOCR
- OpenCV
- Tkinter GUI