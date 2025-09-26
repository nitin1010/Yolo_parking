"""
Dual-feed parking management system with license plate detection and assignment tracking
"""

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import json
import os
from typing import Optional, Tuple
import numpy as np

from LP_w_easyocr_documentation import ImprovedLicensePlateDetector
from parking_manager import ParkingManager


class DualFeedParkingGUI:
    def __init__(self):
        self.config = self.load_config()
        self.root = tk.Tk()
        self.root.title("Parking Management System")
        self.root.geometry("1200x600")

        self.setup_gui()

        # Initialize detectors
        try:
            self.plate_detector = ImprovedLicensePlateDetector(
                model_path=self.config["model_path"]
            )
            self.parking_manager = ParkingManager(
                model_path=self.config["parking_model_path"],
                json_path=self.config["json_path"],
                assignments_json=self.config["assignments_json"]
            )
        except Exception as e:
            print(f"Error initializing models: {e}")
            self.plate_detector = None
            self.parking_manager = None

        # Threading control
        self.running = False
        self.camera_thread = None
        self.video_thread = None

        # Plate assignment tracking
        self.processed_plates = set()  # Track processed plates to avoid duplicates
        self.latest_assignment = None  # Latest assignment info for overlay

    def load_config(self) -> dict:
        """Load configuration from JSON file"""
        try:
            with open("config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "camera_index": 0,
                "video_path": "../videos/8min.mp4",
                "model_path": "license-plate-finetune-v1x.pt",
                "parking_model_path": "best.pt",
                "json_path": "bounding_boxes.json",
                "assignments_json": "parking_assignments.json",
                "confidence_threshold": 0.4,
                "ocr_confidence_threshold": 0.6
            }

    def setup_gui(self) -> None:
        """Setup the GUI layout"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Webcam feed
        left_frame = ttk.LabelFrame(main_frame, text="Live Webcam Feed")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.camera_label = ttk.Label(left_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # Right side - Video feed
        right_frame = ttk.LabelFrame(main_frame, text="Parking Lot Video")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.video_label = ttk.Label(right_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_feeds)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_feeds, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side=tk.RIGHT, padx=5)

    def start_feeds(self) -> None:
        """Start both camera and video feeds"""
        if not self.plate_detector or not self.parking_manager:
            self.status_label.config(text="Error: Models not loaded")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.camera_thread.start()

        # Start video thread
        self.video_thread = threading.Thread(target=self.video_worker, daemon=True)
        self.video_thread.start()

        self.status_label.config(text="Running")

    def stop_feeds(self) -> None:
        """Stop both feeds"""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped")

    def process_plate_detection(self, frame: np.ndarray, label_widget: ttk.Label) -> np.ndarray:
        """Process license plate detection on frame and assign parking slots"""
        detections = self.plate_detector.detect_frame(frame, self.config["confidence_threshold"])

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            plate_text = detection['plate_text']
            ocr_confidence = detection['ocr_confidence']

            # Draw bounding box
            color = (0, 255, 0) if plate_text else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if (plate_text and
                ocr_confidence > self.config["ocr_confidence_threshold"] and
                plate_text not in self.processed_plates):

                # Assign plate to slot
                assigned_slot = self.parking_manager.assign_plate_to_slot(plate_text)

                if assigned_slot:
                    self.processed_plates.add(plate_text)
                    self.latest_assignment = {
                        "plate": plate_text,
                        "slot": assigned_slot,
                        "timestamp": self.parking_manager.assignments[-1]["timestamp"]
                    }
                    print(f"✅ Assigned {plate_text} to slot {assigned_slot}")

                # Display plate text
                cv2.putText(frame, f"{plate_text} ({ocr_confidence:.2f})",
                           (x1, max(y1 - 15, 30)), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 255, 0), 2)

        return frame

    def add_overlay_text(self, frame: np.ndarray) -> np.ndarray:
        """Add overlay text when plate is assigned"""
        if self.latest_assignment:
            text = f"Assigned: {self.latest_assignment['plate']} -> {self.latest_assignment['slot']}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Show timestamp on second line
            time_text = f"Time: {self.latest_assignment['timestamp'][:19]}"
            cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return frame

    def camera_worker(self) -> None:
        """Worker thread for camera feed"""
        cap = cv2.VideoCapture(self.config["camera_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Process license plate detection and assignment
            frame = self.process_plate_detection(frame, self.camera_label)

            # Add overlay text showing latest assignment
            frame = self.add_overlay_text(frame)

            # Convert and display
            self.update_display(frame, self.camera_label)

        cap.release()

    def video_worker(self) -> None:
        """Worker thread for video feed"""
        cap = cv2.VideoCapture(self.config["video_path"])

        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Process parking management (vehicle detection and slot drawing)
            frame = self.parking_manager.process_frame(frame)

            # Add overlay text showing latest assignment
            frame = self.add_overlay_text(frame)

            # Convert and display
            self.update_display(frame, self.video_label)

            # Control playback speed
            threading.Event().wait(0.033)  # ~30 FPS

        cap.release()

    def update_display(self, frame: np.ndarray, label_widget: ttk.Label) -> None:
        """Update display with processed frame"""
        try:
            # Resize frame to fit display
            height, width = frame.shape[:2]
            display_width = 580
            display_height = int(height * display_width / width)

            frame_resized = cv2.resize(frame, (display_width, display_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)

            # Update label
            label_widget.configure(image=photo)
            label_widget.image = photo  # Keep a reference

        except Exception as e:
            print(f"Display update error: {e}")

    def run(self) -> None:
        """Run the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self) -> None:
        """Handle window closing"""
        self.running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1)
        if self.video_thread:
            self.video_thread.join(timeout=1)
        self.root.destroy()


if __name__ == "__main__":
    app = DualFeedParkingGUI()
    app.run()