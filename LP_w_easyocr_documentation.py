"""
Fixed License Plate Detection and OCR from Webcam - Optimized EasyOCR
Uses proper EasyOCR parameters based on official documentation and best practices
"""

import cv2
import numpy as np
import easyocr
import json
import os
import re
from datetime import datetime
from ultralytics import YOLO
from typing import Optional, Tuple, List


class ImprovedLicensePlateDetector:
    def __init__(self, model_path: str = "license-plate-finetune-v1x.pt", json_file: str = "license_plates.json"):
        """Initialize the license plate detector"""
        self.model_path = model_path
        self.json_file = json_file

        # Initialize YOLO model
        self.model = YOLO(model_path)

        # Initialize EasyOCR reader with optimized parameters
        self.ocr_reader = easyocr.Reader(
            ['en'],
            gpu=True,
            model_storage_directory=None,
            download_enabled=True
        )

        # Load existing detections or create new file
        self.detections = self.load_detections()

    def load_detections(self) -> List[dict]:
        """Load existing detections from JSON file"""
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_detection(self, plate_text: str, timestamp: str, confidence_score: float) -> bool:
        """Save a new detection to JSON file"""
        detection = {
            "license_plate": plate_text,
            "timestamp": timestamp,
            "confidence": confidence_score,
            "detection_id": len(self.detections) + 1
        }

        # Avoid duplicate entries (same plate within 3 seconds)
        current_time = datetime.fromisoformat(timestamp)
        for existing in reversed(self.detections[-5:]):
            existing_time = datetime.fromisoformat(existing["timestamp"])
            time_diff = (current_time - existing_time).total_seconds()
            if existing["license_plate"] == plate_text and time_diff < 3:
                return False

        self.detections.append(detection)

        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.detections, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving detection: {e}")
            return False

    def preprocess_license_plate_advanced(self, plate_image: np.ndarray) -> np.ndarray:
        """Advanced preprocessing for license plates"""
        try:
            height, width = plate_image.shape[:2]
            if height < 60 or width < 120:
                scale_factor = max(60 / height, 120 / width) * 1.5
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                plate_image = cv2.resize(plate_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image.copy()

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)

            binary = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

            return cleaned

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return plate_image

    def extract_text_from_plate_optimized(self, plate_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Extract text using optimized EasyOCR parameters"""
        try:
            processed_image = self.preprocess_license_plate_advanced(plate_image)

            results = self.ocr_reader.readtext(
                processed_image,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                decoder='beamsearch',
                beamWidth=5,
                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.3,
                canvas_size=2560,
                mag_ratio=1.0,
                contrast_ths=0.1,
                adjust_contrast=0.5,
                slope_ths=0.1,
                ycenter_ths=0.5,
                height_ths=0.7,
                width_ths=0.5,
                add_margin=0.1,
                detail=1,
                paragraph=False,
                batch_size=1,
                workers=0
            )

            if not results:
                return None, 0.0

            valid_results = []
            for result in results:
                bbox, text, confidence = result
                text = text.strip().upper()
                text = re.sub(r'[^A-Z0-9]', '', text)

                if len(text) >= 4 and len(text) <= 10 and confidence > 0.3:
                    valid_results.append((text, confidence))

            if not valid_results:
                return None, 0.0

            best_text, best_confidence = max(valid_results, key=lambda x: x[1])
            return best_text, best_confidence

        except Exception as e:
            print(f"OCR Error: {e}")
            return None, 0.0

    def detect_frame(self, frame: np.ndarray, conf_threshold: float = 0.4) -> List[dict]:
        """Detect license plates in a single frame and return detection data"""
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0].item()

                    padding = 10
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)

                    plate_crop = frame[y1:y2, x1:x2]

                    if plate_crop.size > 0:
                        plate_text, ocr_confidence = self.extract_text_from_plate_optimized(plate_crop)

                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'plate_text': plate_text,
                            'ocr_confidence': ocr_confidence
                        })

        return detections