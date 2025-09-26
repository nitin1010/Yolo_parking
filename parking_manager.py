"""
Parking Management System with slot tracking and assignment logging
"""

import json
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO


class ParkingManager:
    def __init__(self, model_path: str, json_path: str, assignments_json: str = "parking_assignments.json"):
        self.model = YOLO(model_path)
        self.json_path = json_path
        self.assignments_json = assignments_json
        self.parking_data = self.load_parking_data()
        self.slot_occupancy = {}
        self.assignments = self.load_assignments()

    def load_assignments(self) -> List[Dict]:
        """Load existing parking assignments"""
        try:
            with open(self.assignments_json, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_assignment(self, license_plate: str, parking_slot: str) -> bool:
        """Save a new parking assignment"""
        timestamp = datetime.now().isoformat()

        # Check if plate already has an active assignment
        for assignment in self.assignments:
            if (assignment.get("license_plate") == license_plate and
                assignment.get("status") == "assigned"):
                return False  # Already assigned

        assignment = {
            "license_plate": license_plate,
            "timestamp": timestamp,
            "parking_slot": parking_slot,
            "status": "assigned",
            "assignment_id": len(self.assignments) + 1
        }

        self.assignments.append(assignment)

        try:
            with open(self.assignments_json, 'w') as f:
                json.dump(self.assignments, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving assignment: {e}")
            return False

    def assign_plate_to_slot(self, license_plate: str) -> Optional[str]:
        """Assign a license plate to the closest empty slot"""
        # Find closest empty slot (simplified - using first available)
        empty_slots = [name for name, occupied in self.slot_occupancy.items() if not occupied]

        if not empty_slots:
            return None

        assigned_slot = empty_slots[0]  # Take first available slot

        if self.save_assignment(license_plate, assigned_slot):
            # Mark slot as occupied
            self.slot_occupancy[assigned_slot] = True
            return assigned_slot

        return None

    def load_parking_data(self) -> Dict:
        """Load parking slot data from JSON"""
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)

            # Add slot names if not present
            if isinstance(data, list):
                for i, slot in enumerate(data):
                    if 'name' not in slot:
                        slot['name'] = f"A{i+1}"
                self.save_parking_data(data)

            return data if isinstance(data, list) else []
        except FileNotFoundError:
            return []

    def save_parking_data(self, data: List[Dict]) -> None:
        """Save parking data to JSON"""
        try:
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving parking data: {e}")

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Dict]:
        """Detect vehicles in frame"""
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0].item()
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'confidence': confidence
                    })

        return detections

    def point_in_polygon(self, point: Tuple[int, int], polygon: List[List[int]]) -> bool:
        """Check if point is inside polygon"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def update_slot_occupancy(self, detections: List[Dict]) -> None:
        """Update slot occupancy based on vehicle detections"""
        for slot in self.parking_data:
            slot_name = slot.get('name', 'Unknown')
            points = slot.get('points', [])

            if not points:
                continue

            occupied = False
            for detection in detections:
                if self.point_in_polygon(detection['center'], points):
                    occupied = True
                    break

            self.slot_occupancy[slot_name] = occupied

    def get_closest_empty_slot(self, point: Tuple[int, int]) -> Optional[str]:
        """Find closest empty parking slot to a given point"""
        empty_slots = [name for name, occupied in self.slot_occupancy.items() if not occupied]

        if not empty_slots:
            return None

        closest_slot = None
        min_distance = float('inf')

        for slot in self.parking_data:
            slot_name = slot.get('name', 'Unknown')
            if slot_name in empty_slots:
                points = slot.get('points', [])
                if points:
                    # Calculate centroid of slot
                    cx = sum(p[0] for p in points) // len(points)
                    cy = sum(p[1] for p in points) // len(points)

                    # Calculate distance
                    distance = ((point[0] - cx) ** 2 + (point[1] - cy) ** 2) ** 0.5

                    if distance < min_distance:
                        min_distance = distance
                        closest_slot = slot_name

        return closest_slot

    def draw_slots(self, frame: np.ndarray) -> np.ndarray:
        """Draw parking slots on frame"""
        for slot in self.parking_data:
            points = slot.get('points', [])
            slot_name = slot.get('name', 'Unknown')

            if not points:
                continue

            # Convert points to numpy array
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Determine color based on occupancy
            occupied = self.slot_occupancy.get(slot_name, False)
            color = (0, 0, 255) if occupied else (0, 255, 0)  # Red if occupied, green if free

            # Draw polygon
            cv2.polylines(frame, [pts], True, color, 2)

            # Calculate centroid for text placement
            cx = sum(p[0] for p in points) // len(points)
            cy = sum(p[1] for p in points) // len(points)

            # Draw slot name
            cv2.putText(frame, slot_name, (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with vehicle detection and slot tracking"""
        detections = self.detect_vehicles(frame)
        self.update_slot_occupancy(detections)
        return self.draw_slots(frame)