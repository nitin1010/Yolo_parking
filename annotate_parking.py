"""
Parking Space Annotation Tool - Improved File Handling
This script opens a UI for selecting parking spot regions with better file dialog
"""

import os
from ultralytics import solutions

def create_parking_annotations():
    """
    Launch the parking point selection tool with debugging info
    """
    print("=== Parking Space Annotation Tool ===")
    print("Current working directory:", os.getcwd())
    print("Files in current directory:")

    # Show all image files in current directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
            print(f"  📸 {file}")

    if not image_files:
        print("❌ No image files found in current directory!")
        print("Please add a .jpg, .jpeg, or .png image to your project folder")
        return

    print(f"\n✅ Found {len(image_files)} image(s)")
    print("\nInstructions:")
    print("1. The GUI will open - click 'Upload Image' or similar button")
    print("2. Navigate to your project folder if needed")
    print("3. Select your parking lot image")
    print("4. Click on image corners to create parking spot polygons")
    print("5. Save to create bounding_boxes.json")

    print("\n🚀 Opening annotation tool...")

    # Official method from Ultralytics documentation
    try:
        solutions.ParkingPtsSelection()
    except Exception as e:
        print(f"❌ Error opening annotation tool: {e}")
        print("Try installing tkinter: sudo apt install python3-tk (Linux) or reinstall Python (Windows)")

if __name__ == "__main__":
    create_parking_annotations()