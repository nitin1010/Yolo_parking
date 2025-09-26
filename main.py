"""
Parking Management System using Pre-trained VisDrone Model
Optimized for drone/aerial footage with YOLO11 + VisDrone dataset
"""

import cv2
from ultralytics import solutions

def main():
    """
    Parking management system using pre-trained VisDrone model
    Perfect for drone footage and top-down parking lot views
    """
    print("=== Parking Management System with VisDrone Model ===")

    # Configuration - UPDATE THESE PATHS
    VIDEO_PATH = "../videos/8min.mp4"  # Your parking lot video
    JSON_PATH = "bounding_boxes.json"  # JSON file from annotation tool
    OUTPUT_PATH = "parking_management_output.avi"

    # Using pre-trained VisDrone model (best for drone/aerial footage)
    MODEL_PATH = "best.pt"

    print(f"📹 Video: {VIDEO_PATH}")
    print(f"🤖 Model: {MODEL_PATH} (VisDrone trained)")
    print(f"📍 Parking regions: {JSON_PATH}")
    print(f"💾 Output: {OUTPUT_PATH}")

    # Video capture setup
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"❌ Error reading video file: {VIDEO_PATH}"

    # Get video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    print(f"📏 Video properties: {w}x{h} at {fps} FPS")

    # Video writer setup
    video_writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("🚀 Initializing parking management with VisDrone model...")

    # Initialize parking management object with VisDrone model
    parkingmanager = solutions.ParkingManagement(
        model=MODEL_PATH,  # Pre-trained VisDrone model
        json_file=JSON_PATH,  # Path to parking annotations file

        # Optimized settings for VisDrone model:
        conf=0.25,  # Confidence threshold (VisDrone optimized)
        iou=0.45,   # IoU threshold
        tracker="botsort.yaml",  # Good tracker for vehicles
        show=False,  # Display real-time results - ENABLED
        line_width=2,  # Bounding box line width
    )

    print("✅ Model loaded successfully!")
    print("🔄 Processing video... (this may take a while)")
    print("📺 Video window will open - Press 'q' to quit early")

    frame_count = 0

    # Main processing loop
    while cap.isOpened():
        ret, im0 = cap.read()
        if not ret:
            print("📹 End of video or failed to read frame")
            break

        frame_count += 1

        # Process frame with parking management
        results = parkingmanager(im0)

        # Write processed frame to output video
        video_writer.write(results.plot_im)

        # Display frame in real-time - ENABLED
        cv2.imshow('Parking Management - VisDrone', results.plot_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️ User requested quit")
            break

        # Progress update every 60 frames (every ~2 seconds at 30fps)
        if frame_count % 60 == 0:
            print(f"⏳ Processed {frame_count} frames...")

    # Cleanup
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"\n🎉 Processing complete!")
    print(f"📁 Output saved to: {OUTPUT_PATH}")
    print(f"📊 Total frames processed: {frame_count}")
    print("🚗 The video now shows occupied/available parking spots with counts!")

if __name__ == "__main__":
    main()