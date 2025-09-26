"""
Extract a frame from your parking video for annotation
"""

import cv2
import os


def extract_frame_from_video():
    """Extract a single frame from video for annotation"""

    print("=== Frame Extraction Tool ===")

    # List video files in current directory
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []

    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
            print(f"📹 Found video: {file}")

    if not video_files:
        print("❌ No video files found!")
        print("Please add your parking lot video to the project folder")
        return

    # Use first video file found
    video_path = "parking_1920_1080_loop.mp4"
    print(f"🎯 Using video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error opening video: {video_path}")
        return

    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📊 Video info: {width}x{height}, {total_frames} frames at {fps} FPS")

    # Extract frame from middle of video
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

    ret, frame = cap.read()

    if ret:
        output_name = "parking_frame.jpg"
        cv2.imwrite(output_name, frame)
        print(f"✅ Frame extracted successfully!")
        print(f"💾 Saved as: {output_name}")
        print(f"🎯 Use this image in the annotation tool")
    else:
        print("❌ Failed to extract frame")

    cap.release()
    print("🏁 Done!")


if __name__ == "__main__":
    extract_frame_from_video()