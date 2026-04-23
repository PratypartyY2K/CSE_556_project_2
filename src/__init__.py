"""
src package entrypoint / utility script.
Provides `convert_video_to_images(video_path, output_path)` to extract frames from a video
using OpenCV and save them as JPEGs.
- Opens the input video with `cv2.VideoCapture` and aborts if it can’t be opened.
- Creates the output directory if needed.
- Iterates through frames and saves every 10th frame as `frame_<index>.jpg` in `output_path`.
- When run as a script, reads `assets/main_video.mp4` (relative to this file) and writes
  extracted frames to `assets/images`, then prints how many frames were saved.
"""

import os
import cv2


def convert_video_to_images(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    os.makedirs(output_path, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            filename = os.path.join(output_path, f"frame_{saved_count}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    video_path = os.path.join(base_dir, "assets", "main_video.mp4")
    output_path = os.path.join(base_dir, "assets", "images")

    convert_video_to_images(video_path, output_path)
