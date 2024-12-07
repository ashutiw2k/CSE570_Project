# src/main.py
import os
from preprocess import mask_half_frame
from object_detection import YOLODetector

if __name__ == "__main__":
    # Paths
    original_frames_dir = "../data/Video Frames"
    masked_frames_dir = "../data/Masked Frames"
    os.makedirs(masked_frames_dir, exist_ok=True)

    # 1. Mask the frames (only if not done already)
    mask_half_frame(input_dir=original_frames_dir, output_dir=masked_frames_dir, mask_side='right')

    # 2. Detect persons in masked frames
    detector = YOLODetector(weights='yolov5s.pt', device='cpu', conf_thres=0.5)
    results = detector.detect_in_directory(input_dir=masked_frames_dir, output_file="masked_detections.txt")

    # Print some sample results
    for r in results:
        fname, count, boxes = r
        print(f"Frame: {fname}, Persons Detected: {count}, Boxes: {boxes}")
