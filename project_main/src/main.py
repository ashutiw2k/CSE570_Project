# src/main.py
import os
import json
from CSE570_Project.project_main.src.image_masking import mask_half_frame
from object_detection import YOLODetector

if __name__ == "__main__":
    # Paths
    original_frames_dir = "../data/Video Frames"
    masked_frames_dir = "../data/Masked Frames"
    os.makedirs(masked_frames_dir, exist_ok=True)

    # 1. Mask the frames (only if not done already)
    mask_half_frame(input_dir=original_frames_dir, output_dir=masked_frames_dir, mask_side='right')

    # 2. Detect persons in masked frames
    # Modify YOLODetector to ensure detect_in_directory returns results instead of writing to file.
    detector = YOLODetector(weights='yolov5s.pt', device='cpu', conf_thres=0.5)
    results = detector.detect_in_directory(input_dir=masked_frames_dir)  # should return list of (fname, count, boxes)

    output_file = "masked_detections.txt"
    with open(output_file, 'w') as f:
        for fname, count, boxes in results:
            # Convert boxes list to JSON string for a properly formatted array
            boxes_json = json.dumps(boxes)
            f.write(f"{fname},{count},{boxes_json}\n")

    # Print some sample results
    for fname, count, boxes in results:
        print(f"Frame: {fname}, Persons Detected: {count}, Boxes: {boxes}")
