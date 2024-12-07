# src/object_detection.py
import cv2
import os
import torch

# If you run into issues with YOLOv5 model loading, consider:
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
# and remove ultralytics usage. For now, let's try ultralytics YOLO:
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics")

class YOLODetector:
    def __init__(self, weights='yolov5s.pt', device='cpu', conf_thres=0.5):
        """
        Initialize the YOLO detector.
        weights: Path to YOLO model weights.
        device: 'cpu' or 'cuda'
        conf_thres: Confidence threshold for detection.
        """
        self.device = device
        self.model = YOLO(weights)
        self.model.overrides['conf'] = conf_thres
        # To detect only persons, you can filter after inference.

    def detect_in_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return [], []
        results = self.model.predict(source=img, verbose=False, device=self.device)
        boxes = results[0].boxes
        if boxes is None:
            return [], []
        
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = boxes.conf.cpu().numpy() # (N,)
        classes = boxes.cls.cpu().numpy()# (N,)

        person_scores = []
        person_boxes = []

        # Class 0 is 'person' in COCO
        for i, c in enumerate(classes):
            if c == 0:
                conf = confs[i]
                x1, y1, x2, y2 = xyxy[i]
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w/2.0
                cy = y1 + h/2.0
                person_scores.append(conf)
                person_boxes.append([cx, cy, w, h])

        return person_scores, person_boxes

    def detect_in_directory(self, input_dir, output_file="detections.txt"):
        """
        Runs detection on all images in input_dir, saves results to output_file.
        output_file format: frame_name, person_count, [list_of_boxes]
        """
        results = []
        for fname in os.listdir(input_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_dir, fname)
                scores, boxes = self.detect_in_image(img_path)
                results.append((fname, len(scores), boxes))

        # Save detections:
        with open(output_file, 'w') as f:
            for r in results:
                fname, count, boxes = r
                f.write(f"{fname},{count},{boxes}\n")

        return results





if __name__ == "__main__":
    # Quick test on masked frames
    detector = YOLODetector(weights='yolov5s.pt', device='cpu', conf_thres=0.5)
    test_dir = "../data/Masked Frames"
    detections = detector.detect_in_directory(test_dir, output_file="person_detections.txt")
    print(detections)
