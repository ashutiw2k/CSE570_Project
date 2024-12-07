import os
import json
import cv2
import shutil

# Paths (adjust as needed)
BASE_DIR = "project-main"
MASKED_FRAMES_DIR = os.path.join(BASE_DIR, "data", "Masked Frames")
DETECTIONS_FILE = os.path.join(BASE_DIR, "masked_detections.txt")

TRANSTRACK_DATASET_DIR = os.path.join(BASE_DIR, "TransTrack", "datasets", "my_sequence")
IMG_DIR = os.path.join(TRANSTRACK_DATASET_DIR, "img1")
GT_DIR = os.path.join(TRANSTRACK_DATASET_DIR, "gt")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

# Step 1: Parse the YOLO detections file
# Expected line format:
# frame_name,person_count,[[cx, cy, w, h], [cx2, cy2, w2, h2], ...]
#
# Example:
# frame_0001.jpg,1,[[150,200,50,100]]
#
# We'll parse line by line.

detections = []
with open(DETECTIONS_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Split by commas: frame_name, person_count, boxes_str
        # We use maxsplit=2 because boxes might have commas inside the JSON.
        parts = line.split(',', 2)
        frame_name = parts[0]
        count = int(parts[1])
        
        boxes_str = parts[2].strip()
        # Convert boxes_str to a Python list
        boxes = json.loads(boxes_str)  # e.g. [[cx, cy, w, h], ...]

        detections.append((frame_name, boxes))

# Step 2: Rename and copy masked frames to MOT-like dataset structure
# We will name frames as: 000001.jpg, 000002.jpg, ...
frame_id_map = {}
frame_id = 1

for (frame_name, boxes) in detections:
    new_frame_name = f"{frame_id:06d}.jpg"
    src_path = os.path.join(MASKED_FRAMES_DIR, frame_name)
    dst_path = os.path.join(IMG_DIR, new_frame_name)

    if not os.path.isfile(src_path):
        print(f"Warning: {src_path} does not exist. Check your paths.")
    else:
        shutil.copy(src_path, dst_path)

    frame_id_map[frame_name] = frame_id
    frame_id += 1

# Step 3: Create gt.txt file
# Format per line: frame_id, track_id, x, y, w, h, 1, -1, -1
# Convert center-based boxes (cx, cy, w, h) to top-left (x, y, w, h)
gt_txt_path = os.path.join(GT_DIR, "gt.txt")

with open(gt_txt_path, 'w') as fgt:
    for (frame_name, boxes) in detections:
        fid = frame_id_map[frame_name]
        # If multiple boxes per frame and we want unique IDs, we could increment track_id.
        # For simplicity, we use track_id=1 for all detections.
        # Modify as needed if you have multiple distinct objects.
        track_id = 1
        for box in boxes:
            cx, cy, w, h = box
            x_tl = cx - w/2
            y_tl = cy - h/2
            # conf=1, class=-1, visibility=-1 for GT
            fgt.write(f"{fid},{track_id},{x_tl},{y_tl},{w},{h},1,-1,-1\n")

# Step 4: Create seqinfo.ini
seq_length = len(detections)
# Get image size from the first image:
first_image_path = os.path.join(IMG_DIR, f"{1:06d}.jpg")
img = cv2.imread(first_image_path)
if img is not None:
    height, width = img.shape[:2]
else:
    # If no image is found, assign default. Check if images are copied correctly.
    height, width = 1080, 1920

seqinfo_path = os.path.join(TRANSTRACK_DATASET_DIR, "seqinfo.ini")
with open(seqinfo_path, 'w') as f:
    f.write("[Sequence]\n")
    f.write(f"name=my_sequence\n")
    f.write("imDir=img1\n")
    f.write("frameRate=30\n")  # arbitrary, adjust if known
    f.write(f"seqLength={seq_length}\n")
    f.write(f"imWidth={width}\n")
    f.write(f"imHeight={height}\n")
    f.write("imExt=.jpg\n")

print("Conversion complete!")
print("Your dataset is located at:", TRANSTRACK_DATASET_DIR)
