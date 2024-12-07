import torch
import torchvision.transforms as T
from PIL import Image
import os
import json

# Load the pretrained DETR model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval().to(device)

# Define image transformations
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# COCO class index for "person"
PERSON_CLASS_INDEX = 1

def extract_bounding_boxes(input_dir, output_file, threshold=0.7, visible_threshold=0.75):
    """
    Extracts bounding boxes for each person in the images and saves them to a JSON file.
    Bounding boxes are retained only if at least 75% of their area lies in the visible zone.

    Args:
        input_dir (str): Directory containing input images.
        output_file (str): Path to save the bounding box annotations as JSON.
        threshold (float): Confidence threshold for filtering detections.
        visible_threshold (float): Minimum percentage of the bounding box in the visible zone to retain it.
    """
    annotations = {}
    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, fname)
            img = Image.open(image_path).convert('RGB')
            img_transformed = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_transformed)

            pred_logits = outputs['pred_logits'][0]
            pred_boxes = outputs['pred_boxes'][0]

            # Convert logits to probabilities and filter by person class
            probs = pred_logits.softmax(-1)
            person_probs = probs[:, PERSON_CLASS_INDEX]
            keep = person_probs > threshold

            # Rescale bounding boxes to image size
            boxes = rescale_bboxes(pred_boxes[keep], img.size)

            # Filter bounding boxes based on visibility
            mask_side = 'left' if '_left' in fname else 'right'
            boxes = filter_visible_boxes(boxes, img.size, mask_side, visible_threshold)

            # Save bounding boxes for the current image
            annotations[fname] = []
            for box, score in zip(boxes.tolist(), person_probs[keep].tolist()):
                annotations[fname].append({
                    'bbox': box,  # [xmin, ymin, xmax, ymax]
                    'score': score
                })

    # Save all annotations to the output file
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4)
    print(f"Bounding boxes saved to {output_file}")


def rescale_bboxes(out_bbox, size):
    """
    Rescales bounding boxes to the original image size.

    Args:
        out_bbox (torch.Tensor): Normalized bounding boxes (cx, cy, w, h).
        size (tuple): Original image size (width, height).

    Returns:
        torch.Tensor: Rescaled bounding boxes (xmin, ymin, xmax, ymax).
    """
    img_w, img_h = size
    b = out_bbox.clone()
    b[:, 0] = out_bbox[:, 0] - 0.5 * out_bbox[:, 2]  # xmin
    b[:, 1] = out_bbox[:, 1] - 0.5 * out_bbox[:, 3]  # ymin
    b[:, 2] = out_bbox[:, 0] + 0.5 * out_bbox[:, 2]  # xmax
    b[:, 3] = out_bbox[:, 1] + 0.5 * out_bbox[:, 3]  # ymax
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def filter_visible_boxes(boxes, size, mask_side, visible_threshold):
    """
    Filters bounding boxes to include only those with at least a given percentage of their area
    in the visible zone. Excludes boxes entirely in the masked zone.

    Args:
        boxes (torch.Tensor): Rescaled bounding boxes (xmin, ymin, xmax, ymax).
        size (tuple): Original image size (width, height).
        mask_side (str): 'left' or 'right', indicating the side masked in the image.
        visible_threshold (float): Minimum percentage of the box area in the visible zone.

    Returns:
        torch.Tensor: Filtered bounding boxes.
    """
    img_w, img_h = size
    visible_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        box_area = (xmax - xmin) * (ymax - ymin)

        if mask_side == 'left':
            visible_area = max(0, xmax - img_w // 2) * (ymax - ymin)
        else:  # 'right'
            visible_area = max(0, img_w // 2 - xmin) * (ymax - ymin)

        if visible_area / box_area >= visible_threshold:
            visible_boxes.append(box)

    return torch.tensor(visible_boxes, dtype=torch.float32)


# Example Usage
if __name__ == "__main__":
    input_dir = "../data/Masked Frames"  # Directory containing masked images
    output_file = "../data/Masked BB Labels/bounding_boxes.json"  # File to save bounding box annotations

    extract_bounding_boxes(input_dir, output_file, threshold=0.7, visible_threshold=0.75)
