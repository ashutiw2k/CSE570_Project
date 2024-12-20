import os
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Use a device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pretrained DETR model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval().to(device)

# Standard PyTorch mean-std normalization for the input image
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# Here exists the classes for which we need to detect. I am keeping only person in it for now. 
COCO_CLASSES = [
    'N/A', 'person'
]

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    # Convert (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
    b = out_bbox.clone().to(device)
    b[:, 0] = out_bbox[:, 0] - 0.5 * out_bbox[:, 2]  # xmin
    b[:, 1] = out_bbox[:, 1] - 0.5 * out_bbox[:, 3]  # ymin
    b[:, 2] = out_bbox[:, 0] + 0.5 * out_bbox[:, 2]  # xmax
    b[:, 3] = out_bbox[:, 1] + 0.5 * out_bbox[:, 3]  # ymax
    # Scale to image size
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b


def detect_and_save(image_path, output_path, threshold=0.7):
    
    for fname in os.listdir(image_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame_path = os.path.join(image_path, fname)
            # img = cv2.imread(frame_path)

            # Load and preprocess the image
            img = Image.open(frame_path).convert('RGB')
            img_transformed = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_transformed)    

            pred_logits = outputs['pred_logits'][0]
            pred_boxes = outputs['pred_boxes'][0]

            # Convert logits to probabilities and remove background class
            probs = pred_logits.softmax(-1)[:, :-1]

            # Get predictions for "person" class (class index 1)
            person_class_index = 1
            person_probs = probs[:, person_class_index]
            keep = person_probs > threshold  # Filter based on threshold

            # Get the bounding boxes and probabilities for people
            boxes = rescale_bboxes(pred_boxes[keep], img.size).to('cpu')
            person_probs_filtered = probs[keep]

            plot_and_save(img, probs[keep], boxes, os.path.join(output_path, fname))



def plot_and_save(pil_img, prob, boxes, output_path):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cls = p.argmax()
            if COCO_CLASSES[cls] == "person":  # Only annotate people
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color='blue', linewidth=3))
                text = f"{COCO_CLASSES[cls]}: {p[cls]:0.2f}"
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()



input_image_path = '20201223_140951-005/RGB_anonymized/'
output_image_path = 'project_main/data/Annotated Images/'


detect_and_save(input_image_path, output_image_path, threshold=0.7)
print(f"Saved annotated image to {output_image_path}")