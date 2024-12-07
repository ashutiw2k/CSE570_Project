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
    'person']


def rescale_bboxes(out_bbox, size):
    # Convert the normalized box coordinates [0,1] to pixel coordinates
    img_w, img_h = size
    b = out_bbox * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect_and_save(image_path, output_path, threshold=0.7):
    
    for fname in os.listdir(input_image_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame_path = os.path.join(input_image_path, fname)
            img = cv2.imread(frame_path)

            # Load and preprocess the image
            img = Image.open(image_path).convert('RGB')
            img_transformed = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_transformed)

            pred_logits = outputs['pred_logits'][0]
            pred_boxes = outputs['pred_boxes'][0]

            # Convert logits to probabilities and remove background class
            probs = pred_logits.softmax(-1)[:, :-1]

            # Filter based on threshold
            keep = probs.max(dim=-1).values > threshold
            boxes = rescale_bboxes(pred_boxes[keep], img.size)
            plot_and_save(img, probs[keep], boxes, output_path)


def plot_and_save(pil_img, prob, boxes, output_path):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='blue', linewidth=3))
        cls = p.argmax()
        text = f"{COCO_CLASSES[cls]}: {p[cls]:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()



input_image_path = '../data/Video Frames'
output_image_path = '../data/Annotated Frames'


detect_and_save(input_image_path, output_image_path, threshold=0.7)
print(f"Saved annotated image to {output_image_path}")