import json
import os
from PIL import Image
from data import get_scene_synced_datasets, get_all_sequences_synced_dataset
from torch.utils.data import ConcatDataset

from save_sensor_data import NpEncoder

def extract_bounding_boxes(input_dir, output_file, dataset, visible_threshold=0.90):
    """
    Extracts bounding boxes from the provided dataset and saves them to a JSON file.
    Bounding boxes are retained only if at least 75% of their area lies in the visible zone.

    Args:
        input_dir (str): Directory containing input images.
        output_file (str): Path to save the bounding box annotations as JSON.
        datasets (list): List of synced datasets containing bounding box data.
        visible_threshold (float): Minimum percentage of the bounding box in the visible zone to retain it.
    """
    annotations = {}

    bounding_boxes = {str(data[1]):data[2] for data in dataset}
    # print(list(bounding_boxes.items())[0])

    # Assume the datasets are synchronized with image file order
    for idx, fname in enumerate(sorted(os.listdir(input_dir))):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, fname)
            img = Image.open(image_path).convert('RGB')
            img_size = img.size  # (width, height)

            # Get bounding boxes for the current image
            # print(fname.split('_'))
            timestamp = '_'.join(fname.split('_')[0:3])

            # Convert bounding boxes to [xmin, ymin, xmax, ymax]
            converted_boxes = []

            x, y, w, h = bounding_boxes[timestamp]
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h
            converted_boxes.append([xmin, ymin, xmax, ymax])

            # Determine mask side from file name
            mask_side = 'left' if '_left' in fname else 'right'

            # Filter bounding boxes based on visibility
            # visible_boxes = filter_visible_boxes(converted_boxes, img_size, mask_side, visible_threshold)

            # Save filtered bounding boxes for the current image
            annotations[fname] = []
            for box in converted_boxes:
                annotations[fname].append({
                    'bbox': box  # [xmin, ymin, xmax, ymax]
                })

    # Save all annotations to the output file
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4, cls=NpEncoder)
    print(f"Bounding boxes saved to {output_file}")


# def filter_visible_boxes(boxes, size, mask_side, visible_threshold):
#     """
#     Filters bounding boxes to include only those with at least a given percentage of their area
#     in the visible zone. Excludes boxes entirely in the masked zone.

#     Args:
#         boxes (list): List of bounding boxes [xmin, ymin, xmax, ymax].
#         size (tuple): Original image size (width, height).
#         mask_side (str): 'left' or 'right', indicating the side masked in the image.
#         visible_threshold (float): Minimum percentage of the box area in the visible zone.

#     Returns:
#         list: Filtered bounding boxes.
#     """
#     img_w, img_h = size
#     visible_boxes = []

#     for box in boxes:
#         xmin, ymin, xmax, ymax = box
#         box_area = (xmax - xmin) * (ymax - ymin)

#         if mask_side == 'left':
#             visible_area = max(0, xmax - img_w // 2) * (ymax - ymin)
#         else:  # 'right'
#             visible_area = max(0, img_w // 2 - xmin) * (ymax - ymin)

#         # Check visible area ratio
#         if visible_area / box_area >= visible_threshold:
#             visible_boxes.append(box)
#         else:
#             visible_boxes.append([0,0,0,0])

#     return visible_boxes



if __name__ == "__main__":
    # Load synced datasets (replace this with your dataset loading logic)
    # datasets = get_scene_synced_datasets(full_path='Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951')
    sequences = [f'seq{i}' for i in range(12,15)]
    if len(sequences) != 0:
        with open('project_main/data/files_in_sequence.json', 'r') as f:
            sequence_data = json.load(f)
        
        sequence_paths = [val['sequence'] for seq, val in sequence_data.items() if seq in sequences]
        
        all_datasets = [get_scene_synced_datasets(sequence=seq) for seq in sequence_paths]
        datasets = [ConcatDataset(list(col)) for col in zip(*all_datasets)]

    else:
        datasets = get_all_sequences_synced_dataset()
    
    
    input_dir = "project_main/data/Testing/Masked Images/"  # Directory containing masked images
    output_dir = "project_main/data/Testing/Masked BB Labels/"
    output_file = "bounding_boxes.json"  # File to save bounding box annotations

    ctr = 0
    for dataset in datasets:
        subject = f'Subject{ctr}'
        input_image_dir = input_dir + subject
        output_image_dir = output_dir + subject + '/' + output_file
        print(input_image_dir)
        print(output_image_dir)
        # Extract and save bounding boxes
        extract_bounding_boxes(input_image_dir, output_image_dir, dataset, visible_threshold=0.90)
        ctr +=1




# Output should be in this format:

# {
#     "image1_left.png": [
#         {"bbox": [50, 30, 150, 100]},
#         {"bbox": [200, 120, 250, 170]}
#     ],
#     "image2_right.png": [
#         {"bbox": [40, 25, 140, 90]},
#         {"bbox": [190, 110, 240, 160]}
#     ]
# }
