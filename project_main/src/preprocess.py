import cv2
import os
import numpy as np


def mask_half_frame(input_dir, output_dir, mask_side='right'):
    """
    Masks half of each frame (e.g., the right half).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame_path = os.path.join(input_dir, fname)
            img = cv2.imread(frame_path)
            
            if img is None:
                continue
            
            h, w, c = img.shape
            if mask_side == 'right':
                img[:, w//2:w, :] = 0  # Black out the right half
            else:
                img[:, :w//2, :] = 0  # Black out the left half
            
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, img)





# If needed to run standalone:
if __name__ == "__main__":
    input_dir = "../data/Video Frames"
    output_dir = "../data/Masked Frames"
    mask_half_frame(input_dir, output_dir, mask_side='right')

