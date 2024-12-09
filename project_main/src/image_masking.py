import cv2
import os
import logging

from tqdm import tqdm

def mask_frame_alternately(input_dir, output_dir, suffix_left='_left', suffix_right='_right', sequence_path=None):
    """
    Masks the left and right halves of each frame alternately and saves them with modified filenames.

    Args:
        input_dir (str): Directory containing the input frames.
        output_dir (str): Directory to save the masked frames.
        suffix_left (str): Suffix to append to filenames for left-masked frames.
        suffix_right (str): Suffix to append to filenames for right-masked frames.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # os.rmdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a sorted list of image filenames to ensure consistent processing order
    image_dir = input_dir if sequence_path is None else sequence_path
    image_filenames = sorted([
        fname for fname in os.listdir(image_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    # print(image_filenames)
    # return
    
    if not image_filenames:
        logging.warning(f"No images found in {input_dir}.")
        return
    
    for idx, fname in tqdm(enumerate(image_filenames), desc=image_dir.split('/')[-1]):
        frame_path = os.path.join(input_dir, fname)
        # print(frame_path)
        
        img = cv2.imread(frame_path)
        
        if img is None:
            logging.error(f"Could not read file: {frame_path}")
            continue
        # continue
        h, w, c = img.shape
        
        # Create a copy of the image to apply masking
        img_masked = img.copy()
        
        # Determine which half to mask based on the image index
        if idx % 2 == 0:
            # Even index (starting from 0): Mask the right half
            img_masked[:, w//2:, :] = 0
            suffix = suffix_right
        else:
            # Odd index: Mask the left half
            img_masked[:, :w//2, :] = 0
            suffix = suffix_left
        
        # Construct the output filename with the appropriate suffix
        base_name, ext = os.path.splitext(fname)
        out_fname = f"{base_name}{suffix}{ext}"
        out_path = os.path.join(output_dir, out_fname)
        
        # Save the masked image
        success = cv2.imwrite(out_path, img_masked)
        if not success:
            logging.error(f"Failed to write file: {out_path}")
        # else:
            # logging.info(f"Saved masked image: {out_path}")

# Example Usage
if __name__ == "__main__":
    input_dir = "project_main/data/Annotated Images From Pickle/"  # Directory containing annotated images
    output_dir = "project_main/data/Masked Images/"               # Directory to save masked images
    sequence = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951/RGB_anonymized'
    for subject in os.listdir(input_dir):
        input_image_dir = input_dir + subject
        output_image_dir = output_dir + subject
        # print(input_image_dir)
        # print(output_image_dir)
        mask_frame_alternately(input_image_dir, output_image_dir, sequence_path=None)
