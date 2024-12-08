import cv2
import os

def mask_frame(input_dir, output_dir, suffix_left='_left', suffix_right='_right'):
    """
    Masks the left and right halves of each frame and saves them with modified filenames.

    Args:
        input_dir (str): Directory containing the input frames.
        output_dir (str): Directory to save the masked frames.
        suffix_left (str): Suffix to append to filenames for left-masked frames.
        suffix_right (str): Suffix to append to filenames for right-masked frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame_path = os.path.join(input_dir, fname)
            img = cv2.imread(frame_path)
            
            if img is None:
                print(f"Could not read file: {frame_path}")
                continue
            
            h, w, c = img.shape
            
            # Create a copy for left and right masking
            img_left = img.copy()
            img_right = img.copy()
            
            # Mask the left half of the frame
            img_left[:, :w//2, :] = 0
            
            # Mask the right half of the frame
            img_right[:, w//2:, :] = 0
            
            # Save the left-masked frame
            out_left_path = os.path.join(output_dir, os.path.splitext(fname)[0] + suffix_left + os.path.splitext(fname)[1])
            cv2.imwrite(out_left_path, img_left)
            
            # Save the right-masked frame
            out_right_path = os.path.join(output_dir, os.path.splitext(fname)[0] + suffix_right + os.path.splitext(fname)[1])
            cv2.imwrite(out_right_path, img_right)

# Example Usage
if __name__ == "__main__":
    input_dir = "project_main/data/Annotated Images From Pickle/Subject0"  # Directory containing annotated images
    output_dir = "project_main/data/Masked Images/Subject0"   # Directory to save masked images
    mask_frame(input_dir, output_dir)
