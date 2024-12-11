import pandas as pd
import os
import cv2
from tqdm import tqdm
import numpy as np


PREDICTIONS_DIR = 'project_main/predictions/'
PREDICTED_FRAMES = 'PredictedFrames/'
ANNOTATED_FRAMES = 'project_main/data/Annotated Images From Pickle/'
MASKED_FRAMES = 'project_main/data/Testing/Masked Images/'
PRED_CSV = 'predictions_regression.csv'

def annotate_image_with_centriods(true, pred, readpath, readpath_masked, writepath, writepath_masked):
    image = cv2.imread(readpath)
    image_masked = cv2.imread(readpath_masked)
    
    # true_x, true_y = true
    # pred_x, pred_y = pred
    # true_center_coordinates = (200, 300)  # (x, y)
    radius = 60
    true_color = (0, 0, 255)  # BLUE
    pred_color = (255, 0, 0)  # RED
    thickness = 2  # Line thickness

    # print(true, pred)

    # return

    cv2.circle(image, true, radius, true_color, thickness)
    cv2.circle(image, pred, radius, pred_color, thickness)

    cv2.circle(image_masked, true, radius, true_color, thickness)
    cv2.circle(image_masked, pred, radius, pred_color, thickness)

    # if os.path.isfile(write_img):
    #     raise FileExistsError(f'Trying to write to {write_img} that already exists')
    cv2.imwrite(writepath, image)
    cv2.imwrite(writepath_masked, image_masked)



if __name__ == '__main__':

    for subject in os.listdir(PREDICTIONS_DIR):
        if os.path.isfile(PREDICTIONS_DIR + subject):
            continue

        sub_ann_frames = ANNOTATED_FRAMES + subject + '/'
        sub_out_frames = PREDICTIONS_DIR + subject + '/' + PREDICTED_FRAMES
        sub_csv = PREDICTIONS_DIR + subject + '/' + PRED_CSV

        # with open(sub_csv, 'r') as f:
        sub_pred_df = pd.read_csv(sub_csv)

        for row in tqdm(sub_pred_df.itertuples(index=False), desc=subject):
            read_img = ANNOTATED_FRAMES + subject + '/' + row[0] + '.png'
            read_img_masked = MASKED_FRAMES + subject + '/' + row[0] + '_left.png'
            
            write_img = sub_out_frames + row[0] + '.png'
            write_img_masked = sub_out_frames + row[0] + '_left.png'
            # print(row[2].type)
            true_vals = np.array((row[1], row[2]))
            pred_vals = np.array((row[3], row[4]))
            # pred_vals = np.array((row[5], row[6])) # For Ridge

            true_vals = tuple(map(int, np.round(true_vals)))
            pred_vals = tuple(map(int, np.round(pred_vals)))

            annotate_image_with_centriods(true_vals, pred_vals, readpath=read_img, readpath_masked=read_img_masked, 
                                          writepath=write_img, writepath_masked=write_img_masked)

            # print(read_img, write_img)




