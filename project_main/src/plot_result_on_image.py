import pandas as pd
import os
import cv2
from tqdm import tqdm
import numpy as np
import math


PREDICTIONS_DIR = 'project_main/predictions/'
PREDICTED_FRAMES = 'PredictedFrames/'
ANNOTATED_FRAMES = 'project_main/data/Annotated Images From Pickle/'
MASKED_FRAMES = 'project_main/data/Testing/Masked Images/'
PRED_CSV = 'predictions_regression.csv'

RADIUS = 60

def annotate_image_with_centriods(true, pred, readpath, readpath_masked, writepath, writepath_masked):
    image = cv2.imread(readpath)
    image_masked = cv2.imread(readpath_masked)
    
    # true_x, true_y = true
    # pred_x, pred_y = pred
    # true_center_coordinates = (200, 300)  # (x, y)
    true_color = (0, 0, 255)  # BLUE
    pred_color = (255, 0, 0)  # RED
    thickness = 2  # Line thickness

    # print(true, pred)

    # return

    cv2.circle(image, true, RADIUS, true_color, thickness)
    cv2.circle(image, pred, RADIUS, pred_color, thickness)

    cv2.circle(image_masked, true, RADIUS, true_color, thickness)
    cv2.circle(image_masked, pred, RADIUS, pred_color, thickness)

    # if os.path.isfile(write_img):
    #     raise FileExistsError(f'Trying to write to {write_img} that already exists')
    cv2.imwrite(writepath, image)
    cv2.imwrite(writepath_masked, image_masked)

def circle_overlap_area(center1, center2):
    # Calculate the distance between centers
    d = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    # If the circles do not overlap
    if d >= 2*RADIUS:
        return 0
    elif d == 0:
        return 0


    # Calculate the area of overlap
    rsq = RADIUS**2

    part1 = rsq * math.acos((d**2) / (2 * d * RADIUS))
    part2 = rsq * math.acos((d**2 ) / (2 * d * RADIUS))
    part3 = 0.5 * math.sqrt((-d + 2*RADIUS) * (d) * (d ) * (d + 2*RADIUS))

    return part1 + part2 - part3


if __name__ == '__main__':

    for subject in sorted(os.listdir(PREDICTIONS_DIR)):
        if os.path.isfile(PREDICTIONS_DIR + subject):
            continue

        sub_ann_frames = ANNOTATED_FRAMES + subject + '/'
        sub_out_frames = PREDICTIONS_DIR + subject + '/' + PREDICTED_FRAMES
        sub_csv = PREDICTIONS_DIR + subject + '/' + PRED_CSV

        # with open(sub_csv, 'r') as f:
        sub_pred_df = pd.read_csv(sub_csv)
        area_overlap = []

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
            
            area_overlap.append(circle_overlap_area(true_vals, pred_vals))
        
        print(f'Average IoU for {subject} is {np.mean(area_overlap)/100}')

            # print(read_img, write_img)




