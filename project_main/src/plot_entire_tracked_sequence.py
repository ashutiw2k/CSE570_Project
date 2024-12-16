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

RADIUS = 6

def annotate_image_with_path(current_true, past_true, current_pred, past_pred, image):
    # true_center_coordinates = (200, 300)  # (x, y)
    true_color  = (0, 0, 255)  # BLUE
    pred_color = (255, 0, 0)  # RED
    thickness = 3  # Line thickness

    if current_pred > (0,0):
        cv2.circle(image, current_pred, RADIUS, pred_color, thickness)
        if past_pred != (-1,-1):
            cv2.line(image, past_pred, current_pred, pred_color, thickness) 
    else:
        current_pred = (-1,-1)
    
    if current_true > (0,0):
        cv2.circle(image, current_true, RADIUS, true_color, thickness)     
        if past_true != (-1,-1):
            cv2.line(image, past_true, current_true, true_color, thickness) 
    else:    
        current_true = (-1,-1)
    

    

    return current_true, current_pred




if __name__ == '__main__':

    subject = 'Subject4'

    sub_ann_frames = ANNOTATED_FRAMES + subject + '/'
    sub_out_frames = PREDICTIONS_DIR + subject + '/' + PREDICTED_FRAMES
    sub_csv = PREDICTIONS_DIR + subject + '/' + PRED_CSV

    # with open(sub_csv, 'r') as f:
    sub_pred_df = pd.read_csv(sub_csv)
    area_overlap = []

    read_img = ANNOTATED_FRAMES + subject + '/' + '2020-12-29 16_24_49.163930.png'
    read_img_masked = MASKED_FRAMES + subject + '/' + '2020-12-29 16_24_49.163930_left.png'
    
    write_img = 'project_results/SubjectPathComparisons.png'
    # write_img_masked = sub_out_frames + row[0] + '_left.png'
    # print(row[2].type)

    pred_past = (-1,-1)
    true_past = (-1,-1)

    img_timestamps = [
        "2020-12-29 16_24_49.163930",
        "2020-12-29 16_24_50.163987",
        "2020-12-29 16_24_51.163991",
        "2020-12-29 16_24_52.164076",
        "2020-12-29 16_24_53.164179",
        "2020-12-29 16_24_54.164147",
        "2020-12-29 16_24_55.164149",
        "2020-12-29 16_24_56.164281",
        "2020-12-29 16_24_57.164218",
        "2020-12-29 16_24_58.164412",
        "2020-12-29 16_24_59.164684",
        "2020-12-29 16_25_00.164687",
        "2020-12-29 16_25_01.164634",
        "2020-12-29 16_25_02.164653",
        "2020-12-29 16_25_03.164872",
        "2020-12-29 16_25_04.164961",
        "2020-12-29 16_25_05.164805",
    ]

    image = cv2.imread(read_img_masked)

    valid_rows = []

    for row in tqdm(sub_pred_df.itertuples(index=False), desc=subject):
        
        if row[0] in img_timestamps:
            valid_rows.append(row)


    for row in tqdm(sorted(valid_rows, key=lambda x : x[0]), desc=subject):
        
        if row[0] in img_timestamps:

            true_vals = np.array((row[1], row[2]))
            pred_vals = np.array((row[3], row[4]))
            # pred_vals = np.array((row[5], row[6])) # For Ridge

            true_vals = tuple(map(int, np.round(true_vals)))
            pred_vals = tuple(map(int, np.round(pred_vals)))

            true_past, pred_past = annotate_image_with_path(current_true=true_vals, past_true=true_past, current_pred=pred_vals, past_pred=pred_past, image=image)


            # print(read_img, write_img)
    cv2.imwrite(write_img, image)




