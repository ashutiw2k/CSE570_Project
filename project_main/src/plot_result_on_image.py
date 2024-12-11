import pandas as pd
import os
import cv2

PREDICTIONS_DIR = 'project_main/predictions/'
PREDICTED_FRAMES = 'PredictedFrames/'
ANNOTATED_FRAMES = 'project_main/data/Annotated Images From Pickle/'
PRED_CSV = 'predictions.csv'

def annotate_image_with_centriods(true, pred, readpath, writepath):
    image = cv2.imread(readpath)


for subject in os.listdir(PREDICTIONS_DIR):
    sub_ann_frames = ANNOTATED_FRAMES + subject + '/'
    sub_out_frames = PREDICTIONS_DIR + subject + '/' + PREDICTED_FRAMES
    sub_csv = PREDICTIONS_DIR + subject + '/' + PRED_CSV

    # with open(sub_csv, 'r') as f:
    sub_pred_df = pd.read_csv(sub_csv)



