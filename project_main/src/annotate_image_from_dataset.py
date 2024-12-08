from data import get_all_sequences_synced_dataset
import os
import cv2
import json
from tqdm import tqdm

# datasets = get_scene0_synced_datasets()
# dataset_sub_0 = datasets[0]
# dataset_sub_1 = datasets[1]
# dataset_sub_2 = datasets[2]
# dataset_sub_3 = datasets[3]
# dataset_sub_4 = datasets[4]

DATA_ROOT = 'Data/RAN4model_dfv4p4'
SEQ_INDOOR = 'seqs/indoor'
SEQ_OUTDOOR = 'seqs/outdoor'
SCENE0 = 'scene0'

IMG_PATH = 'Data/AllRawImage'
ANNOTATED_PATH = 'project_main/data/Annotated Images'
OUT_DIR_PATH = 'project_main/data/Annotated Images From Pickle'
# MAX_IMAGES = 10
COLORS = [
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (0,255,255),
    (255,255,0)
]

JSON_ALL_BB_TAGS = 'GND-export.json'


def annotate_subject(dataset, image_path, out_path, color):
    ctr = 0
    subject = out_path.split('/')[-1]
    for data in tqdm(dataset, desc=f'Dataset for {subject}'):
        img_name = data[1]+'.png'
        try:
            img = cv2.imread(image_path + '/' + img_name)   
        except:
            continue
    
        # print('Image JSON does not contain bounding box values')
        x, y, w, h = map(int, tuple(data[2]))
        
        # x,y = rescale_coordinates(x,y)
        # w,h = rescale_coordinates(w,h)
        # print(img.shape)
        # print((x,y))
        # print((x+w,y+h))

        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.imwrite(out_path+'/'+img_name, img)
        # ctr += 1
        # if ctr > MAX_IMAGES:
        #     break



if __name__ == '__main__':

    datasets = get_all_sequences_synced_dataset()

    ctr = 0

    for dataset in datasets:
        subject_out_path = OUT_DIR_PATH + '/' + f'Subject{ctr}'

        annotate_subject(dataset, IMG_PATH, subject_out_path, COLORS[ctr])

        ctr += 1


    
