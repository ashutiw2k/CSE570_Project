from data_scene0 import get_scene0_synced_datasets
import os
import cv2
import json

# datasets = get_scene0_synced_datasets()
# dataset_sub_0 = datasets[0]
# dataset_sub_1 = datasets[1]
# dataset_sub_2 = datasets[2]
# dataset_sub_3 = datasets[3]
# dataset_sub_4 = datasets[4]

SCENE0_IMG_PATH = '20201223_140951-005/RGB_anonymized'
SCENE0_ANNOTATED_PATH = 'project_main/data/Annotated Images'
OUT_DIR_PATH = 'project_main/data/Annotated Images From Pickle'
# MAX_IMAGES = 10
COLORS = [
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (0,255,255),
    (255,255,0)
]

JSON_ALL_BB_TAGS = '20201223_140951-005/GND/vott-json-export/GND_20201223_140951-export.json'

def rescale_coordinates(x, y, original_width=2560, original_height=1440, target_width=1280, target_height=720):
    new_x = x * (target_width / original_width)
    new_y = y * (target_height / original_height)
    return new_x, new_y

def annotate_subject(dataset, image_path, out_path, color, all_assets):
    ctr = 0
    for data in dataset:
        img_name = data[1]+'.png'
        img = cv2.imread(image_path + '/' + img_name)   
    
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


def print_dict_tree(dictionary, indent=''):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            print_dict_tree(value, indent + '\t')
        else:
            print(f"{indent}{key}: {value}")



if __name__ == '__main__':

    datasets = get_scene0_synced_datasets()
    with open(JSON_ALL_BB_TAGS) as file:
        all_json = json.load(file)

    # all_json_str = json.loads(JSON_STR)
    # print(len(all_json['assets']))
    all_assets = list(all_json['assets'].values())
    # for asset in all_assets:
    #     print(asset['asset']['name'])

    # If asset.name = timestamp png,
    # Then for all region in regions, if subject in region.tags then get region.bounding box value
    ctr = 0
    for dataset in datasets:
        subject_out_path = OUT_DIR_PATH + '/' + f'Subject{ctr}'
        image_in_path = SCENE0_IMG_PATH

        annotate_subject(dataset, image_in_path, subject_out_path, COLORS[ctr], all_assets)

        ctr += 1


    
