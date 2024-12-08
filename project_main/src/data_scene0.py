import datetime, pytz
import pickle as pkl
from typing import Literal
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader
# from datasets import Dataset, 

# Aditya may need to modify this variable. 
DATA_ROOT = 'Data/RAN4model_dfv4p4'

# Data File Path
SCENE0_BBX5H_SYNC_PATH = 'seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/BBX5H_sync_dfv4p4.pkl'
SCENE0_BBXC3H_SYNC_PATH = 'seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/BBXC3H_sync_dfv4p4.pkl'
SCENE0_FTM_LI_SYNC_PATH = 'seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/FTM_li_sync_dfv4p4.pkl'
SCENE0_FTM_SYNC_PATH = 'seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/FTM_sync_dfv4p4.pkl'
SCENE0_IMU_19_SYNC_PATH = 'seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/IMU19_sync_dfv4p4.pkl'
SCENE0_IMU_AGM9_SYNC_PATH = 'seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/IMUagm9_sync_dfv4p4.pkl'
SCENE0_IMU_GQ10_SYNC_PATH = 'seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/IMUlgq10_sync_dfv4p4.pkl'
SCENE0_IMU_GQM13_SYNC_PATH = 'seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/IMUlgqm13_sync_dfv4p4.pkl'
SCENE0_RSSI_LI_SYNC_PATH = 'seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/RSSI_li_sync_dfv4p4.pkl'
SCENE0_RSSI_SYNC_PATH = 'seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/RSSI_sync_dfv4p4.pkl'

SCENE0_JSON_FILE_PATH = 'seqs/indoor/scene0/20201223_140951/RGBg_ts16_dfv4p4_ls.json'
JSON_ALL_BB_TAGS = '20201223_140951-005/GND/vott-json-export/GND_20201223_140951-export.json'

with open(JSON_ALL_BB_TAGS) as file:
    ALL_JSON = json.load(file)
    ALL_ASSETS = list(ALL_JSON['assets'].values())

class SyncedDataSet(Dataset):
    '''
    NOTE: Make a synced dataset PER SUBJECT.  
    '''
    def __init__(self, subject_id, timestamps, \
                 bbx5h, bbxc3h, bb_json, \
                 ftm_li, ftm, \
                 imu19, imuagm9, imugq10, imugqm13, \
                 rssi_li, rssi):
        self.timestamps = timestamps
        self.readable_date = []
        self.subject_id = subject_id

        for timestamp in self.timestamps:
            dt_object = datetime.datetime.fromtimestamp(float(timestamp))
            self.readable_date.append(dt_object.strftime('%Y-%m-%d %H_%M_%S.%f'))


        self.bbx5h = np.nan_to_num(bbx5h, nan=0.0)
        self.bbxc3h =  np.nan_to_num(bbxc3h, nan=0.0)
        self.bb_json = np.array(bb_json)
        self.ftm_li =  np.nan_to_num(ftm_li, nan=0.0)
        self.ftm =  np.nan_to_num(ftm, nan=0.0)
        self.imu19 =  np.nan_to_num(imu19, nan=0.0)
        self.imuagm9 =  np.nan_to_num(imuagm9, nan=0.0)
        self.imugq10 =  np.nan_to_num(imugq10, nan=0.0)
        self.imugqm13 =  np.nan_to_num(imugqm13, nan=0.0)
        self.rssi_li =  np.nan_to_num(rssi_li, nan=0.0)
        self.rssi =  np.nan_to_num(rssi, nan=0.0)

        # class_arrs = [v for k,v in vars(self).items() if isinstance(v, np.ndarray)]
        # shapes = [v.shape for v in class_arrs]
        # print(shapes)

    def __len__(self):
        return len(self.bbx5h)

    def __get_sep_item__(self, idx):
        return ((self.bbx5h[idx], self.bbxc3h[idx]), \
                (self.ftm[idx], self.ftm_li[idx]), \
                (self.imu19[idx], self.imuagm9[idx], self.imugq10[idx], self.imugqm13[idx]), \
                (self.rssi[idx], self.rssi_li[idx]))
    
    def __getitem__(self, idx):
        return (self.timestamps[idx], self.readable_date[idx], \
                self.bb_json[idx], self.bbx5h[idx], self.bbxc3h[idx], \
                self.imu19[idx], self.imuagm9[idx], self.imugq10[idx], self.imugqm13[idx], \
                self.ftm[idx], self.ftm_li[idx], \
                self.rssi[idx], self.rssi_li[idx])


def get_bounding_box_for_timestamp(timestamp, assets, subject):
    image_name = datetime.datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S.%f').replace(' ', '%20') + '.png'
    bounding_box = (0,0,0,0)
    for asset in assets:
        if asset['asset']['name'] == image_name:
            regions = asset['regions']
            for region in regions:
                if subject in region['tags']:
                    bounding_box = (region['boundingBox']['left'], region['boundingBox']['top'], region['boundingBox']['width'], region['boundingBox']['height'])
                    break
            break
    return bounding_box

def load_data_from_path(path:str, file_type:Literal['pickle','json']='pickle'):
    with open(path, 'rb') as file:
        if file_type == 'pickle':
            data = pkl.load(file)
        elif file_type == 'json':
            data= json.load(file)
    data = np.array(data)
    return data.reshape((data.shape[0], data.shape[1], data.shape[3]))

def get_scene0_synced_datasets():
    '''
    Dataset at index 'i' corresponds to subject 'i'
    '''
    bbx5h_data = load_data_from_path(DATA_ROOT + '/' + SCENE0_BBX5H_SYNC_PATH)
    bbxc3h_data = load_data_from_path(DATA_ROOT + '/' + SCENE0_BBXC3H_SYNC_PATH)
    ftm_li_data = load_data_from_path(DATA_ROOT + '/' + SCENE0_FTM_LI_SYNC_PATH)
    ftm_data = load_data_from_path(DATA_ROOT + '/' + SCENE0_FTM_SYNC_PATH)
    imu19_data = load_data_from_path(DATA_ROOT + '/' + SCENE0_IMU_19_SYNC_PATH)
    imuagm9_data = load_data_from_path(DATA_ROOT + '/' + SCENE0_IMU_AGM9_SYNC_PATH)
    imugq10_data = load_data_from_path(DATA_ROOT + '/' + SCENE0_IMU_GQ10_SYNC_PATH)
    imugqm13_data = load_data_from_path(DATA_ROOT + '/' + SCENE0_IMU_GQM13_SYNC_PATH)
    rssi_li_data = load_data_from_path(DATA_ROOT + '/' + SCENE0_RSSI_LI_SYNC_PATH)
    rssi_data = load_data_from_path(DATA_ROOT + '/' + SCENE0_RSSI_SYNC_PATH)
    
    with open(DATA_ROOT + '/' + SCENE0_JSON_FILE_PATH) as file:
        json_data = json.load(file)

    with open(JSON_ALL_BB_TAGS) as file:
        all_json_assets = json.load(file)
        all_assets = list(all_json_assets['assets'].values())

    # print(bbx5h_data.shape, bbxc3h_data.shape, ftm_data.shape, ftm_li_data.shape, \
    #       imu19_data.shape, imuagm9_data.shape, imugq10_data.shape, imugqm13_data.shape, \
    #         rssi_data.shape, rssi_li_data.shape, sep=', ')
    # str_timestamps = [datetime.datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H_%M_%S.%f') for timestamp in json_data]
    dataset_list = []
    for i in range(bbx5h_data.shape[1]):
        subject_id = f'Subject{i+1}'
        bb_from_json = [get_bounding_box_for_timestamp(t, all_assets, subject_id) for t in json_data]
        sub_dataset = SyncedDataSet(
            subject_id=i,\
            timestamps=json_data, \
            bb_json = bb_from_json, bbx5h=bbx5h_data[:,i,:], bbxc3h=bbxc3h_data[:,i,:], \
            ftm=ftm_data[:,i,:], ftm_li=ftm_li_data[:,i,:], \
            imu19=imu19_data[:,i,:], imuagm9=imuagm9_data[:,i,:], imugq10=imugq10_data[:,i,:], imugqm13=imugqm13_data[:,i,:], \
            rssi=rssi_data[:,i,:], rssi_li=rssi_li_data[:,i,:]
        )
        dataset_list.append(sub_dataset)
    
    return dataset_list


def get_scene0_synced_dataloaders(batch_size=4,shuffle=True, num_workers=0):
    '''
    Dataloader at index [i] corresponds to subject [i]
    '''
    # dataloader_list = []
    dataset_list = get_scene0_synced_datasets()
    
    return dataset_list


if __name__ == '__main__':
    # print(load_data_from_path(SCENE0_BBX5H_SYNC_PATH).shape)
    data = load_data_from_path(DATA_ROOT + '/' + SCENE0_BBX5H_SYNC_PATH)
    print(data.shape)
    print(data[1])
    split_data = np.array([data[:, i, :] for i in range(data.shape[1])])
    print(split_data.shape)
    print(split_data[3][1])

    try:
        ds = SyncedDataSet(np.arange(10), np.arange(10))
    except:
        print('Not same shape test passed')
    try:
        ds = SyncedDataSet(np.arange(10), np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10))
    except:
        print('Not same shape test failed')

    dataloaders = get_scene0_synced_dataloaders(batch_size=1)
    
    for dataloader in dataloaders:
        print(len(dataloader))
        print(dataloader[0])

    # print(dataloader)
    
    






# def get_torch_dataset_scene0():