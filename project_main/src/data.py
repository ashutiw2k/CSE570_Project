import datetime, pytz
import os

import pickle as pkl
from typing import Literal
import numpy as np
import json

# import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import pytz.zoneinfo

import pytz.zoneinfo.Europe
# from datasets import Dataset, 

# Aditya may need to modify this variable. 
DATA_ROOT = 'Data/RAN4model_dfv4p4'
SEQ_INDOOR = 'seqs/indoor'
SEQ_OUTDOOR = 'seqs/outdoor'
SCENE0 = 'scene0'

# Data File Path
BBX5H_SYNC_PATH = 'sync_ts16_dfv4p4/BBX5H_sync_dfv4p4.pkl'
BBXC3H_SYNC_PATH = 'sync_ts16_dfv4p4/BBXC3H_sync_dfv4p4.pkl'
FTM_LI_SYNC_PATH = 'sync_ts16_dfv4p4/FTM_li_sync_dfv4p4.pkl'
FTM_SYNC_PATH = 'sync_ts16_dfv4p4/FTM_sync_dfv4p4.pkl'
IMU_19_SYNC_PATH = 'sync_ts16_dfv4p4/IMU19_sync_dfv4p4.pkl'
IMU_AGM9_SYNC_PATH = 'sync_ts16_dfv4p4/IMUagm9_sync_dfv4p4.pkl'
IMU_GQ10_SYNC_PATH = 'sync_ts16_dfv4p4/IMUlgq10_sync_dfv4p4.pkl'
IMU_GQM13_SYNC_PATH = 'sync_ts16_dfv4p4/IMUlgqm13_sync_dfv4p4.pkl'
RSSI_LI_SYNC_PATH = 'sync_ts16_dfv4p4/RSSI_li_sync_dfv4p4.pkl'
RSSI_SYNC_PATH = 'sync_ts16_dfv4p4/RSSI_sync_dfv4p4.pkl'

JSON_FILE_PATH = 'RGBg_ts16_dfv4p4_ls.json'

# TIMEZONE = pytz.UTC

class SyncedDataSet(Dataset):
    '''
    NOTE: Make a synced dataset PER SUBJECT.  
    '''
    def __init__(self, timestamps, bbx5h, bbxc3h, \
                 ftm_li, ftm, \
                 imu19, imuagm9, imugq10, imugqm13, \
                 rssi_li, rssi):
        
        self.timestamps = timestamps
        self.readable_date = []
        for timestamp in self.timestamps:
            dt_object = datetime.datetime.fromtimestamp(float(timestamp), tz=pytz.UTC)
            self.readable_date.append(dt_object.strftime('%Y-%m-%d %H_%M_%S.%f'))

        self.bbx5h = bbx5h
        self.bbxc3h = bbxc3h
        self.ftm_li = ftm_li
        self.ftm = ftm
        self.imu19 = imu19
        self.imuagm9 = imuagm9
        self.imugq10 = imugq10
        self.imugqm13 = imugqm13
        self.rssi_li = rssi_li
        self.rssi = rssi

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
                self.bbx5h[idx], self.bbxc3h[idx], \
                self.ftm[idx], self.ftm_li[idx], \
                self.imu19[idx], self.imuagm9[idx], self.imugq10[idx], self.imugqm13[idx], \
                self.rssi[idx], self.rssi_li[idx])


def load_data_from_path(path:str, file_type:Literal['pickle','json']='pickle'):
    with open(path, 'rb') as file:
        if file_type == 'pickle':
            data = pkl.load(file)
        elif file_type == 'json':
            data= json.load(file)
    data = np.array(data)
    return data.reshape((data.shape[0], data.shape[1], data.shape[3]))

def get_scene_synced_datasets(sequence=None, is_indoor=True, scene=None, full_path=None):
    '''
    Dataset at index 'i' corresponds to subject 'i'
    '''
    if scene is None:
        scene=SCENE0
    if is_indoor:
        loc = SEQ_INDOOR
    else:
        loc = SEQ_OUTDOOR

    if sequence is None and full_path is None:
        raise ValueError('Please provide the sequence or the full path of the data')
    
    if full_path is None:
        full_path = DATA_ROOT + '/' + loc + '/' + scene + '/' + sequence

    bbx5h_data = load_data_from_path(full_path + '/' + BBX5H_SYNC_PATH)
    bbxc3h_data = load_data_from_path(full_path + '/' + BBXC3H_SYNC_PATH)
    ftm_li_data = load_data_from_path(full_path + '/' + FTM_LI_SYNC_PATH)
    ftm_data = load_data_from_path(full_path + '/' + FTM_SYNC_PATH)
    imu19_data = load_data_from_path(full_path + '/' + IMU_19_SYNC_PATH)
    imuagm9_data = load_data_from_path(full_path + '/' + IMU_AGM9_SYNC_PATH)
    imugq10_data = load_data_from_path(full_path + '/' + IMU_GQ10_SYNC_PATH)
    imugqm13_data = load_data_from_path(full_path + '/' + IMU_GQM13_SYNC_PATH)
    rssi_li_data = load_data_from_path(full_path + '/' + RSSI_LI_SYNC_PATH)
    rssi_data = load_data_from_path(full_path + '/' + RSSI_SYNC_PATH)

    with open(full_path + '/' + JSON_FILE_PATH) as file:
        json_data = json.load(file)

    # print(bbx5h_data.shape, bbxc3h_data.shape, ftm_data.shape, ftm_li_data.shape, \
    #       imu19_data.shape, imuagm9_data.shape, imugq10_data.shape, imugqm13_data.shape, \
    #         rssi_data.shape, rssi_li_data.shape, sep=', ')

    dataset_list = []
    for i in range(bbx5h_data.shape[1]):
        sub_dataset = SyncedDataSet(
            timestamps=json_data, \
            bbx5h=bbx5h_data[:,i,:], bbxc3h=bbxc3h_data[:,i,:], \
            ftm=ftm_data[:,i,:], ftm_li=ftm_li_data[:,i,:], \
            imu19=imu19_data[:,i,:], imuagm9=imuagm9_data[:,i,:], imugq10=imugq10_data[:,i,:], imugqm13=imugqm13_data[:,i,:], \
            rssi=rssi_data[:,i,:], rssi_li=rssi_li_data[:,i,:]
        )
        dataset_list.append(sub_dataset)
    
    return dataset_list

def get_all_sequences_synced_dataset(is_indoor=True, scene=SCENE0, data_root = DATA_ROOT):
    
    if is_indoor:
        loc = SEQ_INDOOR
    else:
        loc = SEQ_OUTDOOR

    sequence_path = data_root + '/' + loc + '/' + scene + '/'
    all_datasets = []
    for item in os.listdir(sequence_path):
        # print(item)
        full_path = os.path.join(sequence_path, item)
        print(full_path)
        if os.path.isdir(full_path):
            datasets = get_scene_synced_datasets(full_path=full_path)
            all_datasets.append(datasets)
        
    subject_dataset = [ConcatDataset(list(col)) for col in zip(*all_datasets)]

    return subject_dataset


if __name__ == '__main__':
    # print(load_data_from_path(SCENE0_BBX5H_SYNC_PATH).shape)
    datasets = get_all_sequences_synced_dataset()

    print(len(datasets))
    for d in datasets:
        print(len(d))
        print(d[0])
    

    
    






# def get_torch_dataset_scene0():