import pickle as pkl
from typing import Literal
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader
# from datasets import Dataset, 

SCENE0_BBX5H_SYNC_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/BBX5H_sync_dfv4p4.pkl'
SCENE0_BBXC3H_SYNC_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/BBXC3H_sync_dfv4p4.pkl'
SCENE0_FTM_LI_SYNC_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/FTM_li_sync_dfv4p4.pkl'
SCENE0_FTM_SYNC_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/FTM_sync_dfv4p4.pkl'
SCENE0_IMU_19_SYNC_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/IMU19_sync_dfv4p4.pkl'
SCENE0_IMU_AGM9_SYNC_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/IMUagm9_sync_dfv4p4.pkl'
SCENE0_IMU_GQ10_SYNC_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/IMUlgq10_sync_dfv4p4.pkl'
SCENE0_IMU_GQ13_SYNC_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/IMUlgq13_sync_dfv4p4.pkl'
SCENE0_RSSI_LI_SYNC_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/RSSI_li_sync_dfv4p4.pkl'
SCENE0_RSSI_SYNC_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951/sync_ts16_dfv4p4/RSSI_sync_dfv4p4.pkl'

class SyncedDataSet(Dataset):
    '''
    NOTE: Make a synced dataset PER SUBJECT. 
    '''
    def __init__(self, bbx5h, bbxc3h, \
                 ftm_li, ftm, \
                 imu19, imuagm9, imugq10, imugq13, \
                 rssi_li, rssi):
        
        self.bbx5h = bbx5h
        self.bbxc3h = bbxc3h
        self.ftm_li = ftm_li
        self.ftm = ftm
        self.imu19 = imu19
        self.imuagm9 = imuagm9
        self.imugq10 = imugq10
        self.imugq13 = imugq13
        self.rssi_li = rssi_li
        self.rssi = rssi
        

    def __len__(self):
        return len(self.bbx5h)

    def __getitem__(self, idx):
        return ((self.bbx5h[idx], self.bbxc3h[idx]), \
                (self.ftm[idx], self.ftm_li[idx]), \
                (self.imu19[idx], self.imuagm9[idx], self.imugq10[idx], self.imugq13[idx]), \
                (self.rssi[idx], self.rssi_li[idx]))


def load_data_from_path(path:str, file_type:Literal['pickle','json']='pickle'):
    with open(path, 'rb') as file:
        if file_type == 'pickle':
            data = pkl.load(file)
        elif file_type == 'json':
            data= json.load(file)
    data = np.array(data)
    return data.reshape((data.shape[0], data.shape[1], data.shape[3]))

if __name__ == '__main__':
    # print(load_data_from_path(SCENE0_BBX5H_SYNC_PATH).shape)
    data = load_data_from_path(SCENE0_BBX5H_SYNC_PATH)
    print(data.shape)
    print(data[1])




# def get_torch_dataset_scene0():