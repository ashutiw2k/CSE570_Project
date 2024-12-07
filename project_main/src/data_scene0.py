import pickle as pkl
from typing import Literal
import numpy as np
import json

# import torch
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

class SyncedDataSet(Dataset):
    '''
    NOTE: Make a synced dataset PER SUBJECT.  
    '''
    def __init__(self, bbx5h, bbxc3h, \
                 ftm_li, ftm, \
                 imu19, imuagm9, imugq10, imugqm13, \
                 rssi_li, rssi):
        
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
        return (self.bbx5h[idx], self.bbxc3h[idx], \
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

    # print(bbx5h_data.shape, bbxc3h_data.shape, ftm_data.shape, ftm_li_data.shape, \
    #       imu19_data.shape, imuagm9_data.shape, imugq10_data.shape, imugqm13_data.shape, \
    #         rssi_data.shape, rssi_li_data.shape, sep=', ')

    dataset_list = []
    for i in range(bbx5h_data.shape[1]):
        sub_dataset = SyncedDataSet(
            bbx5h=bbx5h_data[:,i,:], bbxc3h=bbxc3h_data[:,i,:], \
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

    # print(bbx5h_data.shape, bbxc3h_data.shape, ftm_data.shape, ftm_li_data.shape, \
    #       imu19_data.shape, imuagm9_data.shape, imugq10_data.shape, imugqm13_data.shape, \
    #         rssi_data.shape, rssi_li_data.shape, sep=', ')

    dataloader_list = []
    for i in range(bbx5h_data.shape[1]):
        sub_dataset = SyncedDataSet(
            bbx5h=bbx5h_data[:,i,:], bbxc3h=bbxc3h_data[:,i,:], \
            ftm=ftm_data[:,i,:], ftm_li=ftm_li_data[:,i,:], \
            imu19=imu19_data[:,i,:], imuagm9=imuagm9_data[:,i,:], imugq10=imugq10_data[:,i,:], imugqm13=imugqm13_data[:,i,:], \
            rssi=rssi_data[:,i,:], rssi_li=rssi_li_data[:,i,:]
        )
        dataloader_list.append(DataLoader(sub_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers))
    
    return dataloader_list


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
        ds = SyncedDataSet(np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10),np.arange(10))
    except:
        print('Not same shape test failed')

    dataloaders = get_scene0_synced_dataloaders(batch_size=1)
    
    for dataloader in dataloaders:
        print(len(dataloader))

    # print(dataloader)
    
    






# def get_torch_dataset_scene0():