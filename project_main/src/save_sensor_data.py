from data import get_all_sequences_synced_dataset, get_scene_synced_datasets
import os
# import cv2
import numpy as np
import json

from torch.utils.data import ConcatDataset

WIFI_PATH = 'project_main/data/Wifi Json'
IMU_PATH = 'project_main/data/IMU Json'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    sequences = ['seq0']
    if len(sequences) != 0:
        with open('project_main/data/files_in_sequence.json', 'r') as f:
            sequence_data = json.load(f)
        
        sequence_paths = [val['sequence'] for seq, val in sequence_data.items() if seq in sequences]
        
        all_datasets = [get_scene_synced_datasets(sequence=seq) for seq in sequence_paths]
        datasets = [ConcatDataset(list(col)) for col in zip(*all_datasets)]

    else:
        datasets = get_all_sequences_synced_dataset()
    
    # datasets = get_scene_synced_datasets(full_path='Data/RAN4model_dfv4p4/seqs/indoor/scene0/20201223_140951')
    ctr = 0
    for dataset in datasets:
        wifi_ftm_dict = {} #9,10
        wifi_rssi_dict = {} #11,12
        imu19_dict = {} #5
        imuagm9_dict = {} #6
        imugq10_dict = {} #7
        imugqm13_dict = {} #8

        for data in dataset:
            timestamp = data[1]
            wifi_ftm_dict.update({timestamp: {'ftm': data[9], 'ftm_li': data[10]}})
            wifi_rssi_dict.update({timestamp: {'rssi': data[11], 'rssi_li': data[12]}})
            imu19_dict.update({timestamp:data[5]})
            imuagm9_dict.update({timestamp:data[6]})
            imugq10_dict.update({timestamp:data[7]})
            imugqm13_dict.update({timestamp:data[8]})

        
        subject_path = f'Subject{ctr}'
        
        with open(WIFI_PATH + '/' + subject_path + '/wifi_ftm.json', 'w') as f:
            json.dump(wifi_ftm_dict, f, cls=NpEncoder)
            f.close()
        with open(WIFI_PATH + '/' + subject_path + '/wifi_rssi.json', 'w') as f:
            json.dump(wifi_rssi_dict, f, cls=NpEncoder)
            f.close()
        with open(IMU_PATH + '/' + subject_path + '/imu19.json', 'w') as f:
            json.dump(imu19_dict, f, cls=NpEncoder)
            f.close()
        with open(IMU_PATH + '/' + subject_path + '/imuagm9.json', 'w') as f:
            json.dump(imuagm9_dict, f, cls=NpEncoder)
            f.close()
        with open(IMU_PATH + '/' + subject_path + '/imugq10.json', 'w') as f:
            json.dump(imugq10_dict, f, cls=NpEncoder)
            f.close()
        with open(IMU_PATH + '/' + subject_path + '/imugqm13.json', 'w') as f:
            json.dump(imugqm13_dict, f, cls=NpEncoder)
            f.close()

        ctr +=1

        

            
            
