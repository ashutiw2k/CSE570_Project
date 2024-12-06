import pickle as pkl
import numpy as np
import json
print('---------------------------------------------')
with open('RAN4model_dfv4p4_root/RAN4model_dfv4p4/seqs/outdoor/scene3/20211007_102724/sync_ts16_dfv4p4/BBX5_sync_dfv4p4.pkl', 'rb') as file:
    bbx_file = pkl.load(file)

print(np.shape(bbx_file))

with open('RAN4model_dfv4p4_root/RAN4model_dfv4p4/seqs/outdoor/scene3/20211007_102724/RGB_ts16_dfv4p4_ls.json', 'rb') as jsonfile:
    rgb_file = json.load(jsonfile)

print(np.shape(rgb_file))

print('---------------------------------------------')
with open('RAN4model_dfv4p4_root/RAN4model_dfv4p4/seqs/outdoor/scene3/20211007_105139/sync_ts16_dfv4p4/BBX5_sync_dfv4p4.pkl', 'rb') as file:
    bbx_file = pkl.load(file)

print(np.shape(bbx_file))
with open('RAN4model_dfv4p4_root/RAN4model_dfv4p4/seqs/outdoor/scene3/20211007_105139/RGB_ts16_dfv4p4_ls.json', 'rb') as jsonfile:
    rgb_file = json.load(jsonfile)

print(np.shape(rgb_file))

print('---------------------------------------------')
with open('RAN4model_dfv4p4_root/RAN4model_dfv4p4/seqs/outdoor/scene3/20211007_111044/sync_ts16_dfv4p4/BBX5_sync_dfv4p4.pkl', 'rb') as file:
    bbx_file = pkl.load(file)

print(np.shape(bbx_file))


print('---------------------------------------------')
with open('RAN4model_dfv4p4_root/RAN4model_dfv4p4/seqs/outdoor/scene3/20211007_113544/sync_ts16_dfv4p4/BBX5_sync_dfv4p4.pkl', 'rb') as file:
    bbx_file = pkl.load(file)

print(np.shape(bbx_file))
