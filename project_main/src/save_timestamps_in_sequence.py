import json
import os

ALL_SEQUENCE_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/'
SEQUENCE_WRITE_PATH = 'project_main/data/'
IMAGE_PATH = 'RGB_anonymized/'
ctr = 0
write_dict = {}
for seq in sorted(os.listdir(ALL_SEQUENCE_PATH)):
    seq_path = ALL_SEQUENCE_PATH + seq + '/' + IMAGE_PATH
    write_dict.update({
        f'seq{ctr}': { 
            'sequence' : seq,
            'timestamps':['.'.join(p.split('.')[0:3]) for p in sorted(os.listdir(seq_path)) if p.endswith('.png')] }
    })
    ctr += 1

with open(SEQUENCE_WRITE_PATH+'files_in_sequence.json', 'w') as f:
    json.dump(write_dict, f)
