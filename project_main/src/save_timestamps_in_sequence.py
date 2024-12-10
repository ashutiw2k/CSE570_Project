import json
import os

ALL_SEQUENCE_PATH = 'Data/RAN4model_dfv4p4/seqs/indoor/scene0/'
SEQUENCE_WRITE_PATH = 'data/'

ctr = 0
write_dict = {}
for seq in os.listdir(ALL_SEQUENCE_PATH):
    seq_path = ALL_SEQUENCE_PATH + seq
    write_dict.update({
        f'seq{ctr}': list(os.listdir(seq_path))
    })
    ctr += 1

with open(SEQUENCE_WRITE_PATH+'files_in_sequence.json') as f:
    json.dump(write_dict, f)
