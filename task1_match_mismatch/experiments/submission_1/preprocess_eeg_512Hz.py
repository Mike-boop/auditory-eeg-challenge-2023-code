import glob
import os
import pickle
from mne.filter import resample, filter_data
import numpy as np
import json

def process_eeg(eeg):
    '''
    eeg: (T, 64) data at 1024Hz sr
    '''

    #import pdb;pdb.set_trace()

    processed_eeg = resample(eeg.astype(float).T, up=1.0, down=2)
    fs = 512.0

    processed_eeg = filter_data(processed_eeg, fs, 70, 220)
    processed_eeg = processed_eeg.T

    mean = np.mean(processed_eeg, axis=0)
    std = np.std(processed_eeg, axis=0)

    processed_eeg = (processed_eeg - mean)/std

    return processed_eeg

raw_eeg_files = glob.glob('./data/test-task1/raw_eeg/*.json')

for file in raw_eeg_files:

    fname = os.path.basename(file)
    save_path = './data/test-task1/eeg_512Hz/'+fname

    raw_data = json.load(open(file, 'r'))
    for k in raw_data:
        raw_data[k][0] = process_eeg(np.asarray(raw_data[k][0])).tolist()

    json.dump(raw_data, open(save_path, 'w'))