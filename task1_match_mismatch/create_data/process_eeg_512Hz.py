import glob
import os
import pickle
from mne.filter import resample, filter_data
import numpy as np

data_root_dir = './data'
folders = ['raw_eeg_1_to_15',
           'raw_eeg_16_to_30',
           'raw_eeg_31_to_45',
           'raw_eeg_46_to_60',
           'raw_eeg_61_to_71'
           ]


for folder in folders:
    subject_folders = glob.glob(
        os.path.join(data_root_dir, folder, 'sub-*')
    )

    for subject_folder in subject_folders:
        session = glob.glob(os.path.join(subject_folder, 'ses-*'))[0]
        
        if 'sub-009' in subject_folder:
            session = glob.glob(os.path.join(subject_folder, 'ses-shortstories01*'))[0]
            
        files = glob.glob(os.path.join(session, '*.pkl'))

        session_name = session.split('/')[-1]

        for file in files:

            data_dict = pickle.load(open(file, 'rb'))

            processed_eeg = resample(data_dict['eeg'].astype(float).T, up=1.0, down=2)
            fs = data_dict['fs'] / 2
            assert fs == 512.0

            processed_eeg = filter_data(processed_eeg, fs, 70, 220)

            raw_fname = file.split('/')[-1]

            proc_fpath = os.path.join(data_root_dir, '500Hz', 'preprocessed_eeg', data_dict['subject'], session_name)
            proc_fname = raw_fname.replace('raw_eeg', 'preproc_eeg')
            os.makedirs(proc_fpath, exist_ok=True)

            data_dict['eeg'] = processed_eeg.astype(np.float32).T
            data_dict['fs'] = fs

            with open(os.path.join(proc_fpath, proc_fname), 'wb') as fp:

                pickle.dump(data_dict, fp)