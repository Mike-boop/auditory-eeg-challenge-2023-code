import glob
import json
from scipy.stats import pearsonr
import os
import numpy as np

# sub2_files = sorted(glob.glob('./task1_match_mismatch/experiments/results_submission_1/heldout_predictions_lda/*.json'))
# sub4_files = sorted(glob.glob('./task1_match_mismatch/experiments/results_submission_4/heldout_predictions_lda/*.json'))

sub2_files = sorted(glob.glob('./task1_match_mismatch/experiments/results_submission_2/heldout_predictions_finetuned_env/*-outputs.json'))
sub4_files = sorted(glob.glob('./task1_match_mismatch/experiments/results_submission_4/heldout_predictions_finetuned_env/*-outputs.json'))

n_files = len(sub2_files)
assert n_files == len(sub4_files)

for i in range(n_files):
    sub2_f = sub2_files[i]
    sub4_f = sub4_files[i]

    assert os.path.basename(sub2_f) == os.path.basename(sub4_f)

    sub2_data = json.load(open(sub2_f, 'r'))
    sub4_data = json.load(open(sub4_f, 'r'))

    subject_sub2_outputs = []
    subject_sub4_outputs = []

    for k in sub2_data:
        subject_sub2_outputs.append(sub2_data[k])
        subject_sub4_outputs.append(sub4_data[k])
    
    # print(
    #     os.path.basename(sub2_f).replace('.json', ''),
    #     np.round(pearsonr(subject_sub2_outputs, subject_sub4_outputs)[0], 4)
    # )

    subject_sub2_preds = []
    subject_sub4_preds = []

    sub2_preds = json.load(open(sub2_f.replace('-outputs', ''), 'r'))
    sub4_preds = json.load(open(sub4_f.replace('-outputs', ''), 'r'))

    for k in sub2_preds:
        subject_sub2_preds.append(sub2_preds[k])
        subject_sub4_preds.append(sub4_preds[k])

    print(
        os.path.basename(sub2_f).replace('.json', '').replace('-outputs', '-agreement'),
        np.round(np.sum(np.array(subject_sub2_preds).astype(int) == np.array(subject_sub4_preds).astype(int))/len(subject_sub2_preds), 4)
        )