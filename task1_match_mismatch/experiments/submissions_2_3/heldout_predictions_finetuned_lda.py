import glob
import os
import json
import pickle
from scipy.stats import pearsonr
import numpy as np

finetuned_env_files = sorted(glob.glob('./task1_match_mismatch/experiments/results_submission_2/heldout_predictions_finetuned_env/*.json'))
finetuned_ffr_files = sorted(glob.glob('./task1_match_mismatch/experiments/results_submission_3/heldout_predictions_finetuned_ffr/*.json'))

finetuned_env_files = [x for x in finetuned_env_files if 'outputs' not in x]
finetuned_ffr_files = [x for x in finetuned_ffr_files if 'outputs' not in x]

subjects = [os.path.basename(f).replace('.json', '') for f in finetuned_env_files]

for i in range(len(subjects)):

    sub_lda_results_dict = {}

    assert os.path.basename(finetuned_env_files[i]) == os.path.basename(finetuned_ffr_files[i])

    lda_model = pickle.load(
        open(f'./task1_match_mismatch/experiments/results_submission_3/finetuned_lda_models/lda-{subjects[i]}.pkl', 'rb')
    )

    subject_env_predictions = json.load(open(finetuned_env_files[i], 'r'))
    subject_ffr_predictions = json.load(open(finetuned_ffr_files[i], 'r'))

    all_env_preds = []
    all_ffr_preds = []

    for key in subject_env_predictions:

        inputs = [[
            subject_env_predictions[key],
            subject_ffr_predictions[key]
        ]]

        sub_lda_results_dict[key] = int(lda_model.predict(inputs).squeeze())
        #print(lda_model.predict(inputs))

        all_env_preds.append(subject_env_predictions[key])
        all_ffr_preds.append(subject_ffr_predictions[key])

    json.dump(
        sub_lda_results_dict,
        open(f'./task1_match_mismatch/experiments/results_submission_3/heldout_predictions_finetuned_lda/{subjects[i]}.json', 'w')
    )

    print(subjects[i], np.round(pearsonr(all_env_preds, all_ffr_preds)[0], 4), len(all_env_preds))