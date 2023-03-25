import glob
import pickle
import json
import numpy as np
from scipy.stats import pearsonr
import os

lda = pickle.load(open('./task1_match_mismatch/experiments/results_submission_1/lda_models/fitted_pop_lda.pkl', 'rb'))

ffr_prediction_files = glob.glob('./task1_match_mismatch/experiments/results_submission_1/heldout_predictions_avg_ffr/*-outputs.json')
env_prediction_files = [f.replace('heldout_predictions_avg_ffr', 'heldout_predictions_avg_baseline') for f in ffr_prediction_files]

ffr_outputs = []
env_outputs = []

for i in range(len(ffr_prediction_files)):

    ffr_data = json.load(open(ffr_prediction_files[i], 'r'))
    env_data = json.load(open(env_prediction_files[i], 'r'))

    subject = os.path.basename(ffr_prediction_files[i]).replace('-outputs.json', '')

    subject_results = {}

    for k in ffr_data:

        ffr_prediction = ffr_data[k]
        env_prediction = env_data[k]

        input = np.array([[env_prediction, ffr_prediction]])
        output = lda.predict(input)[0]

        subject_results[k] = int(output)

        ffr_outputs.append(ffr_prediction)
        env_outputs.append(env_prediction)

    json.dump(subject_results, open(f'./task1_match_mismatch/experiments/results_submission_1/heldout_predictions_lda/{subject}.json', 'w'))

print(pearsonr(ffr_outputs, env_outputs))