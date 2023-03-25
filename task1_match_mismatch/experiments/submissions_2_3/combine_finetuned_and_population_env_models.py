import glob
import os
import shutil

avg_baseline_files = glob.glob('./task1_match_mismatch/experiments/results_submission_1/heldout_predictions_avg_baseline/*.json')
avg_baseline_files = [x for x in avg_baseline_files if 'outputs' not in x]

finetuned_env_models = glob.glob('./task1_match_mismatch/experiments/results_submission_2/heldout_predictions_finetuned_env/*.json')
finetuned_env_models = [x for x in finetuned_env_models if 'outputs' not in x]
finetuned_basenames = [os.path.basename(x) for x in finetuned_env_models]

savedir = './task1_match_mismatch/experiments/results_submission_2/heldout_predictions_ft_and_baseline'

for avg_baseline_file in sorted(avg_baseline_files):

    avg_baseline_basename = os.path.basename(avg_baseline_file)
    if avg_baseline_basename in finetuned_basenames:
        continue

    shutil.copy(avg_baseline_file, savedir)

for finetuned_env_model in sorted(finetuned_env_models):
    shutil.copy(finetuned_env_model, savedir)