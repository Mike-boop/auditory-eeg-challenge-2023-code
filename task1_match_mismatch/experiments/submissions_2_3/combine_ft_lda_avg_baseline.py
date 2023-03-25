import shutil
import glob
import os

for file in glob.glob('./task1_match_mismatch/experiments/results_submission_3/heldout_predictions_finetuned_lda/*.json'):
    shutil.copy(file, './task1_match_mismatch/experiments/results_submission_3/heldout_predictions_ft_lda_avg_bl/')

for file in sorted(glob.glob('./task1_match_mismatch/experiments/results_submission_1/heldout_predictions_avg_baseline/*.json')):
    if 'outputs' in file:
        continue
    
    subject = int(os.path.basename(file).split('-')[1].replace('.json', ''))
    if subject > 71:
        print(subject)
        shutil.copy(
            file,
            './task1_match_mismatch/experiments/results_submission_3/heldout_predictions_ft_lda_avg_bl/'
            )


