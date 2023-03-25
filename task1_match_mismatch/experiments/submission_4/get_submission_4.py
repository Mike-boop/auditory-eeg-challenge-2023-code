import shutil
import os
import glob

finetuned_dir = './task1_match_mismatch/experiments/results_submission_4/heldout_predictions_finetuned_env'
pop_dir = './task1_match_mismatch/experiments/results_submission_4/heldout_predictions_lda'
submission_dir = './task1_match_mismatch/experiments/results_submission_4/combined_heldout_predictions'

finetuned_files = glob.glob(os.path.join(finetuned_dir, '*.json'))
finetuned_files = [x for x in finetuned_files if not 'outputs' in x]
finetuned_basenames = [os.path.basename(x) for x in finetuned_files]

for ft_file in finetuned_files:
    shutil.copy(ft_file, submission_dir)

for pop_file in sorted(glob.glob(os.path.join(pop_dir, '*.json'))):

    if os.path.basename(pop_file) in finetuned_basenames:
        print('skipping', pop_file)
    
    else:
        print('copying', pop_file)
        shutil.copy(pop_file, submission_dir)