
source ~/miniconda3/etc/profile.d/conda.sh
conda activate auditory-eeg

cd ~/Code/auditory-eeg-challenge-2023-code/

python task1_match_mismatch/experiments/submission_3/fine_tune_env_models.py
python task1_match_mismatch/experiments/submission_3/fine_tune_ffr_models.py