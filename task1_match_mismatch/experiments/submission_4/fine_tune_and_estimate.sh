source ~/miniconda3/etc/profile.d/conda.sh
conda activate auditory-eeg

cd ~/Code/auditory-eeg-challenge-2023-code/

# python task1_match_mismatch/experiments/submission_4/predict_avg_ffr_model.py
# python task1_match_mismatch/experiments/submission_4/average_ffr_outputs.py
python task1_match_mismatch/experiments/submission_4/fine_tune_env_models.py