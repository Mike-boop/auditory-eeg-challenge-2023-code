source ~/miniconda3/etc/profile.d/conda.sh
conda activate auditory-eeg

for i in {51..100}

    do
        python task1_match_mismatch/experiments/dilated_convolutional_model.py
        cp "task1_match_mismatch/experiments/results_dilated_convolutional_model/eval.json" "task1_match_mismatch/experiments/results_dilated_convolutional_model/eval_baseline_${i}.json"
        cp "task1_match_mismatch/experiments/results_dilated_convolutional_model/model_env_baseline.h5" "task1_match_mismatch/experiments/results_dilated_convolutional_model/model_env_baseline_${i}.h5" 

    done

for i in {31..100}

    do
        python task1_match_mismatch/experiments/ffr_dilated_convolutional_model.py
        cp "task1_match_mismatch/experiments/results_dilated_convolutional_model/eval_ffr.json" "task1_match_mismatch/experiments/results_submission_1/eval_ffr_${i}.json"
        cp "task1_match_mismatch/experiments/results_dilated_convolutional_model/model_ffr.h5" "task1_match_mismatch/experiments/results_submission_1/model_ffr_${i}.h5" 

    done