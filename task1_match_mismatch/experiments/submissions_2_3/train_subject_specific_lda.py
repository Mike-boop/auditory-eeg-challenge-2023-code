from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate
import numpy as np
import glob
from scipy.stats import pearsonr
import pickle

split = 'test'

ft_subjects = range(2, 72)
ft_subjects = [f'sub-{s:03d}' for s in ft_subjects]

subject_accs = []

for subject in ft_subjects:

    test_env_predictions_files = sorted(glob.glob(f'./task1_match_mismatch/experiments/results_submission_3/predictions_finetuned_env_test_val/{split}_-_{subject}*.npz'))
    test_ffr_predictions_path =  sorted(glob.glob(f'./task1_match_mismatch/experiments/results_submission_3/predictions_finetuned_ffr_test_val/{split}_-_{subject}*.npz'))

    assert len(test_env_predictions_files) == len(test_ffr_predictions_path)

    all_predictions = []
    all_targets = []

    for test_env_file in test_env_predictions_files:
        
        test_ffr_file = test_env_file.replace(
            'predictions_finetuned_env_test_val',
            'predictions_finetuned_ffr_test_val'
        )

        data_env = np.load(test_env_file)
        data_ffr = np.load(test_ffr_file)

        min_length = min(data_env['targets'].size, data_ffr['targets'].size)

        assert np.all(data_env['targets'][:min_length] == data_ffr['targets'][:min_length])
        targets = data_env['targets'][:min_length]
        all_targets.append(targets)

        predictions_env = data_env['predictions'][:min_length]
        predictions_ffr = data_ffr['predictions'][:min_length]

        predictions = np.vstack([predictions_env, predictions_ffr])
        all_predictions.append(predictions)
    
    predictions = np.hstack(all_predictions)
    targets = np.hstack(all_targets)

    cv_lda = LDA()

    cv_results = cross_validate(cv_lda, predictions.T, targets, cv=10)
    print(subject, np.round(np.mean(cv_results['test_score']),4), np.round(pearsonr(*predictions)[0], 4), len(targets))
    subject_accs.append(np.mean(cv_results['test_score']))

    if split == 'test':
        sub_lda = LDA()
        sub_lda.fit(predictions.T, targets)
        pickle.dump(sub_lda, open(f'./task1_match_mismatch/experiments/results_submission_3/finetuned_lda_models/lda-{subject}.pkl', 'wb'))


print('average', np.mean(subject_accs), 'std', np.std(subject_accs))
        