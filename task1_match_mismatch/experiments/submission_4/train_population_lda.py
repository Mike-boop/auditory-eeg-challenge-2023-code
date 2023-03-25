import glob
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate
import pickle
from scipy.stats import pearsonr

def get_preds_targets_for_split(split='test', sub=None):

    env_filenames = sorted(glob.glob(f'./task1_match_mismatch/experiments/results_submission_1/test_predictions_avg_baseline/{split}*.npz'))
    ffr_filenames = sorted(glob.glob(f'./task1_match_mismatch/experiments/results_submission_4/test_predictions_avg_ffr/{split}*.npz'))
    
    if sub is not None:
        env_filenames = [f for f in env_filenames if sub in f]
        ffr_filenames = [f for f in env_filenames if sub in f]

    env_predictions = []
    ffr_predictions = []
    all_targets = []

    for i in range(len(env_filenames)):

        env_data = np.load(env_filenames[i])
        ffr_data = np.load(ffr_filenames[i])

        env_targets = env_data['targets']
        ffr_targets = ffr_data['targets']
        min_length = min(len(env_targets), len(ffr_targets))

        targets = env_targets[:min_length]
        assert np.all(targets == ffr_targets[:min_length])

        env_predictions.append(env_data['predictions'][:min_length])
        ffr_predictions.append(ffr_data['predictions'][:min_length])
        all_targets.append(targets)

    return np.vstack([np.concatenate(env_predictions), np.concatenate(ffr_predictions)]), np.concatenate(all_targets)

if __name__ == '__main__':

    test_predictions, test_targets = get_preds_targets_for_split('test')

    print('correlation between FFR/Env predictions (test): ', pearsonr(*test_predictions))
    #print('correlation between FFR/Env predictions (val): ', pearsonr(*val_predictions))

    ## cross validation within test split
    lda = LDA()
    print('running XV on test split...')
    cv_results = cross_validate(lda, test_predictions.T, test_targets, cv=10)
    print(cv_results['test_score'], np.mean(cv_results['test_score']))

    # ## cross validation within val split
    # lda = LDA()
    # print('running XV on val split...')
    # cv_results = cross_validate(lda, val_predictions.T, val_targets, cv=10)
    # print(cv_results['test_score'], np.mean(cv_results['test_score']))

    # ## training on val split, testing on test split
    # lda_val = LDA()
    # print('training on val split')
    # lda_val.fit(val_predictions.T, val_targets)
    # print('testing on test split')
    # score = lda_val.score(test_predictions.T, test_targets)
    # print('accuracy: ', score)

    # ## training on test split, testing on val split
    # lda = LDA()
    # print('training on test split')
    # lda.fit(test_predictions.T, test_targets)
    # print('testing on val split')
    # score = lda.score(val_predictions.T, val_targets)
    # print('accuracy: ', score)

    ## training final model
    lda = LDA()
    print('training on test split')
    predictions = test_predictions#np.hstack([test_predictions, val_predictions])
    targets = test_targets#np.hstack([test_targets, val_targets])
    lda.fit(predictions.T, targets)
    print('saving final model fitted_pop_lda.pkl')

    pickle.dump(lda, open('./task1_match_mismatch/experiments/results_submission_4/fitted_pop_lda.pkl', 'wb'))

    # # estimate actual score

    # subjects = [f'sub-{i:03d}' for i in range(1, 72)]
    # sub_accs = []

    # for sub in subjects:

    #     test_predictions, test_targets = get_preds_targets_for_split('test', sub)
    #     acc = lda_val.score(test_predictions.T, test_targets)
    #     print(sub, acc)

    #     sub_accs.append(acc)
    
    # print('estimated_score:', np.mean(sub_accs))