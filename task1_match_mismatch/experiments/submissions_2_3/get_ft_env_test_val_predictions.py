"""Example experiment for dilation model."""
import glob
import json
import os
import tensorflow as tf
import numpy as np

from task1_match_mismatch.util.dataset_generator import MatchMismatchDataGenerator, default_batch_equalizer_fn, create_tf_dataset


def get_predictions(model, test_dict):
    """Evaluate a model.
    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: dict
        Mapping between a subject and a tf.data.Dataset containing the test
        set for the subject.
    Returns
    -------
    dict
        Mapping between a recording and the loss/evaluation score on the test set
    """

    predictions = {}

    for recording, ds_test in test_dict.items():
        #for inputs, targets in ds_test:

        predictions = model.predict(ds_test).squeeze()
        data = list(ds_test.as_numpy_iterator())
        targets = np.array([datum[1].squeeze() for datum in data])
        targets = np.hstack(targets)

        binary_predictions = np.copy(predictions)
        binary_predictions[binary_predictions<0.5] = 0
        binary_predictions[binary_predictions>0.5] = 1

        acc = np.sum(targets == binary_predictions)/len(targets)

        print(f'Accuracy for recording {recording}:', acc)
        np.savez(f'./task1_match_mismatch/experiments/results_submission_3/predictions_finetuned_env_test_val/{recording}_predictions.npz', predictions=predictions, targets=targets)

if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    window_length = 3 * 64  # 3 seconds
    # Hop length between two consecutive decision windows
    hop_length = 16
    # Number of samples (space) between end of matched speech and beginning of mismatched speech
    spacing = 64

    split='test'

    # Provide the path of the dataset
    # which is split already to train, val, test
    data_folder = './data/split_data'

    # features!
    stimulus_features = ["envelope"]
    stimulus_dimension = 1
    features = ["eeg"] + stimulus_features

    # finetuned_models

    finetuned_env_models = glob.glob('./task1_match_mismatch/experiments/results_submission_2/heldout_predictions_finetuned_env/*.json')
    finetuned_env_models = [x for x in finetuned_env_models if 'outputs' not in x]
    finetuned_basenames = sorted([os.path.basename(x) for x in finetuned_env_models])
    print(finetuned_basenames)
    finetuned_subjects = [x.replace('.json','') for x in finetuned_basenames]

    for subject in sorted(finetuned_subjects):

        model = tf.keras.models.load_model(f'./task1_match_mismatch/experiments/results_submission_3/finetuned_env_models/finetuned_{subject}.h5')

        # Evaluate the model on test set
        # Create a dataset generator for each recording
        test_files = [x for x in glob.glob(os.path.join(data_folder, f"{split}_-_{subject}*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        rec_filenames = sorted(list(set(['_-_'.join(os.path.basename(x).split('_-_')[:-1]) for x in test_files])))

        datasets_test = {}
        # Create a generator for each recoring
        for rec in rec_filenames:
            files_test_rec = [f for f in test_files if rec in os.path.basename(f)]
            test_generator = MatchMismatchDataGenerator(files_test_rec, window_length, spacing=spacing)
            datasets_test[rec] = create_tf_dataset(test_generator, window_length, default_batch_equalizer_fn, hop_length, 1, feature_dims=(64, stimulus_dimension, stimulus_dimension))

        evaluation = get_predictions(model, datasets_test)