"""Example experiment for dilation model."""
import glob
import os
import tensorflow as tf
import numpy as np

from task1_match_mismatch.util.dataset_generator import MatchMismatchDataGenerator, default_batch_equalizer_fn, create_tf_dataset

def composite_model(models):

    eeg = tf.keras.layers.Input(shape=[512*3, 64])
    env1 = tf.keras.layers.Input(shape=[512*3, 1])
    env2 = tf.keras.layers.Input(shape=[512*3, 1])

    op = models[0]([eeg, env1, env2])

    if len(models) > 1:
        
        for i, model in enumerate(models[1:]):
            models[i+1]._name = f'model{i}'
            op += models[i+1]([eeg, env1, env2])

    op = op/len(models)

    final_model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[op])

    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=["acc"],
        loss=["binary_crossentropy"],
    )
    print(final_model.summary())
    return final_model

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
        np.savez(f'./task1_match_mismatch/experiments/results_submission_4/test_predictions_avg_ffr/{recording}_predictions.npz', predictions=predictions, targets=targets)

if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    window_length = 3 * 512  # 3 seconds
    # Hop length between two consecutive decision windows
    hop_length = 512
    # Number of samples (space) between end of matched speech and beginning of mismatched speech
    spacing = 512

    split='test'

    # Provide the path of the dataset
    # which is split already to train, val, test
    data_folder = './data/512Hz/split_data'

    # features!
    stimulus_features = ["mod"]
    stimulus_dimension = 1

    features = ["eeg"] + stimulus_features

    mdls = []

    for mdl in range(1, 31):

        mdls.append(
            tf.keras.models.load_model(os.path.join('./task1_match_mismatch/experiments/results_submission_1/trained_ffr_models/', f'model_ffr_{mdl}.h5'))
        )

    model = composite_model(mdls)
    del mdls

    # Evaluate the model on test set
    # Create a dataset generator for each recording
    test_files = [x for x in glob.glob(os.path.join(data_folder, f"{split}_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    rec_filenames = sorted(list(set(['_-_'.join(os.path.basename(x).split('_-_')[:-1]) for x in test_files])))

    datasets_test = {}
    # Create a generator for each recoring
    for rec in rec_filenames:
        files_test_rec = [f for f in test_files if rec in os.path.basename(f)]
        test_generator = MatchMismatchDataGenerator(files_test_rec, window_length, spacing=spacing)
        datasets_test[rec] = create_tf_dataset(test_generator, window_length, default_batch_equalizer_fn, hop_length, 1, feature_dims=(64, stimulus_dimension, stimulus_dimension))

    evaluation = get_predictions(model, datasets_test)