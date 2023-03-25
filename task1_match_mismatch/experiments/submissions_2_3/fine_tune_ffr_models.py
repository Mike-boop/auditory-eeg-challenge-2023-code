"""Example experiment for dilation model."""
import glob
import json
import logging
import os
import tensorflow as tf

from task1_match_mismatch.models.regularised_dilated_convolutional_model import dilation_model
from task1_match_mismatch.util.dataset_generator import MatchMismatchDataGenerator, default_batch_equalizer_fn, create_tf_dataset

# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def evaluate_model(model, test_dict):
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
        Mapping between a subject and the loss/evaluation score on the test set
    """
    evaluation = {}
    for subject, ds_test in test_dict.items():
        logging.info(f"Scores for subject {subject}:")
        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))
    return evaluation

if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    window_length = 3 * 512  # 3 seconds
    # Hop length between two consecutive decision windows
    hop_length = 512
    train_hop_length = 64
    # Number of samples (space) between end of matched speech and beginning of mismatched speech
    spacing = 512
    epochs = 100
    patience = 5
    batch_size = 64
    only_evaluate = False
    training_log_filename = "training_log.csv"


    # Get the path to the config gile
    experiments_folder = './task1_match_mismatch/experiments'
    task_folder = os.path.dirname(experiments_folder)
    config_path = os.path.join(task_folder, 'util', 'config_512Hz.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test
    data_folder = os.path.join(config["dataset_folder"], config["split_folder"])

    # stimulus feature which will be used for training the model. Can be either 'envelope' ( dimension 1) or 'mel' (dimension 28)
    stimulus_features = ["mod"]
    stimulus_dimension = 1

    features = ["eeg"] + stimulus_features

    results_folder = os.path.join(experiments_folder, "results_submission_3", "finetuned_ffr_models")
    os.makedirs(results_folder, exist_ok=True)

    subjects = [f'sub-{i:03d}' for i in range(1, 72)]

    for subject in subjects:

        results_filename = f'eval_{subject}.json'

        #load dilation model
        model_path = os.path.join(results_folder, "../", "population_ffr_model", "model_ffr_3.h5")
        subj_model = tf.keras.models.load_model(model_path)
        subj_model_path = os.path.join(results_folder, f'finetuned_{subject}.h5')

        # freeze some layers; recompile with new lr
        layers_to_freeze = ['conv1d_2', 'conv1d_4', 'conv1d_6', 'dense']
        for layer in layers_to_freeze:
            subj_model.get_layer(layer).trainable=False

        subj_model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            metrics=["acc"],
            loss=["binary_crossentropy"],
        )

        print(subj_model.summary())

        train_files = [x for x in glob.glob(os.path.join(data_folder, f"train_-_{subject}*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        # Create list of numpy array files
        train_generator = MatchMismatchDataGenerator(train_files, window_length, spacing=spacing)
        dataset_train = create_tf_dataset(train_generator, window_length, default_batch_equalizer_fn, train_hop_length, batch_size, feature_dims=(64, stimulus_dimension, stimulus_dimension))

        # Create the generator for the validation set
        val_files = [x for x in glob.glob(os.path.join(data_folder, f"val_-_{subject}*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        val_generator = MatchMismatchDataGenerator(val_files, window_length, spacing=spacing)
        dataset_val = create_tf_dataset(val_generator, window_length, default_batch_equalizer_fn, hop_length, batch_size, feature_dims=(64, stimulus_dimension, stimulus_dimension))

        # Train the model
        subj_model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(subj_model_path, save_best_only=True),
                tf.keras.callbacks.CSVLogger(os.path.join(results_folder, training_log_filename)),
                tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
            ],
        )


        # Evaluate the model on test set
        # Create a dataset generator for each test subject
        test_files = [x for x in glob.glob(os.path.join(data_folder, f"test_-_{subject}*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        # Get all different subjects from the test set
        subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
        datasets_test = {}
        # Create a generator for each subject
        for sub in subjects:
            files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
            test_generator = MatchMismatchDataGenerator(files_test_sub, window_length, spacing=spacing)
            datasets_test[sub] = create_tf_dataset(test_generator, window_length, default_batch_equalizer_fn, hop_length, 1, feature_dims=(64, stimulus_dimension, stimulus_dimension))


        # Evaluate the model
        evaluation = evaluate_model(subj_model, datasets_test)

        # We can save our results in a json encoded file
        results_path = os.path.join(results_folder, results_filename)
        with open(results_path, "w") as fp:
            json.dump(evaluation, fp)
        logging.info(f"Results saved at {results_path}")