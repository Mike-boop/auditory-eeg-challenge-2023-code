import json
import numpy as np
import os
import tensorflow as tf
import logging
import glob

from task1_match_mismatch.models.dilated_convolutional_model import dilation_model
from task1_match_mismatch.util.dataset_generator import MatchMismatchDataGenerator, default_batch_equalizer_fn, create_tf_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dir = './task1_match_mismatch/experiments/results_submission_1/'

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
        print(f"Scores for subject {subject}:")
        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))
    return evaluation


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

mdls = []

for mdl in range(1, 5):
    file = os.path.join(dir, f'eval_ffr_{mdl}.json')
    dic = json.load(open(file, 'r'))
    print(np.mean([dic[k]['acc'] for k in dic]))

    mdls.append(
        tf.keras.models.load_model(os.path.join(dir, f'model_ffr_{mdl}.h5'))
    )

cmodel = composite_model(mdls)

### test

# Evaluate the model on test set
# Create a dataset generator for each test subject
test_files = [x for x in glob.glob(os.path.join('./data/512Hz/split_data/', "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in ['eeg', 'mod']]
# Get all different subjects from the test set
subjects = sorted(list(set([os.path.basename(x).split("_-_")[1] for x in test_files])))
datasets_test = {}
# Create a generator for each subject
for sub in subjects:
    files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
    test_generator = MatchMismatchDataGenerator(files_test_sub, 512*3, spacing=512)
    datasets_test[sub] = create_tf_dataset(test_generator, 512*3, default_batch_equalizer_fn, 512, 1, feature_dims=(64, 1, 1))

# Evaluate the model
evaluation = evaluate_model(cmodel, datasets_test)

print(np.mean([evaluation[k]['acc'] for k in evaluation]))