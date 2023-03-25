"""
Sample code to generate labels for test dataset of
match-mismatch task. The requested format for submitting the labels is
as follows:
for each subject a json file containing a python dictionary in the
format of  ==> {'sample_id': prediction, ... }.

"""

import os
import glob
import json
import numpy as np
from task1_match_mismatch.models.dilated_convolutional_model import dilation_model
import tensorflow as tf

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


def create_test_samples(eeg_path, envelope_dir):
    with open(eeg_path, 'r') as f:
        sub = json.load(f)
    eeg_data = []
    spch1_data = []
    spch2_data = []
    id_list = []
    for key, sample in sub.items():
        eeg_data.append(sample[0])

        spch1_path = os.path.join(envelope_dir, sample[1])
        spch2_path = os.path.join(envelope_dir, sample[2])
        envelope1 = np.load(spch1_path)
        env1 = envelope1['envelope']
        envelope2 = np.load(spch2_path)
        env2 = envelope2['envelope']

        spch1_data.append(env1)
        spch2_data.append(env2)
        id_list.append(key)
    eeg = np.array(eeg_data)
    spch1 = np.array(spch1_data)
    spch2 = np.array(spch2_data)
    return (eeg, spch1, spch2), id_list


def get_label(pred):
    if pred >= 0.5:
        label = 1
    else:
        label = 0
    return label

if __name__ == '__main__':

    window_length = 3*512

    # Root dataset directory containing test set
    # Change the path to the downloaded test dataset dir
    dataset_dir = './data/test-task1'

    # load model

    dir = './task1_match_mismatch/experiments/results_submission_1/'
    mdls = []

    for mdl in range(1, 31):

        mdls.append(
            tf.keras.models.load_model(os.path.join(dir, 'trained_ffr_models', f'model_ffr_{mdl}.h5'))
        )

    model = composite_model(mdls)
    print('loaded model')

    test_data = glob.glob(os.path.join(dataset_dir, 'eeg_512Hz', 'sub*.json'))
    
    for sub_path in test_data:
        subject = os.path.basename(sub_path).split('.')[0]

        sub_dataset, id_list = create_test_samples(sub_path, os.path.join(dataset_dir, 'modulation_segments'))
        # Normalize data
        subject_data = []
        for item in sub_dataset:
            item_mean = np.expand_dims(np.mean(item, axis=1), axis=1)
            item_std = np.expand_dims(np.std(item, axis=1), axis=1)
            subject_data.append((item - item_mean) / item_std)
        sub_dataset = tuple(subject_data)

        predictions = model.predict(sub_dataset)
        predictions = np.squeeze(predictions).tolist()
        predicted_labels = map(get_label, predictions)
        sub_predicted_labels = dict(zip(id_list, predicted_labels))
        sub_soft_outputs = dict(zip(id_list, predictions))

        prediction_dir = os.path.join(os.path.dirname(__file__), '../', 'results_submission_4', 'heldout_predictions_avg_ffr')
        os.makedirs(prediction_dir, exist_ok=True)
        with open(os.path.join(prediction_dir, subject + '.json'), 'w') as f:
            json.dump(sub_predicted_labels, f)

        with open(os.path.join(prediction_dir, subject + '-outputs.json'), 'w') as f:
            json.dump(sub_soft_outputs, f)