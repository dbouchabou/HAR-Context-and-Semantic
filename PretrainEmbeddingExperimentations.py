#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np

import tensorflow as tf


from SmartHomeHARLib.datasets.casas import Dataset

# Casas datasets
from SmartHomeHARLib.datasets.casas import Aruba
from SmartHomeHARLib.datasets.casas import Milan
from SmartHomeHARLib.datasets.casas import Cairo


# Word2vec
from experiments.embedding_pre_trained.word2vec.Word2vecLSTMExperiment import Word2vecLSTMExperiment
from experiments.embedding_pre_trained.word2vec.Word2vecBiLSTMExperiment import Word2vecBiLSTMExperiment

# ELMo
from experiments.embedding_pre_trained.elmo.ELMoLSTMExperiment import ELMoLSTMExperiment
from experiments.embedding_pre_trained.elmo.ELMoBiLSTMExperiment import ELMoBiLSTMExperiment
from experiments.embedding_pre_trained.elmo.FCNELMoBiLSTMExperiment import FCNELMoBiLSTMExperiment


# Comparison
from experiments.comparison.LiciottiLSTMExperiment import LiciottiLSTMExperiment
from experiments.comparison.LiciottiBiLSTMExperiment import LiciottiBiLSTMExperiment

from experiments.comparison.LSTMExperiment import LSTMExperiment
from experiments.comparison.BiLSTMExperiment import BiLSTMExperiment
from experiments.comparison.BiLSTM2LExperiment import BiLSTM2LExperiment

#FCN
from experiments.comparison.FCNEmbeddingExperiment import FCNEmbeddingExperiment
from experiments.comparison.FCNNoEmbeddingExperiment import FCNNoEmbeddingExperiment

SEED = 7
DEBUG_MODE = False


# Fix the random seed for tensorflow
tf.random.set_seed(SEED)
# Fix the random seed for numpy
np.random.seed(SEED)

def load_config(config_path):
    f = open(config_path,)

    # returns JSON object as 
    # a dictionary
    return json.load(f)

if __name__ == '__main__':

    # Specify the Tensorflow environment
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
       try:
           # Currently, memory growth needs to be the same across GPUs
           for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
    
       except RuntimeError as e:
           # Memory growth must be set before GPUs have been initialized
           print(e)
    
    strategy = tf.distribute.MirroredStrategy()

    # Set and parse the arguments list
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--d', dest='data', action='store', default='', help='dataset name', required=True)
    p.add_argument('--e', dest='experiment', action='store', default='', help='dataset name', required=True)
    p.add_argument('--c', dest='config', action='store', default='', help='config_file', required=True)

    args = p.parse_args()

    data = str(args.data)
    config_path = str(args.config)
    experiement = str(args.experiment)

    if data == "aruba":
        dataset = Aruba()
    elif data == "milan":
        dataset = Milan()
    elif data == "cairo":
        dataset = Cairo()
    else:
        print("UNKNOWED DATASET")
    
    # Load the config file
    config = load_config(config_path)

    # Group activity of the same nature under a generic label following a dictionary
    RENAME_ACTIVITY = config["activity_renamed"]

    

    with strategy.scope():

        if experiement == "lstm": 
            exp = LSTMExperiment(dataset, config)
        elif experiement == "bi_lstm":
            exp = BiLSTMExperiment(dataset, config)
        elif experiement == "liciotti_lstm":
            exp = LiciottiLSTMExperiment(dataset, config)
        elif experiement == "liciotti_bi_lstm":
            exp = LiciottiBiLSTMExperiment(dataset, config)
        elif experiement == "w2v_lstm":
            exp = Word2vecLSTMExperiment(dataset, config)
        elif experiement == "w2v_bi_lstm":
            exp = Word2vecBiLSTMExperiment(dataset, config)
        elif experiement == "elmo_lstm":
            exp = ELMoLSTMExperiment(dataset, config)
        elif experiement == "elmo_bi_lstm":
            exp = ELMoBiLSTMExperiment(dataset, config)
        elif experiement == "bi_lstm_2L":    
            exp = BiLSTM2LExperiment(dataset, config)
        elif experiement == "fcn_no_embedding":    
            exp = FCNNoEmbeddingExperiment(dataset, config)
        elif experiement == "fcn_embedding":    
            exp = FCNEmbeddingExperiment(dataset, config)
        elif experiement == "elmo_fcn_embedding":    
            exp = FCNELMoBiLSTMExperiment(dataset, config)

        exp.DEBUG = DEBUG_MODE

        exp.start()

    # Save word dict
    exp.save_word_dict()

    # Save activity dict
    exp.save_activity_dict()

    # Save metrics
    exp.save_metrics()

    # Save experiment config
    exp.save_config()


    print('Accuracy: {:.2f}% (+/- {:.2f}%)'.format(np.mean(exp.global_classifier_accuracy) * 100, 
                                                   np.std(exp.global_classifier_accuracy)
                                            )
    )

    print('Balanced Accuracy: {:.2f}% (+/- {:.2f}%)'.format(np.mean(exp.global_classifier_balance_accuracy) * 100,
                                                            np.std(exp.global_classifier_balance_accuracy)
                                                    )
    )
