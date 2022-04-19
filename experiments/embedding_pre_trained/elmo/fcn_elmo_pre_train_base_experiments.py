# coding: utf-8
# !/usr/bin/env python3

import os
import sys
import csv
import time
import json
import itertools
import pandas as pd
import numpy as np

from progress.bar import *

from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.activations import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence


from SmartHomeHARLib.utils import Experiment
from SmartHomeHARLib.utils import Evaluator

from SmartHomeHARLib.datasets.casas import Encoder
from SmartHomeHARLib.datasets.casas import Segmentator

from SmartHomeHARLib.embedding import ELMoEventEmbedder


cairo_dict ={"Other": "Other",
            "R1_wake": "Other",
            "R2_wake": "Other",
            "Night_wandering": "Other",
            "R1_work_in_office": "Work",
            "Laundry": "Work",
            "R2_take_medicine": "Take_medicine",
            "R1_sleep": "Sleep",
            "R2_sleep": "Sleep",
            "Leave_home": "Leave_Home",
            "Breakfast": "Eat",
            "Dinner": "Eat",
            "Lunch": "Eat",
            "Bed_to_toilet": "Bed_to_toilet"}

milan_dict = {"Other": "Other",
                "Master_Bedroom_Activity": "Other",
                "Meditate": "Other",
                "Chores": "Work",
                "Desk_Activity": "Work",
                "Morning_Meds": "Take_medicine",
                "Eve_Meds": "Take_medicine",
                "Sleep": "Sleep",
                "Read": "Relax",
                "Watch_TV": "Relax",
                "Leave_Home": "Leave_Home",
                "Dining_Rm_Activity": "Eat",
                "Kitchen_Activity": "Cook",
                "Bed_to_Toilet": "Bed_to_toilet",
                "Master_Bathroom": "Bathing",
                "Guest_Bathroom": "Bathing"}

aruba_dict ={"Other": "Other",
            "Wash_Dishes": "Work",
            "Sleeping": "Sleep",
            "Respirate": "Take_medicine",
            "Relax": "Relax",
            "Meal_Preparation": "Cook",
            "Housekeeping": "Work",
            "Enter_Home": "Enter_Home",
            "Leave_Home": "Leave_Home",
            "Eating": "Eat",
            "Bed_to_Toilet": "Bed_to_toilet",
            "Work":"Work"}

class Camouflage(Layer):
    """Masks a sequence by using a mask value to skip timesteps based on another sequence.
       LSTM and Convolution layers may produce fake tensors for padding timesteps. We need
       to eliminate those tensors by replicating their initial values presented in the second input.
       inputs = Input()
       lstms = LSTM(units=100, return_sequences=True)(inputs)
       padded_lstms = Camouflage()([lstms, inputs])
       ...
    """

    def __init__(self, mask=None, **kwargs):
        super(Camouflage, self).__init__(**kwargs)
        self.mask = mask

    def call(self, inputs):
        #input_shape = tf.shape(inputs)
        #batch_size = input_shape[0]
        #seq_len = input_shape[1]
        #causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)

        #output = MultiHeadAttention(num_heads=1, key_dim=inputs.shape[2])(inputs, inputs, attention_mask=causal_mask)

        input_layer = inputs[0]
        original_sequence = inputs[1]

        original_sequence = tf.cast(original_sequence, tf.float32)
        masks = tf.cast(tf.math.not_equal(original_sequence, 0.), tf.float32)
        #print(masks)
        #input("Press Enter to continue...")
        masks = tf.expand_dims(masks, axis=2)
        #print(masks)
        #input("Press Enter to continue...")
        masks = tf.repeat(masks,inputs[0].shape[2],axis=2)
        #print(masks)
        #input("Press Enter to continue...")
        output =  tf.math.multiply(input_layer,masks)
        
        return output
        

    def get_config(self):
        config = {'mask': self.mask}
        base_config = super(Camouflage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class FCNELMoPreTrainBaseExperiment(Experiment):

    def __init__(self, dataset, experiment_parameters):
        super().__init__(dataset, experiment_parameters)

        # General
        self.global_classifier_accuracy = []
        self.global_classifier_balance_accuracy = []
        self.current_time = None

        # Embedding
        self.elmo_model = ELMoEventEmbedder()
        self.elmo_model.load_model(self.experiment_parameters["pre_train_embedding"])

        # Classifier
        self.classifier_dataset_encoder = None
        self.classifier_segmentator = None

        self.classifier_model = None
        self.classifier_best_model_path = None
        self.classifier_data_X = []
        self.classifier_data_Y = []
        self.classifier_data_X_train = []
        self.classifier_data_Y_train = []
        self.classifier_data_X_test = []
        self.classifier_data_Y_test = []
        self.classifier_data_X_val = None
        self.classifier_data_Y_val = None


        self.experiment_tag = "Dataset_{}_Encoding_{}_Segmentation_{}_Batch_{}_Patience_{}_SeqLenght_{}_EmbDim_{}".format(
            self.dataset.name, self.experiment_parameters["encoding"],
            self.experiment_parameters["segmentation"],
            self.experiment_parameters["batch_size"],
            self.experiment_parameters["patience"],
            self.experiment_parameters["sequence_lenght"],
            self.elmo_model.embedding_size
        )


    def encode_data(self):
        X = self.classifier_segmentator.X

        sentences = []
        for activity in X:
            sentence = " ".join(activity)
            sentence = "<start> " + sentence + " <end>"

            # store sentence
            sentences.append(sentence)

        tokenizer = Tokenizer(filters = '', lower = False, oov_token="<UNK>")
        tokenizer.fit_on_texts(sentences)

        tokenizer.word_index = self.elmo_model.vocabulary

        encoded_sentences = tokenizer.texts_to_sequences(sentences)

        return encoded_sentences

    def rename_labels(self, dict_act):
        inv_map = {v: k for k, v in self.classifier_dataset_encoder.actDict.items()}

        labels_full_name = list(map(lambda x: inv_map[x], self.classifier_segmentator.Y))

        labels_renamed = list(map(lambda x: dict_act[x], labels_full_name))

        # generate dictionnaire
        valList = np.unique(np.array(labels_renamed))
        valList.sort()

        valDict={}
        for i, v in enumerate(valList):
            valDict[v] = i	

        return list(map(lambda x: valDict[x], labels_renamed)),valDict
    
    def encode_dataset_for_classifier(self):
        self.classifier_dataset_encoder = Encoder(self.dataset)
        #self.classifier_dataset_encoder.eventEmbedding5()
        self.classifier_dataset_encoder.basic_raw()


    def segment_dataset_for_classifier(self):
        self.classifier_segmentator = Segmentator(self.classifier_dataset_encoder)
        self.classifier_segmentator.explicitWindow()


    def pad_fcn(self,sequences, maxlen):

        new_sequences = []
        

        for seq in sequences:
            if len(seq) > maxlen:
                new_seq = np.array(seq[:maxlen])
            else:
                new_len = maxlen-len(seq)
                new_seq = np.pad(seq, (0, new_len), 'wrap')
        
            new_sequences.append(new_seq)
        
        return np.array(new_sequences)


    def prepare_data_for_classifier(self):

        #self.classifier_data_X = pad_sequences(self.classifier_segmentator.X, maxlen = self.experiment_parameters["sequence_lenght"], padding = 'post')

        encoded_sentences = self.encode_data()

        self.classifier_data_X = self.pad_fcn(encoded_sentences, maxlen = self.experiment_parameters["sequence_lenght"])
        self.classifier_data_X = pad_sequences(self.classifier_data_X, maxlen = self.experiment_parameters["sequence_lenght"], padding = 'post')
        
        if self.experiment_parameters["activity_renamed"] == True:

            dataset_name = self.dataset.name.lower()

            if "aruba" in dataset_name:
                print("ARUBA DICT!")
                dataset_dict = aruba_dict
            elif "milan" in dataset_name:
                print("MILAN DICT!")
                dataset_dict = milan_dict
            elif "cairo" in dataset_name:
                print("CAIRO DICT!")
                dataset_dict = cairo_dict

            self.classifier_data_Y, self.classifier_dataset_encoder.actDict = self.rename_labels(dataset_dict)
        else :
            print("NO DICT!")
            self.classifier_data_Y = self.classifier_segmentator.Y


    def prepare_dataset(self):

        bar = IncrementalBar('Prepare Dataset', max=3)

        self.encode_dataset_for_classifier()
        bar.next()

        self.segment_dataset_for_classifier()
        bar.next()

        self.prepare_data_for_classifier()
        bar.next()

        bar.finish()


    def model_selection(self):

        bar = IncrementalBar('Dataset Spliting', max=2)

        kfold = StratifiedKFold(n_splits=self.experiment_parameters["nb_splits"], shuffle=True, random_state=self.experiment_parameters["seed"])

        k = 0
        for train, test in kfold.split(self.classifier_data_X, self.classifier_data_Y, groups=None):
            self.classifier_data_X_train.append(np.array(self.classifier_data_X)[train])
            self.classifier_data_Y_train.append(np.array(self.classifier_data_Y)[train])


            self.classifier_data_X_test.append(np.array(self.classifier_data_X)[test])
            self.classifier_data_Y_test.append(np.array(self.classifier_data_Y)[test])
        
        bar.next()

        self.classifier_data_X_train = np.array(self.classifier_data_X_train)
        self.classifier_data_Y_train = np.array(self.classifier_data_Y_train)

        self.classifier_data_X_test = np.array(self.classifier_data_X_test)
        self.classifier_data_Y_test = np.array(self.classifier_data_Y_test)
        bar.next()

        if self.DEBUG:
            print("")
            print(self.classifier_data_X_train.shape)
            print(self.classifier_data_Y_train.shape)
            print(self.classifier_data_X_test.shape)
            print(self.classifier_data_Y_test.shape)

            input("Press Enter to continue...")

        bar.finish()


    def train(self, X_train_input, Y_train_input, X_val_input, Y_val_input, run_number=0):

        root_logdir = os.path.join(self.experiment_parameters["name"],
                                   "logs_{}_{}".format(self.experiment_parameters["name"], 
                                   self.dataset.name)
        )

        run_id = self.classifier_model.name + "_" + self.experiment_tag + "_" + str(self.current_time) + str(run_number)
        log_dir = os.path.join(root_logdir, run_id)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        best_model_name_saved = self.classifier_model.name + "_" + self.experiment_tag + "_BEST_" + str(run_number) + ".h5"
        self.classifier_best_model_path = os.path.join(self.experiment_result_path, best_model_name_saved)

        csv_name = self.classifier_model.name + "_" + self.experiment_tag + "_" + str(run_number) + ".csv"
        csv_path = os.path.join(self.experiment_result_path, csv_name)

        # create a callback for the tensorboard
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir)

        # callbacks
        csv_logger = CSVLogger(csv_path)

        # simple early stopping
        es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = self.experiment_parameters["patience"])
        mc = ModelCheckpoint(self.classifier_best_model_path, monitor = 'val_sparse_categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)

        # cbs = [csv_logger,tensorboard_cb,mc,es,cm_callback]
        cbs = [csv_logger, tensorboard_cb, mc, es]

        nb_classes = len(self.dataset.activitiesList)

        self.classifier_model.fit(X_train_input, 
                        Y_train_input, 
                        epochs = self.experiment_parameters["nb_epochs"],
                        batch_size=self.experiment_parameters["batch_size"], 
                        verbose=self.experiment_parameters["verbose"],
                        callbacks=cbs, 
                        validation_split=0.2, 
                        shuffle=True
        )


    def check_input_model(self, run_number = 0):

        X_val_input = None
        Y_val_input = None

        if self.DEBUG:
            print(self.classifier_data_X_train.shape)
            print(self.classifier_data_X_test.shape)
            if self.classifier_data_X_val != None:
                print(self.classifier_data_X_val.shape)
            else:
                print("None")
            input("Press Enter to continue...")

        # Check number size of exemples
        if len(self.classifier_data_X_train) < 2:
            data_X_train = self.classifier_data_X_train[0]
            data_Y_train = self.classifier_data_Y_train[0]
        else:
            data_X_train = self.classifier_data_X_train[run_number]
            data_Y_train = self.classifier_data_Y_train[run_number]

        if len(self.classifier_data_X_test) < 2:
            data_X_test = self.classifier_data_X_test[0]
            data_Y_test = self.classifier_data_Y_test[0]
        else:
            data_X_test = self.classifier_data_X_test[run_number]
            data_Y_test = self.classifier_data_Y_test[run_number]

        if self.classifier_data_X_val != None:
            if len(self.classifier_data_X_val) < 2:
                data_X_val = self.classifier_data_X_val[0]
                data_Y_val = self.classifier_data_Y_val[0]
            else:
                data_X_val = self.classifier_data_X_val[run_number]
                data_Y_val = self.classifier_data_Y_val[run_number]

        # Nb features depends on data shape
        if data_X_train.ndim > 2:
            nb_features = data_X_train.shape[2]
        else:
            nb_features = 1

        if self.DEBUG:
            print(len(data_X_train[:, :, 0]))
            print(len(data_X_train[:, :, 1]))
            print(len(data_X_train[:, :, 2]))
            print(data_X_train.shape)

        X_train_input = [data_X_train[:, :, 0], data_X_train[:, :, 1], data_X_train[:, :, 2]]
        X_test_input = [data_X_test[:, :, 0], data_X_test[:, :, 1], data_X_test[:, :, 2]]

        if self.classifier_data_X_val != None:
            X_val_input = [data_X_val[:, :, 0], data_X_val[:, :, 1], data_X_val[:, :, 2]]

        Y_train_input = data_Y_train
        Y_test_input = data_Y_test

        if self.classifier_data_X_val != None:
            Y_val_input = data_Y_val

        if self.DEBUG:
            print("Train {}:".format(np.array(X_train_input).shape))
            print("Test : {}".format(np.array(X_test_input).shape))

            if self.classifier_data_X_val != None:
                print("Val : {}".format(np.array(X_val_input).shape))
            else:
                print("Val : None")

            input("Press Enter to continue...")

        return X_train_input, Y_train_input, X_val_input, Y_val_input, X_test_input, Y_test_input, nb_features


    def compile_model(self):
        self.classifier_model.compile(loss = 'sparse_categorical_crossentropy', 
                                        optimizer = tf.keras.optimizers.Adam(),
                                        metrics = ['sparse_categorical_accuracy']
        )

        # print summary
        print(self.classifier_model.summary())


    def evaluate(self, X_test_input, Y_test_input, run_number=0):

        if self.DEBUG:
            print("")
            print("EVALUATION")
            print(np.array(X_test_input).shape)
            print(np.array(Y_test_input).shape)
            print(self.classifier_best_model_path)
            input("Press Enter to continue...")

        #evaluator = Evaluator(X_test_input, Y_test_input, model_path=self.classifier_best_model_path)
        evaluator = Evaluator(
                                X_test_input, 
                                Y_test_input, 
                                model_path=self.classifier_best_model_path,
                                custom_objects={
                                            'Camouflage': Camouflage
                                }
        )

        nb_classes = len(self.dataset.activitiesList)

        evaluator.simpleEvaluation(self.experiment_parameters["batch_size"], Y_test_input=Y_test_input)
        self.global_classifier_accuracy.append(evaluator.ascore)

        evaluator.evaluate()

        listActivities = list(self.classifier_dataset_encoder.actDict.keys())
        indexLabels = list(self.classifier_dataset_encoder.actDict.values())
        evaluator.classificationReport(listActivities, indexLabels)
        # print(evaluator.report)

        report_name = self.classifier_model.name + "_repport_" + self.experiment_tag + "_" + str(run_number) + ".csv"
        report_path = os.path.join(self.experiment_result_path, report_name)
        evaluator.saveClassificationReport(report_path)

        evaluator.confusionMatrix()
        # print(evaluator.cm)

        confusion_name = self.classifier_model.name + "_confusion_matrix_" + self.experiment_tag + "_" + str(
            run_number) + ".csv"
        confusion_path = os.path.join(self.experiment_result_path, confusion_name)
        evaluator.saveConfusionMatrix(confusion_path)

        evaluator.balanceAccuracyCompute()
        self.global_classifier_balance_accuracy.append(evaluator.bscore)


    def start(self):

        # Star time of the experiment
        self.current_time = time.strftime("%Y_%m_%d_%H_%M_%S")

        self.experiment_result_path = os.path.join(self.experiment_parameters["name"], self.experiment_parameters["model_type"],
                                           "run_" + self.experiment_tag + "_" + str(self.current_time))

        # create a folder with the model name
        # if the folder doesn't exist
        if not os.path.exists(self.experiment_result_path):
            os.makedirs(self.experiment_result_path)

        self.prepare_dataset()

        # Split the dataset into train, val and test examples
        self.model_selection()

        nb_runs = len(self.classifier_data_X_train)

        if self.DEBUG:
            print("")
            print("NB RUN: {}".format(nb_runs))

        for run_number in range(nb_runs):
            # prepare input according to the model type
            X_train_input, Y_train_input, X_val_input, Y_val_input, X_test_input, Y_test_input, nb_features = self.check_input_model(
                run_number)

            self.build_model_classifier(nb_features)

            # compile the model
            self.compile_model()

            self.train(X_train_input, Y_train_input, X_val_input, Y_val_input, run_number)

            self.evaluate(X_test_input, Y_test_input, run_number)

    def __save_dict_to_json(self, where_to_save, dict_to_save):

        with open(where_to_save, "w") as json_dict_file:
            json.dump(dict_to_save, json_dict_file, indent = 4)


    def save_word_dict(self):

        word_dict_name = "wordDict.json"
        word_dict_path = os.path.join(self.experiment_result_path, word_dict_name)

        self.__save_dict_to_json(word_dict_path, self.elmo_model.vocabulary)

    
    def save_activity_dict(self):

        activity_dict_name = "activityDict.json"
        activity_dict_path = os.path.join(self.experiment_result_path, activity_dict_name)

        self.__save_dict_to_json(activity_dict_path, self.classifier_dataset_encoder.actDict)
        

    def save_metrics(self):
        
        csv_name = "cv_scores" + self.classifier_model.name + "_" + self.experiment_tag + "_" + str(self.current_time) + ".csv"
        csv_path = os.path.join(self.experiment_result_path, csv_name)

        with open(csv_path, "w") as output:
            writer = csv.writer(output, lineterminator='\n')

            writer.writerow(["accuracy score :"])
            for val in self.global_classifier_accuracy:
                writer.writerow([val * 100])
            writer.writerow([])
            writer.writerow([np.mean(self.global_classifier_accuracy) * 100])
            writer.writerow([np.std(self.global_classifier_accuracy)])

            writer.writerow([])
            writer.writerow(["balanced accuracy score :"])

            for val2 in self.global_classifier_balance_accuracy:
                writer.writerow([val2 * 100])
            writer.writerow([])
            writer.writerow([np.mean(self.global_classifier_balance_accuracy) * 100])
            writer.writerow([np.std(self.global_classifier_balance_accuracy)])
