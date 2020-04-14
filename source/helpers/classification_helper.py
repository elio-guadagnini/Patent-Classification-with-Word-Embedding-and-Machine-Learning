# -*- coding: utf-8 -*-
import sys
import glob
import os
from datetime import datetime
import numpy as np
import pandas as pd
from multiprocessing import Queue, Process
import csv
import re

import pickle as pkl

import gensim
from gensim.corpora.dictionary import Dictionary

from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# from skmultilearn.problem_transform import BinaryRelevance
# from skmultilearn.problem_transform import ClassifierChain
# from skmultilearn.problem_transform import LabelPowerset

# from tensorflow.contrib import learn

sys.path.append(os.path.abspath('..'))
from helpers import directory
from helpers import tool_helper as th
from helpers import folder_helper as fh
from helpers import word_model_helper as wmh
from helpers import lstm_readers_helper as lrh
from helpers import sequential_doc2vec_helper as sdh

current_path = directory.get_working_directory()

script_key = "classification_helper"

settings = {
"classify" : {
                        "data_frame_file_name" : "data/model_dataframe/dataframe.csv",
                        "metrics" : [],
                        "parameters" : [],
},
"sequential_doc2vec" : {
                        "data_frame_file_name" : "data/model_dataframe/dataframe.csv",
                        "metrics" : [],
                        "parameters" : [],
},
"cross_validation" : {
                        "results_file_name" : "data/models_results/cross_validation_results.csv",
                        "metrics" : ['accuracy'],
                        "parameters" : ['cv'],
},
"cross_validation_multiclass" : {
                        "results_file_name" : "data/models_results/cross_validation_multiclass_results.csv",
                        "metrics" : ['algorithm', 'accuracy', 'precision', 'recall', 'class'],
},
"LogisticRegression" : {
                        "results_file_name" : "data/models_results/logistic_regressor_results.csv",
                        "metrics" : ['Default', 'Binary', 'Micro', 'Macro'],
                        "parameters" : ['C', 'class_weight', 'dual', 'fit_intercept', 'intercept_scaling', 'max_iter', 'multi_class', 'n_jobs', 'penalty', 'random_state', 'solver', 'tol', 'verbose', 'warm_start'],
},
"GaussianNB" : {
                        "results_file_name" : "data/models_results/gaussian_nb_results.csv",
                        "metrics" : ['Default', 'Binary', 'Micro', 'Macro'],
                        "parameters" : ['priors', 'var_smoothing'],
},
"SVC" : {
                        "results_file_name" : "data/models_results/svc_results.csv",
                        "metrics" : ['Default', 'Binary', 'Micro', 'Macro'],
                        "parameters" : ['C', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 'degree', 'gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose'],
},
"MultinomialNB" : {
                        "results_file_name" : "data/models_results/mulnomial_nb_results.csv",
                        "metrics" : ['Default', 'Binary', 'Micro', 'Macro'],
                        "parameters" : ['alpha', 'class_prior', 'fit_prior'],
},
"DecisionTreeClassifier" : {
                        "results_file_name" : "data/models_results/decision_tree_classifier_results.csv",
                        "metrics" : ['Default', 'Binary', 'Micro', 'Macro'],
                        "parameters" : ['class_weight', 'criterion', 'max_depth', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'min_impurity_split', 'min_samples_leaf', 'min_sampes_split', 'min_weight_fraction_leaf', 'presort', 'random_state', 'splitter'],
},
"KNeighborsClassifier" : {
                        "results_file_name" : "data/models_results/k_neighbors_results.csv",
                        "metrics" : ['Default', 'Binary', 'Micro', 'Macro'],
                        "parameters" : ['algorithm', 'leaf_size', 'metric', 'metric_params', 'n_jobs', 'n_neighbors', 'p', 'wights'],
},
"LinearSVC" : {
                        "results_file_name" : "data/models_results/linear_svc_results.csv",
                        "metrics" : ['Default', 'Binary', 'Micro', 'Macro'],
                        "parameters" : ['C', 'class_weight', 'dual', 'fit_intercept', 'intercept_scaling', 'loss', 'max_iter', 'multi_class', 'penalty', 'random_state', 'tol', 'verbose'],
},
"RandomForestClassifier" : {
                        "results_file_name" : "data/models_results/random_forest_results.csv",
                        "metrics" : ['Default', 'Binary', 'Micro', 'Macro'],
                        "parameters" : ['bootstrap', 'class_weight', 'criterion', 'max_depth', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'min_impurity_split', 'min_samples_split', 'min_weight_fraction_leaf', 'n_estimators', 'n_jobs', 'oob_score', 'random_state', 'verbose', 'warm_start', 'onemoreparam'],
},
"SGDClassifier" : {
                        "results_file_name" : "data/models_results/sgd_results.csv",
                        "metrics" : ['Default', 'Binary', 'Micro', 'Macro'],
                        "parameters" : ['loss', 'alpha', 'n_jobs', 'random_state', 'learning_rate', 'early_stopping'],
},
"ExtraTreesClassifier" : {
                        "results_file_name" : "data/models_results/extra_tree_results.csv",
                        "metrics" : ['Default', 'Binary', 'Micro', 'Macro'],
                        "parameters" : ['bootstrap', 'class_weight', 'criterion', 'max_depth', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'n_estimators', 'n_jobs', 'oob_scoree', 'random_state', 'verbosearm_starte'],
},
"MLkNN" : {
                        "results_file_name" : "data/models_results/mlknn_results.csv",
                        "metrics" : ['Default', 'Binary', 'Micro', 'Macro'],
                        "parameters" : ['ignore_first_neighbours', 'k', 's'],
},
"TextCNN" : {
                        "results_file_name" : "data/models_results/cnn_results.csv",
                        "metrics" : ['step', 'loss', 'accuracy'],
                        "parameters" : ['...'],
},
"Sequential" : {
                        "results_file_name" : "data/models_results/sequential_results.csv",
                        "metrics" : ['loss', 'accuracy', 'mse'],
                        "parameters" : ['layers'],
},
"Sequential Test" : {
                        "results_file_name" : "data/models_results/sequential_results.csv",
                        "metrics" : ['loss', 'accuracy', 'mse', 'micro', 'macro'],
                        "parameters" : ['layers'],
},
"Sequential_LSTM" : {
                        "results_file_name" : "data/models_results/sequential_lstm_results.csv",
                        "metrics" : ['coverage_error', 'micro', 'macro', 'tops'],
                        "parameters" : ['layers'],
},
"FastText" : {
                        "results_file_name" : "data/models_results/sequantial_results.csv",
                        "metrics" : ['precision', 'recall'],
                        "parameters" : ['parameters'],
},
"convolutional" : {
                        "results_file_name" : "data/models_results/fasttext_results.csv",
                        "metrics" : ['step', 'loss', 'accuracy'],
                        "parameters" : ['layers'],
},
"neural" : {
                        "results_file_name" : "data/models_results/nn_results.csv",
                        "metrics" : ['loss', 'accuracy'],
                        "parameters" : ['epochs', '...'],
},
"lstm" : {
                        "results_file_name" : "data/models_results/lstm_results.csv",
                        "metrics" : [],
                        "parameters" : [],
},
"fastext" : {
                        "results_file_name" : "data/models_results/fastext_results.csv",
                        "metrics" : ['precision', 'recall'],
                        "parameters" : ['dim', 'minn', 'maxn', 'epochs', 'learning_rate', 'loss', 'bucket'],
},
"fuzzy" : {
                        "results_file_name" : "data/models_results/fuzzy_results.csv",
                        "metrics" : [],
                        "parameters" : [],
},
}

def shuffle_rows(data_frame):
    return data_frame.sample(frac=1)

def get_train_test_from_data(X, Y, shuffle=True):
    return train_test_split(X, Y, random_state=0, test_size=.2, shuffle=shuffle)

def get_train_test_from_dataframe(data_frame):
    return train_test_split(data_frame, random_state=0, test_size=.2, shuffle=True)

def manually_split_train_test(X, Y, dev_sample_percentage=.2):
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(Y)))
    X_shuffled = X[shuffle_indices]
    Y_shuffled = Y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(Y)))
    X_train, X_test = X_shuffled[:dev_sample_index], X_shuffled[dev_sample_index:]
    y_train, y_test = Y_shuffled[:dev_sample_index], Y_shuffled[dev_sample_index:]
    del X, Y, X_shuffled, Y_shuffled
    return X_train, X_test, Y_train, Y_test

#########################################################################################################################
# classifications utils:

# def apply_method_to_get_distinct_classes(element):
#     return list(map(lambda class_ : class_, th.tokenize_text(element)))

# def get_alternative_distinct_class_substrings(classification_list, first_int, second_int):
#     list_ = list(map(lambda element : apply_method_to_get_distinct_classes(element), classification_list))
#     flatten = [item for sublist in list_ for item in sublist]
#     flatten = list(set(flatten))
#     flatten.sort()
#     return flatten

def get_distinct_class_substrings(classification_list, first_int, second_int):
    list_ = []
    for element in classification_list:
        for class_ in th.tokenize_text(element):
            if class_[first_int:second_int] not in list_:
                list_.append(class_[first_int:second_int])
    list_.sort()
    return list_

# def apply_method_to_get_classes(element):
#     list_ = list(map(lambda class_ : class_, th.tokenize_text(element)))
#     return " ".join(item for item in list(set(list_)))

# def get_alternative_class_substrings(classification_list, first_int, second_int):
#     list_ = list(map(lambda element : apply_method_to_get_classes(element), classification_list))
#     return list_

def get_class_substrings(classification_list, first_int, second_int):
    list_ = []
    for element in classification_list:
        string = ""
        for class_ in th.tokenize_text(element):
            if class_[first_int:second_int] not in string:
                string += class_[first_int:second_int] + ' '
        list_.append(string)
    return list_

def get_classifications(temp_df):
    ## Load utility data
    classifications = temp_df['classification'].tolist()

    valid_sections = get_distinct_class_substrings(classifications, 0, 1)
    valid_classes = get_distinct_class_substrings(classifications, 1, 3)
    valid_subclasses = get_distinct_class_substrings(classifications, 3, 4)

    sections = get_class_substrings(classifications, 0, 1)
    classes = get_class_substrings(classifications, 0, 3)
    subclasses = get_class_substrings(classifications, 0, 4)

    classification_types = {
        "sectors": sections,
        "classes": classes,
        "subclasses": subclasses
    }
    return classification_types

def handle_item(row, item):
    return [row[1][1], item, row[1][0]]

def handle_row(row):
    if len(th.tokenize_text(row[1][2])) > 1:
        return list(map(lambda item : handle_item(row, item), th.tokenize_text(row[1][2])))
    return [[row[1][1], row[1][2][0], row[1][0]]]

# expand the dataframe to text - classification series with only one classification:
def get_list_each_text_a_different_classification(data_frame):
    list_ = list(map(lambda row : handle_row(row), data_frame.iterrows()))
    list_ = [item for sub_list in list_ for item in sub_list]

    df_single_classification = pd.DataFrame(list_, columns=['text', 'classification', 'patent_id'])
    return df_single_classification

# expand the dataframe to text - classification series with only one classification:
# def get_list_each_text_a_different_classification(data_frame):
#     df_single_classification = pd.DataFrame(columns=['text', 'classification', 'patent_id'])
#     # remove the multi labeled and add the same text with just one label!
#     for index, row in data_frame.iterrows():
#         if len(th.tokenize_text(row[2])) > 1:
#             for item in th.tokenize_text(row[2]):
#                 df_single_classification.loc[df_single_classification.shape[0] + 1] = [row[1], item, row[0]]
#         else:
#             df_single_classification.loc[df_single_classification.shape[0] + 1] = [row[1], row[2][0], row[0]]
#         data_frame = data_frame.drop(index)
#     return df_single_classification

#########################################################################################################################
# saving tools:

def save_training_set(training_set, model_name):
    print('###  saving_training_set  ###')
    root_location = fh.get_root_location('data/saved_training_set/')
    model_name = model_name.replace("/", "-")
    path = fh.link_paths(root_location, 'training '+model_name+' '+str(datetime.now())[:-10]+'.npy')
    np.save(path, training_set)

def save_sets(sets_location, train_data, test_data, val_data, train_labels, test_labels, val_labels, settings):
    try:
        date = datetime.datetime.now().isoformat()
        actual_sets_location = fh.join_paths(sets_location, date)
        with open(fh.link_paths(actual_sets_location, 'training_set '+date+'.pkl'), "wb") as f:
            pkl.dump([train_data, train_labels], f)
        with open(fh.link_paths(actual_sets_location, 'testing_set '+date+'.pkl'), "wb") as f:
            pkl.dump([test_data, test_labels], f)
        with open(fh.link_paths(actual_sets_location, 'validation_set '+date+'.pkl'), "wb") as f:
            pkl.dump([val_data, val_labels], f)
        with open(fh.link_paths(actual_sets_location, 'settings '+date+'.pkl'), "wb") as f:
            pkl.dump(settings, f)
    except:
        print('A problem occurred while saving the sets!')

def load_sets(sets_location, date='01-01-2020'):
    try:
        actual_sets_location = fh.join_paths(sets_location, date)
        print(actual_sets_location, date, fh.link_paths(actual_sets_location, 'training_set '+date+'.pkl'))
        with open(fh.link_paths(actual_sets_location, 'training_set '+date+'.pkl'), "rb") as f:
            train_data, train_labels = pkl.load(f)
        with open(fh.link_paths(actual_sets_location, 'testing_set '+date+'.pkl'), "rb") as f:
            test_data, test_labels = pkl.load(f)
        with open(fh.link_paths(actual_sets_location, 'validation_set '+date+'.pkl'), "rb") as f:
            val_data, val_labels = pkl.load(f)
        with open(fh.link_paths(actual_sets_location, 'settings '+date+'.pkl'), "rb") as f:
            settings = pkl.load(f)
            # classes, n_classes, vocab_processor, len_vocabulary = pkl.load(f)
        return train_data, test_data, val_data, train_labels, test_labels, val_labels, settings
    except:
        print('A problem occurred while loading the sets!')
        return None, None, None, None, None, None, None

def get_elements_list(training_set):
    return list(map(lambda element : element[0], training_set.iterrows()))

# def get_elements_list(training_set):
#     indexes = []
#     for element in training_set.iterrows():
#         indexes.append(element[0])
#     return indexes

def get_csv_path(model_key):
    index = settings[model_key]["results_file_name"].rfind('/')
    path_to_csv = fh.get_root_location(settings[model_key]["results_file_name"][:index])
    return fh.link_paths(path_to_csv, settings[model_key]["results_file_name"][index+1:])

def apply_method_to_create_metrics(tuple_):
    return list(tuple_) if tuple_ else 'None'

def get_metrics_values(metrics):
    return list(map(lambda tuple_ : apply_method_to_create_metrics(tuple_), metrics))

# def get_metrics_values(metrics):
#     metrics_values = []
#     for tuple_ in metrics:
#         if tuple_:
#             metrics_values.append(list(tuple_))
#         else:
#             metrics_values.append('None')
#     return metrics_values

def get_sequential_LSTM_metrics_values(metrics):
    micro = ["precision:"+str(metrics["precision_micro"]), "recall:"+str(metrics["recall_micro"]), "f1:"+str(metrics["f1_micro"])]
    macro = ["precision:"+str(metrics["precision_macro"]), "recall:"+str(metrics["recall_macro"]), "f1:"+str(metrics["f1_macro"])]
    tops = ["top_1:"+str(metrics["top_1"]), "top_3:"+str(metrics["top_3"]), "top_5:"+str(metrics["top_5"])]
    return [str(metrics["coverage_error"]), micro, macro, tops]

def get_sequential_layers_values(parameters):
    return [list(map(lambda token : token, th.tokenize_text(parameters[:-1])))]

# def get_sequential_layers_values(parameters):
#     return [[token for token in parameters[:-1].split(' ')]]

def get_parameters_values(parameters):
    parameters.replace(" ", "")
    return list(map(lambda token : token.split('=')[1], parameters.split(',')))

# def get_parameters_values(parameters):
#     parameters.replace(" ", "")
#     return [token.split('=')[1] for token in parameters.split(',')]

def get_parameters_list(model, metrics, parameters, classif_approach, classif_level, classif_type, dataset_location):
    list_ = [model]
    list_ += list(map(lambda item : item, metrics))
    list_ += [classif_approach, classif_level, classif_type]
    list_ += list(map(lambda item : item, parameters))
    list_ += [dataset_location]
    return list_

# def get_parameters_list(model, metrics, parameters, classif_approach, classif_level, classif_type, dataset_location):
#     list_ = [model]
#     list_ += [item for item in metrics]
#     list_ += [classif_approach, classif_level, classif_type]
#     list_ += [item for item in parameters]
#     list_ += [dataset_location]
#     return list_

# level: title, abstract, claims, description
# type: sectors, classes, subclasses
# approach: word2vec/doc2vec/tf-idf + one-vs-rest/...
def get_saving_dataframe(model_key, metrics_values, parameters_values, classif_approach, classif_level, classif_type, dataset_location):
    columns_list = get_parameters_list('model_name', settings[model_key]["metrics"], settings[model_key]["parameters"], 'approach', 'level', 'type', 'dataset_location')
    values_list = get_parameters_list(model_key, metrics_values, parameters_values, classif_approach, classif_level, classif_type, dataset_location)

    data_frame = pd.DataFrame(columns=columns_list)
    data_frame.loc[data_frame.shape[0] + 1] = values_list
    return data_frame

def write_dataframe_as_csv(data_frame, path_to_csv):
    if os.path.isfile(path_to_csv):
        with open(path_to_csv, 'a') as f:
            data_frame.to_csv(f, sep=',', header=False)
    else:
        data_frame.to_csv(path_to_csv, sep=',', header=True)

def save_results(model_key, metrics, parameters, classif_approach, classif_level, classif_type, dataset_location):
    print('###  saving_results  ###')
    path_to_csv = get_csv_path(model_key)

    if model_key == 'Sequential':
        metrics = list(map(lambda metric : [metric], metrics))
        # metrics = [[metric] for metric in metrics]
        metrics_values = get_metrics_values(metrics)
        parameters_values = get_sequential_layers_values(parameters)
    elif model_key == 'Sequential Test':
        metrics = [[-1], [-1], [-1], # metrics['loss'], metrics['accuracy'], metrics['mse'],
                   [metrics['precision_micro'], metrics['recall_micro'], metrics['f1_micro']],
                   [metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro']]]
        metrics_values = get_metrics_values(metrics)
        parameters_values = get_sequential_layers_values(parameters)
    elif model_key == 'FastText':
        metrics_values = metrics[1:3]
        parameters_values = [parameters]
    elif model_key == 'Sequential_LSTM':
        temp_metrics = metrics
        del temp_metrics["total_positive"], temp_metrics["average_num_of_labels"]
        metrics_values = get_sequential_LSTM_metrics_values(temp_metrics)
        parameters_values = get_sequential_layers_values(parameters)
    else:
        metrics_values = get_metrics_values(metrics)
        parameters_values = get_parameters_values(parameters)

    data_frame = get_saving_dataframe(model_key, metrics_values, parameters_values, classif_approach, classif_level, classif_type, dataset_location)
    write_dataframe_as_csv(data_frame, path_to_csv)

def get_alternative_saving_cross_dataframe(metrics, parameters, classif_level, classif_type, dataset_location):
    metrics_values = list(map(lambda element : element[0], metrics))
    algorithms = list(map(lambda element : element['algorithm'], metrics))

    data_frame = pd.DataFrame(columns=['algorithm', 'accuracy', 'parameters', 'classif_level', 'classif_type', 'dataset_location'])
    for index, value in enumerate(metrics_values):
        data_frame.loc[data_frame.shape[0] + 1] = [algorithms[index], value, parameters, classif_level, classif_type, dataset_location]
    return data_frame

def get_saving_cross_dataframe(metrics, parameters, classif_level, classif_type, dataset_location):
    metrics_values = []
    algorithms = []
    for element in metrics:
        metrics_values.append(element[0])
        algorithms.append(element['algorithm'])

    data_frame = pd.DataFrame(columns=['algorithm', 'accuracy', 'parameters', 'classif_level', 'classif_type', 'dataset_location'])
    for index, value in enumerate(metrics_values):
        data_frame.loc[data_frame.shape[0] + 1] = [algorithms[index], value, parameters, classif_level, classif_type, dataset_location]

    return data_frame

def save_results_cross_validation(model_key, metrics, parameters, classif_level, classif_type, dataset_location):
    print('###  saving_results_cross_validation  ###')
    path_to_csv = get_csv_path(model_key)

    data_frame = get_saving_cross_dataframe(metrics, parameters, classif_level, classif_type, dataset_location)
    write_dataframe_as_csv(data_frame, path_to_csv)

def save_data_frame(script_key, data_frame, csvfile):
    index = settings[script_key]["data_frame_file_name"].rfind('/')
    path_to_csv = fh.get_root_location(settings[script_key]["data_frame_file_name"][:index])
    if csvfile:
        output_path = fh.link_paths(path_to_csv, csvfile)
    else:
        output_path = fh.link_paths(path_to_csv, settings[script_key]["data_frame_file_name"][index+1:])

    data_frame.to_csv(output_path, index=False, sep=',', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")

def handle_row(row, ids_list):
    if isinstance(row, pd.Series):
        try:
            id_, patent_id, text, classcodes = row.tolist()
        except:
            patent_id, text, classcodes = row.tolist()
        tokens = th.tokenize_text(text)
        if len(tokens) < 2:
            ids_list.append(patent_id)
        else:
            if isinstance(classcodes, str):
                temp_classcodes = th.tokenize_text(classcodes)
                for class_ in temp_classcodes:
                    if len(class_) == 4:
                        if class_[0] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] and class_[1].isdigit() and class_[2].isdigit() and class_[3].isalpha():
                            pass
                        else:
                            ids_list.append(patent_id)
                            break
                    else:
                        ids_list.append(patent_id)
                        break
            else:
                print('not string')

def check_out_for_whitespaces(text):
    if isinstance(text, str):
        return ' '.join([element for element in th.tokenize_text(text) if len(element) > 2 and len(element) < 31])

def check_out_empty_texts_and_wrong_classcodes(data_frame):
    print('dataframe shape: ', data_frame.shape)
    ids_list = []
    data_frame.apply(lambda row : handle_row(row, ids_list), axis=1)
    data_frame.set_index(data_frame['patent_id'], inplace=True)

    data_frame.drop(ids_list, axis=0, inplace=True)

    data_frame['text'] = data_frame['text'].apply(lambda text : check_out_for_whitespaces(text))
    print('ids_list len: ', len(ids_list))
    print('dataframe shape: ', data_frame.shape)
    return data_frame

def clean_text(text):
    text = re.sub(r'\b\w{1,3}\b', '', text)
    text = " ".join(text.split())
    return text

def load_data_frame(script_key, csvfile):
    index = settings[script_key]["data_frame_file_name"].rfind('/')
    path_to_csv = fh.get_root_location(settings[script_key]["data_frame_file_name"][:index])
    if csvfile:
        input_path = fh.link_paths(path_to_csv, csvfile)
    else:
        input_path = fh.link_paths(path_to_csv, settings[script_key]["data_frame_file_name"][index+1:])

    print('input_path: ', input_path)

    data_frame = pd.read_csv(input_path, sep=',', quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ", header=None, engine='python')
    data_frame.columns = ['patent_id', 'text', 'classification']

    # data_frame['text'] = data_frame['text'].apply(lambda text : clean_text(text))
    data_frame = check_out_empty_texts_and_wrong_classcodes(data_frame)

    classification_df = pd.DataFrame(columns=['class', 'count'])
    # data_frame['classification'].apply(lambda classcode : th.calculate_class_distribution(classcode, classification_df))
    return data_frame, classification_df

#########################################################################################################################
# binarizer tools:

# as a list of strings, which are consequently made of labels (as above separated by an empty space)
def apply_simple_binarizer(classification, classes):
    print('###  label_binarizer  ###')
    binarized_classification = label_binarize(classification, classes=classes)
    return binarized_classification, classes, binarized_classification.shape[1]

# TODO: may be the same as the simple
def apply_label_binarizer(classification):
    lb = LabelBinarizer()
    return lb.fit_transform(classification)

def apply_multilabel_binarizer(data_frame):
    print('###  multi_label_binarizer  ###')
    ################################################ classification: from text to sparse binary matrix [[0, 1, 0],[1, 0, 1]]
    temp_classification = data_frame.apply(lambda row : th.tokenize_complex_text_in_set(row['classification']), axis=1)
    df_to_list = temp_classification.tolist()
    mlb = MultiLabelBinarizer()
    mlb.fit(df_to_list)
    classes = list(mlb.classes_)
    return mlb.transform(df_to_list), classes, len(classes)

def apply_tfidf_vectorizer_fit(data_frame, pth):
    # list of strings, which are consequently made of words (no stopwords, stemming already applied ...) - preprocessed text
    print('###  tfidf_vectorizer_with_lambda  ###')
    # results_tfidf_strip_accents_ascii_analyzer_char_wb_ngram_range_(2, 2)_norm_l1_max_df_0.9_min_df_0.1_max_features_150
    vectorizer = TfidfVectorizer(strip_accents='ascii', analyzer='char', ngram_range=(2,2), norm='l1', max_df=.9,
                                 min_df=.1, max_features=200)
    vectorizer.fit(data_frame['text'].apply(lambda x : np.str_(x)))

    date = datetime.now().isoformat()
    with open(fh.link_paths(pth, 'tfidf_model '+date+'.pkl'), "wb") as f:
        pkl.dump(vectorizer, f)
    return vectorizer

def apply_tfidf_vectorizer_transform(vectorizer, data_frame):
    # vectorizer = pkl.load(f)
    return vectorizer.transform(data_frame['text'].apply(lambda x : np.str_(x)))

def apply_word2vec_word_averaging(data_frame):
    print('###  word2vec_word_averaging_vectorizer  ###')
    helper_word2vec = wmh.Word2VecHelper()
    text_tokenized = data_frame.apply(lambda row : helper_word2vec.w2v_tokenize_text(row['text']), axis=1).values
    if False:
        try:
            path_ = fh.join_paths(current_path[:current_path.rfind('/', 0, current_path.rfind('/', 0, -1)-1)], "/data/GoogleNews-vectors-negative300.bin")
            print("google vectors : ", path_)
            wv = gensim.models.KeyedVectors.load_word2vec_format(path_, binary=True)
        except:
            print("unable to find word2vec model from google, downloading...")
            wv = gensim.models.KeyedVectors.load_word2vec_format("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", binary=True)
            print("done...")
    else:
        model_path = fh.join_paths(current_path[:current_path.rfind('/', 0, current_path.rfind('/', 0, -1)-1)], "data/vectorizer_models")
        model = wmh.get_word2vec_model(text_tokenized, model_path)
        wv = model.wv
    wv.init_sims(replace=True)
    return helper_word2vec.word_averaging_list(wv, text_tokenized)

def apply_doc2vec(data_frame, type_, doc2vec_path):
    index = type_.rfind('/')
    if index != -1:
        temp_type = type_[index+1:]
    else:
        temp_type = type_

    Y, classes, n_classes = apply_classification_vectorizer(temp_type, data_frame)
    print('finisched class vect')

    # if index == -1:
    X_train, X_test, y_train, y_test = get_train_test_from_data(data_frame, Y)
    # else:
    #     # CHECKED
    #     # total_indexes = 90588+384464
    #     total_indexes = data_frame.shape[0]
    #     breakpoint_ = int(total_indexes * .8)
    #     # breakpoint_ = 380380 # for custom esperimental data
    #     # breakpoint_ = 4

    #     aranged_indexes = np.arange(total_indexes)
    #     train_ids = aranged_indexes[:breakpoint_]
    #     test_ids = aranged_indexes[breakpoint_:]

    #     X_train = data_frame.iloc[train_ids]
    #     X_test = data_frame.iloc[test_ids]

    #     y_train = Y[:breakpoint_]
    #     y_test = Y[breakpoint_:]

    print('finisched get train test')
    patent_ids = X_train['patent_id']
    print('finisched ids')
    helper_doc2vec = wmh.Dov2VecHelper()

    load_doc2vec = False
    if not load_doc2vec:
        # more than our esperiment with 5 training years
        if data_frame.shape[0] <= 475052+1:
            print('finisched doc2vec helpers - if -> array')
            X_train = helper_doc2vec.label_sentences(X_train['text'], 'Train')
            print('finisched label sent')
            X_test = helper_doc2vec.label_sentences(X_test['text'], 'Test')
            print('finisched label sent')
            all_data = X_train + X_test
            print('finisched all data')
            model_dbow = wmh.train_doc2vec(all_data, doc2vec_path)
            print('finisched train doc2vec')
            train_vectors_dbow = helper_doc2vec.get_vectors(model_dbow, len(X_train), 200, 'Train')
            print('finisched get vect: ', len(X_train))
            test_vectors_dbow = helper_doc2vec.get_vectors(model_dbow, len(X_test), 200, 'Test')
            print('finisched get vect: ', len(X_test))
        else:
            print('finisched doc2vec helpers - if -> dataframe')
            X_train = helper_doc2vec.alternative_label_sentences(X_train['text'], 'Train')
            print('finisched label sent')
            X_test = helper_doc2vec.alternative_label_sentences(X_test['text'], 'Test')
            print('finisched label sent')
            all_data = X_train.append(X_test)
            print('finisched all data')
            model_dbow = wmh.train_alternative_doc2vec(all_data, doc2vec_path)
            print('finisched train doc2vec')
            train_vectors_dbow = helper_doc2vec.alternative_get_vectors(model_dbow, len(X_train), 200, 'Train')
            print('finisched get vect: ', len(X_train))
            test_vectors_dbow = helper_doc2vec.alternative_get_vectors(model_dbow, len(X_test), 200, 'Test')
            print('finisched get vect: ', len(X_test))
    else:
        sequence_size, embedding_size = 1, 200
        training_docs_list = X_train['patent_id']
        test_docs_list = X_test['patent_id']
        doc2vec_path = fh.link_paths(doc2vec_path, 'doc2vec_model_reference')
        train_vectors_dbow, test_vectors_dbow, _ = get_df_data(2, training_docs_list, test_docs_list, None, sequence_size, embedding_size, doc2vec_path)
    return train_vectors_dbow, test_vectors_dbow,  y_train, y_test, classes, n_classes, patent_ids

def apply_vocabulary_processor(text):
    # Build vocabulary (similar to CountVectorizer)
    max_document_length = max([len(th.tokenize_text(x)) for x in text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    ######## tensorflow/transform or tf.data
    return np.array(list(vocab_processor.fit_transform(text))), vocab_processor, len(vocab_processor.vocabulary_)

def apply_standard_vectorizer(data_frame, type_):
    Y, classes, n_classes = apply_classification_vectorizer(type_, data_frame)
    try:
        X, vocab_processor, len_vocabulary = apply_vocabulary_processor(data_frame['text'])
    except:
        X = apply_count_vectorizer(data_frame)
        vocab_processor, len_vocabulary = None, 0
    # a list of the words used in the text, identified by a unique number for each different word

    X_train, X_test, y_train, y_test = get_train_test_from_data(X, Y)

    print("Vocabulary Size: {:d}".format(len_vocabulary))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))
    return X_train, X_test, y_train, y_test, classes, n_classes, vocab_processor, len_vocabulary

def set_string_for_fasttext(item):
    return str(item).strip().replace(' or ', ' ').replace(', or ', ' ').replace(', ', ' ').replace(',', ' __label__').replace('$$', ' ').replace('$', ' ').replace(' ', ' __label__').replace('___', '__')

def apply_classification_vectorizer(type_, data_frame):
    classification = data_frame['classification']
    if type_ == 'simple':
        classes = ['H', 'B', 'C'] # useful if i need to shrink the set of classes
        return apply_simple_binarizer(classification, classes)
    elif type_ == 'label_binarizer':
        return apply_label_binarizer(classification), None, 0
    elif type_ == 'multi_label':
        return apply_multilabel_binarizer(data_frame)
    elif type_ == 'fasttext':
        return ['__label__'+set_string_for_fasttext(item) for item in data_frame['classification']], None, 0
    return classification, None, 0

def apply_df_vectorizer(data_frame, type_, classification_type, model_name):
    model_path = fh.get_root_location('data/vectorizer_models/')
    # type_ = "doc2vec"
    # classification_type = "multi_label"
    index = type_.rfind('/')
    if index != -1:
        train, test = data_frame
        data_frame = pd.concat([train, test])
        type_ = type_[index+1:]

    if type_ == 'doc2vec':
        X_train, X_test, y_train, y_test, classes, n_classes, patent_ids = apply_doc2vec(data_frame, classification_type, model_path)
        # array - not numpy
        print('finisched apply doc2vec')
        vocab_processor, len_vocabulary = None, 0
    elif type_ == 'standard':
        standard_results = apply_standard_vectorizer(data_frame, classification_type)
        X_train, X_test, y_train, y_test, classes, n_classes, vocab_processor, len_vocabulary = standard_results
        patent_ids = data_frame['patent_id']
    else:
        Y_vect, classes, n_classes = apply_classification_vectorizer(classification_type, data_frame)
        vocab_processor, len_vocabulary = None, 0
        if type_ == 'tfidf':
            X_vect = pd.DataFrame({'text' : data_frame['text'], 'patent_id' : data_frame['patent_id']})
            X_train, X_test, y_train, y_test = get_train_test_from_data(X_vect, Y_vect)

            patent_ids = X_train['patent_id']

            vectorizer = apply_tfidf_vectorizer_fit(X_vect, model_path)
            # print('prima: ', X_test.iloc[0])
            X_train = apply_tfidf_vectorizer_transform(vectorizer, X_train)
            X_test = apply_tfidf_vectorizer_transform(vectorizer, X_test)
            # print('dopo: ', X_test[0])
            # X_train = apply_tfidf_vectorizer_fit_transform(vectorizer, X_train)
            # X_test = apply_tfidf_vectorizer_fit_transform(vectorizer, X_test)
            # csr matrix
        elif type_ == 'word2vec': # required google pre-trained vectors, it starts downloading if you don't have
            X_vect = apply_word2vec_word_averaging(data_frame)
            # ndarray
            print('finished word2vec word averaging')
            text_indexes = data_frame.index.values

            X_vect_temp = pd.DataFrame({'text' : text_indexes, 'patent_id' : data_frame['patent_id']})
            X_train_temp, X_test_temp, y_train, y_test = get_train_test_from_data(X_vect_temp, Y_vect)

            X_train, patent_ids = th.get_set_from_index(X_train_temp, X_vect)
            X_test, _ = th.get_set_from_index(X_test_temp, X_vect)
            print('finished word2vec vectorizer')
        else:
            data_frame.drop(columns=['classification'], inplace=True)
            X_train, X_test, y_train, y_test = get_train_test_from_data(data_frame, Y_vect)
            return X_train, X_test, y_train, y_test, classes, n_classes, vocab_processor, len_vocabulary
    save_training_set(patent_ids, model_name)
    return X_train, X_test, y_train, y_test, classes, n_classes, vocab_processor, len_vocabulary

def get_tfidf_vectorizer(input_):
    return TfidfVectorizer(analyzer = lambda input_ : input_)

def apply_count_vectorizer(data_frame):
    # this should split every row as a instance - 8669 (both training and test)
    # converts a set of strings to a set of integers
    print('###  count_vectorizer  ###')
    df_to_list = data_frame['text'].tolist()
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(df_to_list).toarray()

def apply_label_encoder(classification):
    # converts a set of strings to a set of integers
    print('###  label_encoder  ###')
    encoder = LabelEncoder()
    return encoder.fit_transform(classification)

def apply_doc2vec_separated_train_test(data_frame, model_name):
    data_frame['classification'] = data_frame.apply(lambda row : th.tokenize_complex_text(row['classification']), axis=1)

    train, test = get_train_test_from_dataframe(data_frame)

    # train_indexes = get_elements_list(train) # indexes in the dataframe (0,3,4,6)
    save_training_set(train, model_name)

    train_tagged = train.apply(lambda row : wmh.get_tagged_document(row), axis=1)
    test_tagged = test.apply(lambda row : wmh.get_tagged_document(row), axis=1)

    model_dbow = wmh.train_doc2vec(train_tagged.values)

    y_train, X_train = wmh.vec_for_learning(model_dbow, train_tagged)
    y_test, X_test = wmh.vec_for_learning(model_dbow, test_tagged)

    return [X_train, y_train, X_test, y_test, model_dbow, train_tagged, test_tagged]

#########################################################################################################################
# problem transformation tools:

# def get_binary_relevance(model_):
#     return BinaryRelevance(model_)

# def get_classifier_chain(model_):
#     return ClassifierChain(model_)

# def get_label_powerset(model_):
#     return LabelPowerset(model_)

#########################################################################################################################
# string tools:

def get_classifier_information(text):
    return text.split('(')[0], text.split('(')[1].split(')')[0]

def get_complex_classifier_information(text, index_1, index_2, index_3, index_4):
    return text.split('(')[index_1].split('=')[index_2], text.split('(')[index_3].split(')')[index_4]

def get_extratree_classifier_information(text):
    return text.split('(')[3].split(', ')[1], text.split('(')[4].split(')')[0]

def get_sequential_classifier_information(model):
    text = str(model)
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    layers = ' '.join(string_list)

    parameters = ""
    for index, el in enumerate(layers.split(' (')):
        if index > 0 and index < len(layers.split(' ('))-1:
            if len(el.split(' ')[-1]) >= 5:
                parameters += el.split(' ')[-1] + " "

    return text.split(' ')[0].split('.')[-1], parameters

def get_fasttext_classifier_information(text):
    return text.split('.')[1]

#########################################################################################################################
# empty space:

def delete_variables(history, metrics_callback):
    del history, metrics_callback

#########################################################################################################################
# preprocess data DEEP LEARNING:

def fill_matrix(data_matrix, source_dict, docs_list):
    """
    the use_get flag is for doc2vec_model.docvecs since it doesnt support .get(), so we catch the exception and
    fill with zeros in that case. This should really happen very rarely (if ever) so this exception handling
    should not be a drain on performance
    """
    for i, doc_id in enumerate(docs_list):
        child_ids = doc_id
        j = 0
        try:
            if source_dict[child_ids] is not None:
                data_matrix[i][j] = source_dict[child_ids]
            else:
                data_matrix[i][j] = [0]
        except:
            data_matrix[i][j] = [0]

def get_single_df(docs_list, sequence_size, embedding_size, doc2vec_docvecs):
    data_ = np.ndarray((len(docs_list), sequence_size, embedding_size), dtype=np.float32)
    fill_matrix(data_, doc2vec_docvecs, docs_list)
    return data_

def get_df_data(num_data, training_docs_list, val_docs_list, test_docs_list, sequence_size, embedding_size, doc2vec_model_location):
    doc2vec_model = sdh.get_doc2vec_model(doc2vec_model_location)
    if num_data == 2:
        X_data = get_single_df(training_docs_list, sequence_size, embedding_size, doc2vec_model.docvecs)
        Xv_data = get_single_df(val_docs_list, sequence_size, embedding_size, doc2vec_model.docvecs)
        Xt_data = None
    elif num_data == 3:
        X_data = get_single_df(training_docs_list, sequence_size, embedding_size, doc2vec_model.docvecs)
        Xv_data = get_single_df(val_docs_list, sequence_size, embedding_size, doc2vec_model.docvecs)
        Xt_data = get_single_df(test_docs_list, sequence_size, embedding_size, doc2vec_model.docvecs)
    return X_data, Xv_data, Xt_data

def batch_generator(input_file, label_file, batch_size, QUEUE_SIZE, is_mlp=False, validate=False):
    q = Queue(maxsize=QUEUE_SIZE)
    p = lrh.ArrayReader(input_file, label_file, q, batch_size, is_mlp, validate)
    p.start()
    while True:
        item = q.get()
        if not item:
            p.terminate()
            print('Finished batch iteration')
            raise StopIteration()
        else:
            yield item

#############################################################################################################################
# handle classes:

def shrink_classes(df, row, class_list):
    if isinstance(row, pd.Series):
        patent_id, text, class_ = row.tolist()
        new_classcodes = []
        classcodes = th.tokenize_text(class_)
        for classcode in classcodes:
            if not classcode in class_list:
                new_classcodes.append(classcode)
        if new_classcodes != []:
            new_class = ' '.join(el for el in new_classcodes)
            df.loc[df.shape[0] + 1] = [patent_id, text, new_class]

def reduce_amount_of_classes(data_frame, classification_df):
    if isinstance(classification_df, pd.DataFrame):
        threshold = int(data_frame.shape[0]/1000*0.35)
        if threshold == 0:
            threshold = 1
        temp_df = classification_df[classification_df['count'] <= threshold]
        classes_list = temp_df['class'].tolist()
    elif isinstance(classification_df, list):
        classes_list = classification_df
    df = pd.DataFrame(columns=['patent_id', 'text', 'classification'])
    data_frame.apply(lambda row : shrink_classes(df, row, classes_list), axis=1)
    return df, classes_list

#############################################################################################################################
# handle vocabulary:

def shrink_to_sectors(classcode_list):
    new_list = []
    for class_ in classcode_list:
        if class_[0] not in new_list:
            new_list.append(class_[0])
    return ' '.join(element for element in new_list)

def shrink_vocabulary(row, vocabulary, data_frame, ids_list):
    if isinstance(row, pd.Series):
        patent_id, text, classification = row.tolist()

        new_tokens = [element for element in text if element in vocabulary]
        if new_tokens != [] or len(new_tokens) > 2:
            data_frame.loc[data_frame.shape[0]+1] = [patent_id, ' '.join(element for element in new_tokens), classification]
        else:
            ids_list.append(patent_id)

def further_preprocessing_phase(temp_data_frame):
    temp_data_frame['text'] = temp_data_frame['text'].apply(lambda text: th.tokenize_text(text) if text != None else '')
    # textlist = temp_data_frame['text'].to_numpy()
    textlist = temp_data_frame['text'].tolist()

    # if it raises an exeption could be the empty texts
    patent_dictionary = Dictionary(textlist)
    corpus = [patent_dictionary.doc2bow(text) for text in textlist]

    print('original dictionary size: ', len(patent_dictionary))

    vocab_tf={}
    for i in corpus:
        for item, count in dict(i).items():
            if item in vocab_tf:
                vocab_tf[item]+=int(count)
            else:
                vocab_tf[item] =int(count)

    remove_ids=[]
    no_of_ids_below_limit=0
    for id,count in vocab_tf.items():
        if count<=5:
            remove_ids.append(id)
    patent_dictionary.filter_tokens(bad_ids=remove_ids)

    patent_dictionary.filter_extremes(no_below=0)
    patent_dictionary.filter_n_most_frequent(30)

    print('parsed dictionary size: ', len(patent_dictionary))

    vocabulary = list(patent_dictionary.token2id.keys())

    ids_list = []
    data_frame = pd.DataFrame(columns=['patent_id', 'text', 'classification'])
    temp_data_frame.apply(lambda row : shrink_vocabulary(row, vocabulary, data_frame, ids_list), axis=1)
    print(len(ids_list))
    data_frame.set_index(data_frame['patent_id'], inplace=True)
    data_frame.drop(ids_list, axis=0, inplace=True)
    return data_frame
