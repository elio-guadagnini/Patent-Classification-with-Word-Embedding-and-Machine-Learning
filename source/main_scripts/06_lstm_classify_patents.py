# -*- coding: utf-8 -*-
import sys
import os
import time
import multiprocessing
import numpy as np
from collections import namedtuple
import _pickle as pickle # object serialization

import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.abspath('..'))
from helpers import folder_helper as fh
from helpers import tool_helper as th
from helpers import txt_data_helper as txth
from helpers import classification_helper as ch
from helpers import metrics_helper as mh
from helpers import word_model_helper as wmh
from helpers import predicting_model_helper as pmh

script_key = "lstm classify"

GLOBAL_VARS = namedtuple('GLOBAL_VARS', ['MODEL_NAME', 'DOC2VEC_MODEL_NAME', 'DOC2VEC_MODEL', 'NN_MODEL_NAME'])
NN_PARAMETER_SEARCH_PREFIX = "lstm_{}_level_{}_batch_{}_nn_parameter_searches.pkl"


#########################################################

# predicting model helper
def get_lstm(optimizer,
            init_mode_1, activation_1,
            init_mode_2, activation_2,
            init_mode_3, activation_3,
            init_mode_4, activation_4,
            init_mode_5, activation_5,
            init_mode_6, activation_6,
            weight_constraint_1,
            weight_constraint_2,
            weight_constraint_3,
            weight_constraint_4,
            weight_constraint_5,
            weight_constraint_6,
            dropout_rate_1,
            dropout_rate_2,
            dropout_rate_3,
            dropout_rate_4,
            neurons_1, neurons_2, neurons_3,
            filters_1, filters_2, filters_3,
            kernel_size_1, kernel_size_2, kernel_size_3,
            strides_1, strides_2, strides_3,
            activation_lstm_1, activation_lstm_2, activation_lstm_3,
            recurrent_activation_1, recurrent_activation_2, recurrent_activation_3,
            w_dropout_do_1, w_dropout_do_2, w_dropout_do_3,
            u_dropout_do_1, u_dropout_do_2, u_dropout_do_3,
            backwards_1, backwards_2, backwards_3,
            unroll_1, unroll_2, unroll_3,
            lstm_output_size_1, lstm_output_size_2, lstm_output_size_3,
            input_size):
    len_vocabulary, n_classes = 64846, 302
    # len_vocabulary, n_classes = 9924, 62
    # build up the model

    stack_layers = 1

    # set up the layers - keras layer
    model = Sequential()
    # model.add(keras.layers.Embedding(input_dim=len_vocabulary, output_dim=neurons_1))

    # model.add(keras.layers.ZeroPadding1D((1,1), input_shape=(len_vocabulary, neurons_1))) # (sequence_size, input_size) = X_data.shape[2], X_data.shape[0]
    # model.add(keras.layers.Conv1D(filters=filters_1, kernel_size=kernel_size_1, strides=strides_1, activation=activation_1,
    #                               kernel_initializer=init_mode_1, kernel_constraint=tf.keras.constraints.max_norm(weight_constraint_1)))
    # model.add(keras.layers.MaxPooling1D())

    # model.add(keras.layers.ZeroPadding1D((1,1), input_shape=(len_vocabulary, neurons_1)))
    # model.add(keras.layers.Conv1D(filters=filters_2, kernel_size=kernel_size_2, strides=strides_2, activation=activation_2,
    #                               kernel_initializer=init_mode_2, kernel_constraint=tf.keras.constraints.max_norm(weight_constraint_2)))
    # model.add(keras.layers.GlobalMaxPooling1D())

    # model.add(keras.layers.ZeroPadding1D((1,1), input_shape=(len_vocabulary, neurons_1)))
    # model.add(keras.layers.Conv1D(filters=filters_3, kernel_size=kernel_size_3, strides=strides_3, activation=activation_3, kernel_initializer=init_mode_3, kernel_constraint=max_norm(weight_constraint_3)))
    # model.add(keras.layers.GlobalMaxPooling1D())

    model.add(LSTM(lstm_output_size_1, activation=activation_lstm_1, recurrent_activation=recurrent_activation_1, use_bias=False,
                   input_dim=input_size, dropout_W=w_dropout_do_1, dropout_U=u_dropout_do_1,
                   implementation=1,
                   return_sequences=False if 1 == stack_layers else True,
                   go_backwards=backwards_1, stateful=False, unroll=unroll_1))

    # model.add(LSTM(lstm_output_size_2, activation=activation_lstm_2, recurrent_activation=recurrent_activation_2, use_bias=False,
    #                input_dim=input_size, dropout_W=w_dropout_do_2, dropout_U=u_dropout_do_2,
    #                implementation=1,
    #                return_sequences=False if 2 == stack_layers else True,
    #                go_backwards=backwards_2, stateful=False, unroll=unroll_2)

    # model.add(LSTM(lstm_output_size_3, activation=activation_lstm_3, recurrent_activation=recurrent_activation_3, use_bias=False,
    #                input_dim=input_size, dropout_W=w_dropout_do_3, dropout_U=u_dropout_do_3,
    #                implementation=1,
    #                return_sequences=False if 3 == stack_layers else True,
    #                go_backwards=backwards_3, stateful=False, unroll=unroll_3)

    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dropout(dropout_rate_1))
    # model.add(keras.layers.Dense(neurons_1, activation=activation_4, kernel_initializer=init_mode_4,
    #                              kernel_constraint=tf.keras.constraints.max_norm(weight_constraint_4)))
    # model.add(keras.layers.Dropout(dropout_rate_2))
    # model.add(keras.layers.Dense(neurons_2, activation=activation_5, kernel_initializer=init_mode_5,
    #                              kernel_constraint=tf.keras.constraints.max_norm(weight_constraint_5)))
    # model.add(keras.layers.Dropout(dropout_rate_3))
    # model.add(keras.layers.Dense(neurons_3, activation=activation_6, kernel_initializer=init_mode_6,
    #                              kernel_constraint=tf.keras.constraints.max_norm(weight_constraint_6)))
    # model.add(keras.layers.Dropout(dropout_rate_4))
    model.add(Dense(n_classes, activation=tf.nn.sigmoid))

    model.summary()

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'mse'])
    return model

def run_lstm(model,
            train_data, train_labels, test_data, test_labels, val_data, val_labels,
            batch_size, callbacks_list, queue_size,
            training_docs_list, val_docs_list,
            predicting_batch_size = None, predicting_steps = None, predicting_max_queue_size = 10, predicting_callbacks_list = None,
            epoch = 200):
    print('###  LSTM  ###')
    # compile the model

    # train the model
    history = model.fit_generator(
        generator=batch_generator(train_data, train_labels, batch_size, queue_size, is_mlp=False, validate=False),
        validation_data=batch_generator(val_data, val_labels, batch_size, queue_size, is_mlp=False, validate=True),
        samples_per_epoch=len(training_docs_list),
        nb_val_samples=len(val_docs_list),
        nb_epoch=epoch,
        callbacks=callbacks_list,
        max_q_size=queue_size)

    # using the recorded weights of the best recorded validation loss
    # last_model_weights = model.get_weights()
    # model.set_weights(metrics_callback.best_weights)

    val_loss, val_acc, val_mse = model.evaluate(val_data, val_labels)
    # check on the test dataset
    test_loss, test_acc, test_mse = model.evaluate(test_data, test_labels)

    predictions = model.predict_generator(
        generator=batch_generator(train_data, train_labels, predicting_batch_size, predicting_max_queue_size, is_mlp=False, validate=True),
        max_q_size=predicting_max_queue_size,
        val_samples=len(val_docs_list))
    # binary_predictions = get_binary_0_5(predictions)

    # return history, predictions, binary_predictions
    return [test_loss, test_acc, test_mse], predictions, [val_loss, val_acc, val_mse]

def train_basic_LSTM(data_frame, text_vectorizer, classif_level, classif_type, dataset_location):
    print('### LSTM - Training ###')

    root_location = get_root_location('data/lstm_outcome/')

    doc2vec_model_location = link_paths(join_paths(root_location, "doc2vec_model/vocab_model/"), "doc2vec_model")
    model_location = link_paths(join_paths(root_location, "lstm_model"), "lstm_model")
    sets_location = join_paths(root_location, "model_sets")

    save_results = False
    save_model = True

    model_name = text_vectorizer+'/'+class_vectorizer+'/LSTM'
    results = apply_df_vectorizer(data_frame, text_vectorizer, class_vectorizer, model_name)
    X_train, X_val, y_train, y_val, classes, n_classes, vocab_processor, len_vocabulary = results

    save_sets(sets_location, X_train, None, X_val, y_train, None, y_val, [classes, n_classes, vocab_processor, len_vocabulary])
    # val_path problem

    training_docs_list = X_train['patent_id']
    val_docs_list = X_val['patent_id']

    sequence_size, embedding_size = 1, 150
    X_data, Xv_data, _ = ch.get_df_data(2, training_docs_list, val_docs_list, None, sequence_size, embedding_size, doc2vec_model_location)
    # sequence size is the (, number, ) in the tuple data
    print(X_data.shape)
    input_size, output_size = X_data.shape[2], n_classes

    parameters = {
        "estimator__epochs": 200,
        "estimator__batch_size": 64,
        "estimator__optimizer_1": 'Adam',
        "estimator__optimizer_2": 'Adam',
        "estimator__optimizer_3": 'Adam',
        "estimator__optimizer_4": 'Adam',
        "estimator__optimizer_5": 'Adam',
        "estimator__optimizer_6": 'Adam',
        "estimator__init_mode_1": 'uniform',
        "estimator__init_mode_2": 'uniform',
        "estimator__init_mode_3": 'uniform',
        "estimator__init_mode_4": 'uniform',
        "estimator__init_mode_5": 'uniform',
        "estimator__init_mode_6": 'uniform',
        "estimator__activation_1": 'softmax',
        "estimator__activation_2": 'softmax',
        "estimator__activation_3": 'softmax',
        "estimator__activation_4": 'softmax',
        "estimator__activation_5": 'softmax',
        "estimator__activation_6": 'softmax',
        "estimator__dropout_rate_1": .0,
        "estimator__dropout_rate_2": .0,
        "estimator__dropout_rate_3": .0,
        "estimator__dropout_rate_4": .0,
        "estimator__weight_constraint_1": 1,
        "estimator__weight_constraint_2": 1,
        "estimator__weight_constraint_3": 1,
        "estimator__weight_constraint_4": 1,
        "estimator__weight_constraint_5": 1,
        "estimator__weight_constraint_6": 1,
        "estimator__neurons_1": n_classes*20,
        "estimator__neurons_2": n_classes*5,
        "estimator__neurons_3": n_classes*2,
        "estimator__neurons_4": n_classes*1.5,
        "estimator__filters": 16,
        "estimator__filters_2": 16,
        "estimator__filters_3": 16,
        "estimator__kernel_size_1": 8,
        "estimator__kernel_size_2": 8, "estimator__kernel_size_3": 8,
        "estimator__strides_1": 8,
        "estimator__strides_2": 8,
        "estimator__strides_3": 8,
        "estimator__activation_lstm_1": 'tanh',
        "estimator__activation_lstm_2": 'tanh',
        "estimator__activation_lstm_3": 'tanh',
        "estimator__recurrent_activation_1": 'hard_sigmoid',
        "estimator__recurrent_activation_2": 'hard_sigmoid',
        "estimator__recurrent_activation_3": 'hard_sigmoid',
        "estimator__w_dropout_do_1": .2,
        "estimator__w_dropout_do_2": .2,
        "estimator__w_dropout_do_3": .2,
        "estimator__u_dropout_do_1": .2,
        "estimator__u_dropout_do_2": .2,
        "estimator__u_dropout_do_3": .2,
        "estimator__backwards_1": False,
        "estimator__backwards_2": False,
        "estimator__backwards_3": False,
        "estimator__unroll_1": False,
        "estimator__unroll_2": False,
        "estimator__unroll_3": False
    }

    model = get_lstm(optimizer,
                    init_mode_1, activation_1,
                    init_mode_2, activation_2,
                    init_mode_3, activation_3,
                    init_mode_4, activation_4,
                    init_mode_5, activation_5,
                    init_mode_6, activation_6,
                    weight_constraint_1,
                    weight_constraint_2,
                    weight_constraint_3,
                    weight_constraint_4,
                    weight_constraint_5,
                    weight_constraint_6,
                    dropout_rate_1,
                    dropout_rate_2,
                    dropout_rate_3,
                    dropout_rate_4,
                    neurons_1, neurons_2, neurons_3,
                    filters_1, filters_2, filters_3,
                    kernel_size_1, kernel_size_2, kernel_size_3,
                    strides_1, strides_2, strides_3,
                    activation_lstm_1, activation_lstm_2, activation_lstm_3,
                    recurrent_activation_1, recurrent_activation_2, recurrent_activation_3,
                    w_dropout_do_1, w_dropout_do_2, w_dropout_do_3,
                    u_dropout_do_1, u_dropout_do_2, u_dropout_do_3,
                    backwards_1, backwards_2, backwards_3,
                    unroll_1, unroll_2, unroll_3,
                    lstm_output_size_1, lstm_output_size_2, lstm_output_size_3,
                    input_size) # input size
                    # check the X_data shape and vocabulary size
                    # check the X_data shape and vocabulary size

    if save_model:
        model.save(model_location)

    min_delta, patience = 0.00001, 15
    early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1, mode='auto')
    metrics_callback = MetricsCNNCallback(Xv_data, y_val, patience)

    metrics, val_predictions, val_metrics = run_lstm(model,
                                                X_data, y_train, Xt_data, y_test, Xv_data, y_val,
                                                batch_size, [early_stopper, metrics_callback], queue_size,
                                                training_docs_list, val_docs_list)
    binary_val_predictions = mh.get_binary_0_5(val_predictions)

    print('\nGenerating Validation Metrics')
    validation_metrics = mh.get_sequential_metrics(y_val, val_predictions, binary_val_predictions)
    mh.display_sequential_metrics(validation_metrics)

    if save_results:
        classifier_name, parameters = ch.get_sequential_classifier_information(model)
        model_name = text_vectorizer+'/'+class_vectorizer+'/'+classifier_name
        ch.save_results(classifier_name+'_LSTM', validation_metrics, parameters, model_name, classif_level, classif_type, dataset_location)

    print('end training step')

def test_basic_LSTM(data_frame, text_vectorizer, classif_level, classif_type, dataset_location):
    print('### LSTM Doing Testing ###')

    root_location = fh.get_root_location('data/lstm_outcome/')

    doc2vec_model_location = fh.link_paths(fh.join_paths(root_location, "doc2vec_model/vocab_model/"), "doc2vec_model")
    model_location = link_paths(join_paths(root_location, "lstm_model"), "lstm_model")
    sets_location = join_paths(root_location, "model_sets")

    save_results = True
    load_model = True

    # date problem
    X_train, X_test, X_val, y_train, y_test, y_val, settings = load_sets(sets_location)
    classes, n_classes, vocab_processor, len_vocabulary = settings

    training_docs_list = X_train['patent_id']
    test_docs_list = X_test['patent_id']
    val_docs_list = X_val['patent_id']

    sequence_size, embedding_size = 1, 150
    X_data, Xv_data, Xt_data = ch.get_df_data(3, training_docs_list, val_docs_list, test_docs_list, sequence_size, embedding_size, doc2vec_model_location)

    #####
    print(X_data.shape)
    input_size, output_size = X_data.shape[2], n_classes

    if load_model:
        model = tf.keras.models.load_model(model_location)

    min_delta, patience = 0.00001, 15
    early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1, mode='auto')
    metrics_callback = MetricsCNNCallback(Xv_data, y_val, patience)

    metrics, predictions, val_metrics = run_lstm(model,
                                                X_data, y_train, Xt_data, y_test, Xv_data, y_val,
                                                batch_size, [early_stopper, metrics_callback], queue_size,
                                                training_docs_list, val_docs_list)
    binary_predictions = mh.get_binary_0_5(predictions)

    print('\nGenerating Testing Metrics')
    metrics = mh.get_sequential_metrics(y_val, predictions, binary_predictions)
    mh.display_sequential_metrics(metrics)

    if save_results:
        classifier_name, parameters = ch.get_sequential_classifier_information(model)
        model_name = text_vectorizer+'/'+class_vectorizer+'/'+classifier_name
        ch.save_results(classifier_name+'_LSTM', metrics, parameters, model_name, classif_level, classif_type, dataset_location)

    print('end testing step')

def train_LSTM_from_web(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, dataset_location):
    print('### LSTM Doing Training ###')

    root_location = fh.get_root_location('data/lstm_outcome/')

    exports_location = fh.join_paths(root_location, "exported_data/")
    matrices_save_location = fh.join_paths(root_location, "fhv_matrices/")
    nn_parameter_search_location = fh.join_paths(root_location, "nn_fhv_parameter_search")
    doc2vec_model_location = fh.link_paths(fh.join_paths(root_location, "doc2vec_model/vocab_model/"), "doc2vec_model")

    load_existing_results = True # it was True
    save_results = True

    sequence_size = 1
    EMBEDDING_SIZE = 150

    model_name = text_vectorizer+'/'+class_vectorizer+'/LSTM'
    results = ch.apply_df_vectorizer(data_frame, text_vectorizer, class_vectorizer, model_name)
    X_train, X_val, y_train, y_val, classes, n_classes, vocab_processor, len_vocabulary = results

    training_docs_list = X_train['patent_id']
    val_docs_list = X_val['patent_id']

    X_data, Xv_data, _ = ch.get_df_data(2, training_docs_list, val_docs_list, None, sequence_size, EMBEDDING_SIZE, doc2vec_model_location)
    GLOBAL_VARS.DOC2VEC_MODEL_NAME, GLOBAL_VARS.MODEL_NAME = wmh.set_parameters_lstm_doc2vec(nn_parameter_search_location, classif_level, classif_type)

    # print(X_data.shape) # 64, 1, 200
    # print(Xt_data.shape) # 20, 1, 200
    # print(Xv_data.shape) # 16, 1, 200

    NN_INPUT_NEURONS, NN_SEQUENCE_SIZE, NN_OUTPUT_NEURONS = pmh.get_lstm_shapes(X_data, n_classes)
    NN_BATCH_SIZE, PARTS_LEVEL, NN_MAX_EPOCHS, QUEUE_SIZE = pmh.get_lstm_basic_parameters()
    param_sampler = pmh.get_lstm_training_parameters()
    EARLY_STOPPER_MIN_DELTA, EARLY_STOPPER_PATIENCE = pmh.get_early_stopping_parameters()

    param_results_dict = dict()
    param_results_path = fh.link_paths(fh.join_paths(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME), NN_PARAMETER_SEARCH_PREFIX.format(classif_type, classif_level, NN_BATCH_SIZE))
    index = param_results_path.rfind('/')
    fh.create_folder(fh.link_paths(fh.join_paths(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME), NN_PARAMETER_SEARCH_PREFIX.format(classif_type, classif_level, NN_BATCH_SIZE))[:index])

    ###########
    # print(X_data.shape)
    # input_size, sequence_size, output_size = X_data.shape[2], X_data.shape[0], n_classes
    # NN_BATCH_SIZE, PARTS_LEVEL, NN_MAX_EPOCHS, QUEUE_SIZE = 1, 2, 200, 100

    # lstm_output_sizes = [500, 1000]
    # w_dropout_options = [0., 0.5]
    # u_dropout_options = [0., 0.5]
    # stack_layers_options = [1, 2, 3]

    # lstm_output_size, w_dropout_do, u_dropout_do, stack_layers = 500, 0.0, 0.0, 1
    # conv_size, conv_filter_length, max_pooling_length = 128, 2, 2

    # EARLY_STOPPER_MIN_DELTA, EARLY_STOPPER_PATIENCE = 0.00001, 15

    # import tensorflow as tf
    # from tensorflow import keras
    # from keras.layers import Input, Dense, Dropout, Activation
    # from keras.models import Model, Sequential
    # from keras.layers.convolutional import MaxPooling1D, Convolution1D
    # from keras.layers.recurrent import LSTM

    # model = Sequential()

    # # model.add(Convolution1D(nb_filter=conv_size, input_shape=(sequence_size, input_size),
    # #                             filter_length=conv_filter_length, border_mode='same', activation='relu'))
    # # model.add(MaxPooling1D(pool_length=max_pooling_length))

    # model.add(LSTM(lstm_output_size, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=False,
    #                input_dim=input_size, dropout_W=w_dropout_do, dropout_U=u_dropout_do,
    #                implementation=1,
    #                return_sequences=False if 1 == stack_layers else True,
    #                go_backwards=False, stateful=False, unroll=False,
    #                name='lstm_{}_w-drop_{}_u-drop_{}_layer_{}'.format(lstm_output_size, str(u_dropout_do),
    #                                                                   str(w_dropout_do), str(1))))
    # # model.add(LSTM(lstm_output_size, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=False,
    # #                input_dim=input_size, dropout_W=w_dropout_do, dropout_U=u_dropout_do,
    # #                implementation=1,
    # #                return_sequences=False if 2 == stack_layers else True,
    # #                go_backwards=False, stateful=False, unroll=False,
    # #                name='lstm_{}_w-drop_{}_u-drop_{}_layer_{}'.format(lstm_output_size, str(u_dropout_do),
    # #                                                                   str(w_dropout_do), str(2))))
    # # model.add(LSTM(lstm_output_size, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=False,
    # #                input_dim=input_size, dropout_W=w_dropout_do, dropout_U=u_dropout_do,
    # #                implementation=1,
    # #                return_sequences=False if 3 == stack_layers else True,
    # #                go_backwards=False, stateful=False, unroll=False,
    # #                name='lstm_{}_w-drop_{}_u-drop_{}_layer_{}'.format(lstm_output_size, str(u_dropout_do),
    # #                                                                   str(w_dropout_do), str(3))))
    # model.add(Dense(output_size, activation='sigmoid', name='sigmoid_output'))
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    # input_matrix = fh.join_paths(matrices_save_location, GLOBAL_VARS.MODEL_NAME)
    # early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=EARLY_STOPPER_MIN_DELTA,
    #                                               patience=EARLY_STOPPER_PATIENCE, verbose=1, mode='auto')
    # metrics_callback = mh.MetricsCallback(input_matrix, classif_type, PARTS_LEVEL, NN_BATCH_SIZE, is_mlp=False)

    # history, yvp, yvp_binary = pmh.fit_predict_generator(model, X_data, y_train, Xv_data, y_val, training_docs_list, val_docs_list, early_stopper, metrics_callback, NN_BATCH_SIZE, NN_MAX_EPOCHS, QUEUE_SIZE)

    # validation_metrics = mh.get_sequential_metrics(y_val, yvp, yvp_binary)
    # mh.display_sequential_metrics(validation_metrics)
    # ###########



    # useful to skip all the already tested models
    if load_existing_results:
        param_results_path = fh.link_paths(fh.join_paths(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME),
                                           NN_PARAMETER_SEARCH_PREFIX.format(classif_type, classif_level, NN_BATCH_SIZE))
        if fh.ensure_exists_path_location(param_results_path):
            print('Loading Previous results in {}'.format(param_results_path))
            param_results_dict = pickle.load(open(param_results_path, 'rb'))
        else:
            print('No Previous results exist in {}'.format(param_results_path))

    for params in param_sampler:
        start_time = time.time()
        lstm_output_size = params['lstm_output_size']
        w_dropout_do = params['w_dropout']
        u_dropout_do = params['u_dropout']
        stack_layers = params['stack_layers']
        conv_size = params['conv_size']
        conv_filter_length = params['conv_filter_length']
        conv_max_pooling_length = params['conv_max_pooling_length']

        GLOBAL_VARS.NN_MODEL_NAME = 'lstm_size_{}_w-drop_{}_u-drop_{}_stack_{}_conv_{}'.format(
            lstm_output_size, w_dropout_do, u_dropout_do, stack_layers, str(conv_size))

        if conv_size:
            GLOBAL_VARS.NN_MODEL_NAME += '_conv-filter-length_{}_max-pooling-size_{}'.format(conv_filter_length,
                                                                                             conv_max_pooling_length)
        if GLOBAL_VARS.NN_MODEL_NAME in param_results_dict.keys():
            print("skipping: {}".format(GLOBAL_VARS.NN_MODEL_NAME))
            continue

        # creating the actual keras model
        model = pmh.get_keras_rnn_model(NN_INPUT_NEURONS, NN_SEQUENCE_SIZE, NN_OUTPUT_NEURONS,
                                       lstm_output_size, w_dropout_do, u_dropout_do, stack_layers, conv_size,
                                       conv_filter_length, conv_max_pooling_length)

        classifier_name, parameters = ch.get_sequential_classifier_information(model)
        model_name = text_vectorizer+'/'+class_vectorizer+'/'+classifier_name

        input_matrix = fh.join_paths(matrices_save_location, GLOBAL_VARS.MODEL_NAME)
        early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=EARLY_STOPPER_MIN_DELTA,
                                                      patience=EARLY_STOPPER_PATIENCE, verbose=1, mode='auto')
        metrics_callback = mh.MetricsCallback(input_matrix, classif_type, PARTS_LEVEL, NN_BATCH_SIZE, is_mlp=False)

        history, yvp, yvp_binary = pmh.fit_predict_generator(model, X_data, y_train, Xv_data, y_val, training_docs_list, val_docs_list, early_stopper, metrics_callback, NN_BATCH_SIZE, NN_MAX_EPOCHS, QUEUE_SIZE)

        print('\nGenerating Validation Metrics')
        validation_metrics = mh.get_sequential_metrics(y_val, yvp, yvp_binary)
        mh.display_sequential_metrics(classifier_name, validation_metrics)

        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME] = dict()
        # param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['best_validation_metrics'] = best_validation_metrics
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['epochs'] = len(history.history['val_loss'])
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['best_weights'] = metrics_callback.best_weights
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['best_val_loss'] = metrics_callback.best_val_loss
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['training_loss'] = metrics_callback.losses
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['validation_loss'] = metrics_callback.val_losses

        duration = time.time() - start_time
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['duration'] = duration

        ch.delete_variables(history, metrics_callback)

        ch.save_results(classifier_name+'_LSTM', validation_metrics, parameters, model_name, classif_level, classif_type, dataset_location)

    if save_results:
        file = open(param_results_path, 'wb')
        pickle.dump(param_results_dict, file)

def test_LSTM_from_web(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, dataset_location):
    print('### LSTM Doing Testing ###')

    root_location = fh.get_root_location('data/lstm_outcome/')

    nn_parameter_search_location = fh.join_paths(root_location, "nn_fhv_parameter_search")
    doc2vec_model_location = fh.link_paths(fh.join_paths(root_location, "doc2vec_model/vocab_model/"), "doc2vec_model")

    save_results = True

    sequence_size = 1
    EMBEDDING_SIZE = 150

    model_name = text_vectorizer+'/'+class_vectorizer+'/LSTM'
    results = ch.apply_df_vectorizer(data_frame, text_vectorizer, class_vectorizer, model_name)
    X_train, X_test, y_train, y_test, classes, n_classes, vocab_processor, len_vocabulary = results

    X_train, X_val, y_train, y_val = ch.get_train_test_from_data(X_train, y_train)

    training_docs_list = X_train['patent_id']
    test_docs_list = X_test['patent_id']
    val_docs_list = X_val['patent_id']

    X_data, Xv_data, Xt_data = ch.get_df_data(3, training_docs_list, val_docs_list, test_docs_list, sequence_size, EMBEDDING_SIZE, doc2vec_model_location)
    GLOBAL_VARS.DOC2VEC_MODEL_NAME, GLOBAL_VARS.MODEL_NAME = wmh.set_parameters_lstm_doc2vec(nn_parameter_search_location, classif_level, classif_type)

    NN_INPUT_NEURONS, NN_SEQUENCE_SIZE, NN_OUTPUT_NEURONS = pmh.get_lstm_shapes(X_data, n_classes)
    NN_BATCH_SIZE, PARTS_LEVEL, NN_MAX_EPOCHS, QUEUE_SIZE = pmh.get_lstm_basic_parameters()
    params = pmh.get_lstm_testing_parameters()
    lstm_output_size,w_dropout_do,u_dropout_do, stack_layers, conv_size, conv_filter_length, conv_max_pooling_length = params
    EARLY_STOPPER_MIN_DELTA, EARLY_STOPPER_PATIENCE = pmh.get_early_stopping_parameters()

    TEST_METRICS_FILENAME = '{}_level_{}_standard_nn_test_metrics_dict.pkl'

    test_metrics_dict = dict()
    test_metrics_path = fh.link_paths(fh.link_paths(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME), TEST_METRICS_FILENAME.format(classif_type, PARTS_LEVEL))

    param_results_path = fh.link_paths(fh.link_paths(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME), NN_PARAMETER_SEARCH_PREFIX.format(classif_type, classif_level, NN_BATCH_SIZE))

    param_results_dict = pickle.load(open(param_results_path, 'rb'))
    GLOBAL_VARS.NN_MODEL_NAME = 'lstm_size_{}_w-drop_{}_u-drop_{}_stack_{}_conv_{}'.format(lstm_output_size,
                                                                                            w_dropout_do,
                                                                                            u_dropout_do,
                                                                                            stack_layers,
                                                                                            str(conv_size)
                                                                                            )
    if conv_size:
        GLOBAL_VARS.NN_MODEL_NAME += '_conv-filter-length_{}_max-pooling-size_{}'.format(conv_filter_length,
                                                                                         conv_max_pooling_length)
    if GLOBAL_VARS.NN_MODEL_NAME not in param_results_dict.keys():
        print("Can't find model: {}".format(GLOBAL_VARS.NN_MODEL_NAME))
        raise Exception()

    if fh.ensure_exists_path_location(test_metrics_path):
        test_metrics_dict = pickle.load(open(fh.link_paths(fh.link_paths(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME), TEST_METRICS_FILENAME.format(classif_type,PARTS_LEVEL)), 'rb'))
        if GLOBAL_VARS.NN_MODEL_NAME in test_metrics_dict.keys():
            print("Test metrics already exist for: {}".format(GLOBAL_VARS.NN_MODEL_NAME))
            test_metrics = test_metrics_dict[GLOBAL_VARS.NN_MODEL_NAME]

            print("** Test Metrics: Cov Err: {:.3f}, Avg Labels: {:.3f}, \n\t\t Top 1: {:.3f}, Top 3: {:.3f}, Top 5: {:.3f}, \n\t\t F1 Micro: {:.3f}, F1 Macro: {:.3f}".format(
                test_metrics['coverage_error'], test_metrics['average_num_of_labels'],
                test_metrics['top_1'], test_metrics['top_3'], test_metrics['top_5'],
                test_metrics['f1_micro'], test_metrics['f1_macro']))
            raise Exception()

    print('***************************************************************************************')
    print(GLOBAL_VARS.NN_MODEL_NAME)

    model = pmh.get_keras_rnn_model(NN_INPUT_NEURONS, NN_SEQUENCE_SIZE, NN_OUTPUT_NEURONS,
                                   lstm_output_size, w_dropout_do, u_dropout_do, stack_layers, conv_size,
                                   conv_filter_length, conv_max_pooling_length)

    # get model best weights
    weights = param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['best_weights']
    model.set_weights(weights)

    print('Evaluating on Test Data using best weights')
    _, ytp, ytp_binary = pmh.predict_generator(None, model, Xt_data, y_test, NN_BATCH_SIZE, QUEUE_SIZE, test_docs_list)

    print('Generating Test Metrics')
    test_metrics = mh.get_sequential_metrics(y_test, ytp, ytp_binary)
    mh.display_sequential_metrics(test_metrics)

    if save_results:
        classifier_name, parameters = ch.get_sequential_classifier_information(model)
        ch.save_results(classifier_name+'_LSTM', test_metrics, parameters, model_name, classif_level, classif_type, dataset_location)

        test_metrics_dict[GLOBAL_VARS.NN_MODEL_NAME] = test_metrics
        pickle.dump(test_metrics_dict, open(test_metrics_path, 'wb'))

if __name__ == '__main__':
    try:
        if len(sys.argv) == 2:
            # source_path = 'test_clean/*/directories - and inside all the patents'
            source_path = sys.argv[1]

            # here the source_path must be passed as a string of the root directory of all data folders!
            source_path, folder_level = th.handle_partial_args(source_path)
        else:
            print(usage())
            sys.exit(1)
    except:
        # source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_classification/cleaned/B/']
        source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_classification/cleaned/B - 1500 patents/']
        # source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_classification/cleaned/B - 9000 patents/']

    text_vectorizer = 'None'
    class_vectorizer = 'multi_label'
    classif_level = 'description_claim_abstract_title'
    classif_type = 'subclasses'

    patent_ids, temp_df, classifications_df = txth.load_data(source_path)
    data_frame, classif_level, classif_type = txth.get_final_df(patent_ids, temp_df)

    # apply_basic_LSTM() # not tested yet
    train_LSTM_from_web(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path)
    # test_LSTM_from_web(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path)

    print("end lstm classifition step")

# add agorithm to display conv metrics, removed a del in 414 classification helper

# TODO list:
# try all the state-of-the art and maybe we can combine two of them.
#
# turn on the validation settings
#
# get_lstm_shapes, try with validation data. change batch size. change testing parameters according to training results