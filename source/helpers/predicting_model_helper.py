# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterSampler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
# from skmultilearn.adapt import MLkNN

# import tensorflow as tf
# from tensorflow import keras
# from keras.layers import Input, Dense, Dropout, Activation
# from keras.models import Model, Sequential
# from keras.layers.convolutional import MaxPooling1D, Convolution1D
# from keras.layers.recurrent import LSTM

import fasttext

sys.path.append(os.path.abspath('..'))
from helpers import classification_helper as ch
from helpers import metrics_helper as mh

# from keras.backend import sigmoid
# def swish(x, beta = 1):
#     return (x * sigmoid(beta * x))
# from keras.utils.generic_utils import get_custom_objects
# from keras.layers import Activation
# get_custom_objects().update({'swish': Activation(swish)})

#########################################################################################################################
# MACHINE LEARNING :

def get_logistic():
    return LogisticRegression(C=1e5, solver='lbfgs', multi_class='auto', max_iter=2500)
    # return LogisticRegression(C=1e5, solver='sag', multi_class='multinomial')

def get_SGD_classifier():
    return SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)

def get_SVC():
    return SVC(C=5.0, kernel='sigmoid', degree=1, gamma='scale', decision_function_shape='ovo')

def get_linear_SVC():
    return LinearSVC()

def get_decision_tree():
    return DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=150, max_features='log2')

# def get_random_forest_regressor():
#     return RandomForestRegressor(n_estimators=10)

def get_random_forest_classifier():
    return RandomForestClassifier(n_estimators=150, criterion='entropy', max_depth=10, max_features='auto', n_jobs=-1)

def get_extra_tree():
    return ExtraTreesClassifier(n_estimators=200)

def get_kneighbors():
    return KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size=45, p=2, n_jobs=-1)

def get_multinomialNB():
    return MultinomialNB()

def get_gaussianNB():
    return GaussianNB()

def get_MLkNN():
    return MLkNN(k=10)

def fit_predict_functions(model, X_train, y_train, X_test):
    model.fit(X_train, y_train) # ValueError: setting an array element with a sequence.
    return model.predict(X_test)

#########################################################################################################################
# DEEP LEARNING :
#########################################################################################################################
# CONVOLUTIONAL :

def get_cnn():
    len_vocabulary, n_classes = 9924, 62
    # build up the model
    vocab_size = len_vocabulary # it was 10.000
    dense_layer_size = n_classes*2 # it was 16, when the last was 8
    dense_layer_size_middleware_1 = n_classes*1.75
    dense_layer_size_middleware_2 = n_classes*1.25
    dense_layer_size_middleware_2 = n_classes*1.25
    num_classes = n_classes

    dropout_rate_1 = 0.1
    dropout_rate_2 = 0.1
    dropout_rate_3 = 0.1

    # set up the layers
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, dense_layer_size))

    model.add(keras.layers.ZeroPadding1D((1,1), input_shape=(vocab_size, dense_layer_size)))
    model.add(keras.layers.Conv1D(filters=10, kernel_size=10, strides=10, activation=tf.nn.relu))
    model.add(keras.layers.GlobalMaxPooling1D())

    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dropout(dropout_rate_1))
    model.add(keras.layers.Dense(dense_layer_size, activation=tf.nn.relu))
    # model.add(keras.layers.Dropout(dropout_rate_2))
    model.add(keras.layers.Dense(dense_layer_size_middleware_1, activation=tf.nn.relu))
    model.add(keras.layers.Dense(dense_layer_size_middleware_2, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(dropout_rate_3))
    model.add(keras.layers.Dense(num_classes, activation=tf.nn.sigmoid))
    # it was 1, after 8 as sectors, sigmoid may be not suitable for multiclass
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])
    return model

def get_cnn_test(len_vocabulary, n_classes, sequence_length):
    # build up the model
    vocab_size = len_vocabulary # it was 10.000
    dense_layer_size = n_classes*2 # it was 16, when the last was 8
    num_classes = n_classes

    embedding_size = 128

    # set up the layers
    model = keras.Sequential()
    # W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
    model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=num_classes, embeddings_initializer='random_uniform'))
    #######
    # model.add(keras.layers.Reshape((64,embedding_size,num_classes)))
    model.add(keras.layers.Conv1D(filters=embedding_size, kernel_size=3, strides=5, padding='valid', activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling1D(pool_size=embedding_size, strides=1, padding='valid'))

    model.add(keras.layers.Conv1D(filters=embedding_size, kernel_size=4, strides=5, padding='valid', activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling1D(pool_size=embedding_size, strides=1, padding='valid'))

    model.add(keras.layers.Conv1D(filters=embedding_size, kernel_size=5, strides=5, padding='valid', activation=tf.nn.relu))
    # model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'))
    model.add(keras.layers.GlobalMaxPooling1D())
    #######
    # model.add(keras.layers.Reshape((num_classes, )))
    # model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation=tf.nn.sigmoid))
    # it was 1, after 8 as sectors, sigmoid may be not suitable for multiclass
    model.summary()
    return model

def get_text_convolutional_from_web(len_vocabulary, n_classes):
    # build up the model
    vocab_size = len_vocabulary # it was 10.000
    dense_layer_size = n_classes*2 # it was 16, when the last was 8
    # dense_layer_size = n_classes*100
    dense_layer_size_middleware_1 = n_classes*1.75
    # dense_layer_size_middleware_1 = n_classes/2
    dense_layer_size_middleware_2 = n_classes*1.5
    # dense_layer_size_middleware_2 = dense_layer_size_middleware_2/10
    dense_layer_size_middleware_3 = n_classes*1.25
    # dense_layer_size_middleware_3 = dense_layer_size_middleware_2/10
    num_classes = n_classes

    # set up the layers
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, dense_layer_size))

    model.add(keras.layers.ZeroPadding1D((1,1), input_shape=(vocab_size, dense_layer_size)))
    model.add(keras.layers.Conv1D(filters=25, kernel_size=25, strides=25, activation=tf.nn.relu))
    model.add(keras.layers.AveragePooling1D())

    model.add(keras.layers.ZeroPadding1D((1,1), input_shape=(vocab_size, dense_layer_size)))
    model.add(keras.layers.Conv1D(filters=75, kernel_size=75, strides=75, activation=tf.nn.relu))
    model.add(keras.layers.GlobalMaxPooling1D())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(dense_layer_size, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.15))
    model.add(keras.layers.Dense(dense_layer_size_middleware_1, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(dense_layer_size_middleware_2, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(num_classes, activation=tf.nn.sigmoid))
    # it was 1, after 8 as sectors, sigmoid may be not suitable for multiclass
    model.summary()
    return model

def get_image_convolutional_from_web(train_data, n_classes):
    # build up the model
    flatten_size = (1, train_data.shape[2])
    dense_size = 128
    num_classes = n_classes

    # set up the layers of the model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=flatten_size), # it was (28, 28) - it should be the size of the vocabulary and the number of training patents, but i can't. I may not need this layer!!!!!
        keras.layers.Dense(dense_size, activation=tf.nn.relu), # it was 128
        keras.layers.Dense(num_classes, activation=tf.nn.softmax) # it was 10 - it is range of values of predictions: [0, 10) -> 10 classes
    ])
    return model

def get_fourth_attempt_model_from_web(train_data, n_classes):
    dense_input_size = train_data.shape[1] # it was 10000
    dense_size = 1000
    dense_size_2 = 500
    dense_size_3 = 50
    dense_size_4 = n_classes

    # #del model
    model = Sequential()
    model.add(Dense(dense_size, input_shape=(dense_input_size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_size_2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_size_3))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_size_4))
    model.add(Activation('softmax'))
    return model

def run_text_cnn_model(model, train_data, train_labels, test_data, test_labels):
    print('###  CNN  ###')
    # compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'mse'])

    train_data, val_data, train_labels, val_labels = ch.get_train_test_from_data(train_data, train_labels)

    # TODO: convert to multi label classification
    # train_labels = train_labels[:, 0]
    # val_labels = val_labels[:, 0]

    # train the model
    history = model.fit(train_data,
                        train_labels,
                        epochs=40,
                        batch_size=512,
                        validation_data=(val_data, val_labels),
                        verbose=1)

    # check on the test dataset
    test_loss, test_acc, test_mse = model.evaluate(test_data, test_labels)
    predictions = model.predict(test_data)
    return [test_loss, test_acc, test_mse], predictions

def run_cnn_test(model, train_data, train_labels, test_data, test_labels, val_data, val_labels, model_path, weights_path, training_flag):
    print('###  CNN  ###')

    if not training_flag:
        model = tf.keras.models.load_model(model_path)
        # weights = model.load_weights(weights_path)

    # compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'mse'])

    min_delta, patience = 0.00001, 15
    early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta,
                                                  patience=patience, verbose=1, mode='auto')
    metrics_callback = mh.MetricsCNNCallback(val_data, val_labels, patience) # should add it to the list of callbacks
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                          save_weights_only=True,
    #                                                          verbose=1,
    #                                                          period=5)

    # train the model
    history = model.fit(train_data,
                        train_labels,
                        epochs=200,
                        batch_size=64,
                        validation_data=(val_data, val_labels),
                        callbacks=[early_stopper, metrics_callback],
                        verbose=1)
    if training_flag:
        model.save(model_path)
        # model.save_weights(weights_path)

    predictions = model.predict(test_data,
                                batch_size=64,
                                steps=1,
                                max_queue_size=100,
                                ocallbacks=[early_stopper, metrics_callback],
                                verbose=1)
    return model, predictions

def run_image_cnn_model(model, train_data, train_labels, test_data, test_labels):
    print('###  CNN  ###')
     # compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'mse'])

    train_data, val_data, train_labels, val_labels = ch.get_train_test_from_data(train_data, train_labels)

    # TODO:
    # test if it is multi label or not
    # train the model
    model.fit(train_data,
              train_labels,
              epochs=5)
              #   ,
              # batch_size=512,
              # validation_data=(val_data, val_labels),
              # verbose=1)

    # check on the test dataset
    test_loss, test_acc, test_mse = model.evaluate(test_data, test_labels)
    # make predictions
    predictions = model.predict(test_data)
    return [test_loss, test_acc, test_mse], predictions

def run_fourth_attempt_model(model, train_data, train_labels, test_data, test_labels):
    print('###  CNN  ###')
    batch_size = 5
    nb_epochs = 20
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'mse'])

    model.fit(train_data, train_labels, batch_size=batch_size, epochs=nb_epochs, verbose=1)

    y_train_predclass = model.predict_classes(train_data, batch_size=batch_size)
    y_test_predclass = model.predict_classes(test_data, batch_size=batch_size)

    train_loss, train_acc, train_mse = model.evaluate(train_data, train_labels)
    test_loss, test_acc, test_mse = model.evaluate(test_data, test_labels)

    train_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)

    return y_train_predclass, y_test_predclass, [train_loss, train_acc, train_mse], [test_loss, test_acc, test_mse], train_predictions, test_predictions

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):
        print('###  CNN  ###')
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding layer
        with tf.device('cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

#########################################################################################################################
# FASTTEXT :

def get_fasttext(training_path):
    print('### training model ###')
    # epoch: 5-50 (how many times it loops over your data)
    # lr: 0.1-1 (larger it is, faster it converges to a solution)
    # wordsNgrams: 1-5 (ngrams for words)
    # loss: hs (if it is too slow), one-vs-all/ova (indipendent binary classifier)
    # k: -1 (as many predictions as possible), 1 (only one), 5 (5 predictions) etc..
    # threshold: 0.1-1 (0.5 ideally) of the predictions

    # pretrainedVectors='/Users/elio/Desktop/Patent-Classification/data/crawl-300d-2M.vec'
    # pretrainedVectors='/Users/elio/Desktop/Patent-Classification/data/wiki-news-300d-1M.vec'
    # return fasttext.train_supervised(input=training_path, dim=300, minCount=4, minn=0, maxn=0, epoch=50, lr=.618225, loss='softmax', bucket=937478, wordNgrams=2), '[dim=382, minn=0, maxn=0, epoch=1, lr=0.835603, loss=softmax, bucket=0, wordNgrams=1]'
    return fasttext.train_supervised(input=training_path, dim=300, minCount=4, minn=0, maxn=0, epoch=200, lr=.618225, loss='softmax', bucket=937478, wordNgrams=2, pretrainedVectors='/Users/elio/Desktop/Patent-Classification/data/crawl-300d-2M.vec'), '[dim=382, minn=0, maxn=0, epoch=1, lr=0.835603, loss=softmax, bucket=0, wordNgrams=1]'
    # return fasttext.train_supervised(input=training_path, dim=300, minn=2, maxn=5, epoch=25, lr=0.5, loss='ova', bucket=200000), '[dim=300, minn=2, maxn=5, epoch=25, lr=0.5, loss=ova, bucket=200000, wordNgrams=2]' # wordNgrams 1 2 3, lr 0.05 0.1 0.25, dim 100 200 300, ws 5 10 25, epochs 5 50 100, loss ns hs softmax

def predict_test_fasttext(model, testing_path):
    print('### calculating precitions ###')

    y_true = np.array([])
    y_pred = np.array([])
    texts = pd.read_csv(testing_path)
    for index, row in texts.iterrows():
        if isinstance(row, pd.Series):
            text, classes = row.tolist()
            # result = model.predict_prob(text[0], '__label__', k=-1, threshold=0.0000001)
            # print(result)
            try:
                y_predicted, metrics_array = model.predict(text[0], k=-1, threshold=0.05)
            except:
                print('no predictions')
                continue
            print(classes)
            y_true = np.append(y_true, classes.split())
            y_pred = np.append(y_pred, list(y_predicted))

    print(y_true)
    print(y_true.shape)
    print(y_pred)
    print(y_pred.shape)

    print('### calculating metrics ###')
    result = model.test(testing_path, k=-1)
    return y_true, y_pred, result

def save_fasttext_model(model, bin_path):
    model.save_model(bin_path)

def load_fasttext_model(model, bin_path):
    model.load_model(bin_path)

#########################################################################################################################
# LSTM :

def get_lstm_shapes(training_data, n_classes):
    # thus, i should convert every document to a list of integers with fixed length (i suppose)
    # NN_INPUT_NEURONS = Xt.shape[2] # number of words (i guess)
    # NN_SEQUENCE_SIZE = Xt.shape[1] # number of documents (i guess)
    NN_INPUT_NEURONS = training_data.shape[2] # 200
    # NN_SEQUENCE_SIZE = training_data.shape[1] # 1
    NN_SEQUENCE_SIZE = training_data.shape[0] # 1213
    # NN_SEQUENCE_SIZE = Xv_data.shape[0] # 16
    NN_OUTPUT_NEURONS = n_classes
    # NN_OUTPUT_NEURONS = classification_types[classif_type]
    return NN_INPUT_NEURONS, NN_SEQUENCE_SIZE, NN_OUTPUT_NEURONS

def get_lstm_basic_parameters():
    # TODO:
    NN_BATCH_SIZE = 1
    PARTS_LEVEL = 2 # it is used to define the format for creating the path to training and testing set
    NN_MAX_EPOCHS = 200
    QUEUE_SIZE = 100
    return NN_BATCH_SIZE, PARTS_LEVEL, NN_MAX_EPOCHS, QUEUE_SIZE

def get_lstm_training_parameters():
    # TODO: tuning here
    lstm_output_sizes = [500, 1000]
    # w_dropout_options = [None, 0.5] # i have received a problem on None value, it can't be handled
    w_dropout_options = [0., 0.5]
    # u_dropout_options = [None, 0.5] # i have received a problem on None value, it can't be handled
    u_dropout_options = [0., 0.5]
    stack_layers_options = [1, 2, 3]
    conv_size_options = [None]
    conv_filter_length_options = [None]
    conv_max_pooling_length_options = [None]

    NN_RANDOM_SEARCH_BUDGET = 10
    NN_PARAM_SAMPLE_SEED = 1234
    # MODEL_VERBOSITY = 1
    param_sampler = ParameterSampler({
        'lstm_output_size': lstm_output_sizes,
        'w_dropout': w_dropout_options,
        'u_dropout': u_dropout_options,
        'stack_layers': stack_layers_options,
        'conv_size': conv_size_options,
        'conv_filter_length': conv_filter_length_options,
        'conv_max_pooling_length': conv_max_pooling_length_options,
    }, n_iter=NN_RANDOM_SEARCH_BUDGET, random_state=NN_PARAM_SAMPLE_SEED)
    return param_sampler

def get_lstm_testing_parameters():
    # TODO:
    # i should train each model first and then i use it
    lstm_output_size = 500
    w_dropout_do = 0.5
    u_dropout_do = 0.5
    stack_layers = 2
    conv_size = None
    conv_filter_length = None
    conv_max_pooling_length = None
    return [lstm_output_size, w_dropout_do, u_dropout_do, stack_layers, conv_size, conv_filter_length, conv_max_pooling_length]

def get_early_stopping_parameters():
    EARLY_STOPPER_MIN_DELTA = 0.00001
    EARLY_STOPPER_PATIENCE = 15
    return EARLY_STOPPER_MIN_DELTA, EARLY_STOPPER_PATIENCE

def get_keras_rnn_model(input_size, sequence_size, output_size, lstm_output_size, w_dropout_do, u_dropout_do,
                               stack_layers=1, conv_size=None, conv_filter_length=3, max_pooling_length=None):
    """
    Creates an LSTM keras Model - types: RNN (as a class), SIMPLERNN, LSTM, GRU, LSTM, CONVLSTM2D (input/recurrent transformations are conv)
    """
    model = Sequential()
    if conv_size:
        model.add(Convolution1D(nb_filter=conv_size, input_shape=(sequence_size, input_size),
                                filter_length=conv_filter_length, border_mode='same', activation='relu'))
        if max_pooling_length is not None:
            # pool_length:
            model.add(MaxPooling1D(pool_length=max_pooling_length))
    for i in range(stack_layers):
        model.add(LSTM(lstm_output_size, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=False,
                       input_dim=input_size, dropout_W=w_dropout_do, dropout_U=u_dropout_do,
                       implementation=1,
                       return_sequences=False if i + 1 == stack_layers else True,
                       go_backwards=False, stateful=False, unroll=False,
                       name='lstm_{}_w-drop_{}_u-drop_{}_layer_{}'.format(lstm_output_size, str(u_dropout_do),
                                                                          str(w_dropout_do), str(i + 1))))
    model.add(Dense(output_size, activation='sigmoid', name='sigmoid_output'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

def predict_generator(history, model, x_data, y_label, NN_BATCH_SIZE, QUEUE_SIZE, docs_list):
    y_pred = model.predict_generator(
        # generator=ch.batch_generator(Xv_file, yv_file, NN_BATCH_SIZE QUEUE_SIZE, is_mlp=False, validate=True),
        generator=ch.batch_generator(x_data, y_label, NN_BATCH_SIZE, QUEUE_SIZE, is_mlp=False, validate=True),
        max_q_size=QUEUE_SIZE,
        val_samples=len(docs_list))
    y_pred_binary = mh.get_binary_0_5(y_pred)
    return history, y_pred, y_pred_binary

def fit_predict_generator(model, x_data, y_train, val_data, y_val, training_docs_list, val_docs_list, early_stopper, metrics_callback, NN_BATCH_SIZE, NN_MAX_EPOCHS, QUEUE_SIZE):
    # Model Fitting
    history = model.fit_generator(
        # generator=ch.batch_generator(X_file, y_file, NN_BATCH_SIZE, QUEUE_SIZE, is_mlp=False, validate=False),
        generator=ch.batch_generator(x_data, y_train, NN_BATCH_SIZE, QUEUE_SIZE, is_mlp=False, validate=False),
        # validation_data=ch.batch_generator(Xv_file, yv_file, NN_BATCH_SIZE, QUEUE_SIZE, is_mlp=False, validate=True),
        validation_data=ch.batch_generator(val_data, y_val, NN_BATCH_SIZE, QUEUE_SIZE, is_mlp=False, validate=True),
        samples_per_epoch=len(training_docs_list),
        nb_val_samples=len(val_docs_list),
        nb_epoch=NN_MAX_EPOCHS,
        callbacks=[early_stopper, metrics_callback],
        max_q_size=QUEUE_SIZE)

    # using the recorded weights of the best recorded validation loss
    last_model_weights = model.get_weights()
    model.set_weights(metrics_callback.best_weights)

    print('Evaluating on Validation Data using saved best weights')
    return predict_generator(history, model, val_data, y_val, NN_BATCH_SIZE, QUEUE_SIZE, val_docs_list)
