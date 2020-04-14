# -*- coding: utf-8 -*-
import sys
import pandas as pd
import os
import stat
import time
import datetime
import numpy as np

from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras

from keras.utils import np_utils

ssh_source_dir = '/chakrm/workspace/2019-MastersProject/'
sys.path.append(os.path.abspath('..'))
# sys.path.append(os.path.abspath('..')+ssh_source_dir)
from helpers import tool_helper as th
from helpers import folder_helper as fh
from helpers import txt_data_helper as txth
from helpers import classification_helper as ch
from helpers import metrics_helper as mh
from helpers import predicting_model_helper as pmh

script_key = "convolutional"

# def load_data_and_labels(positive_data_file, negative_data_file):
#     """
#     Loads MR polarity data from files, splits the data into words and generates labels.
#     Returns split sentences and labels.
#     """
#     # Load data from files
#     positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
#     positive_examples = [s.strip() for s in positive_examples]
#     negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
#     negative_examples = [s.strip() for s in negative_examples]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generate a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def first_attempt_based_on_text_classification_paper(data_frame, text_vectorizer, class_vectorizer):
    # Parameters - can be placed at the beginning of the script
    # ==================================================

    # Data loading params
    tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
    tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
    tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    # FLAGS._parse_flags()
    # print("\nParameters:")
    # for attr, value in sorted(FLAGS.__flags.items()):
    #     print("{}={}".format(attr.upper(), value))
    # print("")

    dev_sample_percentage = FLAGS.dev_sample_percentage

    # x_train, y_train, vocab_processor, len_vocabulary, x_dev, y_dev, classe, n_classes = preprocess(data_frame, dev_sample_percentage)
    model_name = text_vectorizer+'/'+class_vectorizer+'/TextCNN'
    standard_results = ch.apply_df_vectorizer(data_frame, 'standard', 'multi_label', model_name)
    x_train, x_dev, y_train, y_dev, classes, n_classes, vocab_processor, len_vocabulary = standard_results

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement = FLAGS.allow_soft_placement,
            log_device_placement = FLAGS.log_device_placement)
        sess = tf.Session(config = session_conf)
        with sess.as_default():
            # Code that operates on the default graph and session comes here…
            cnn = pmh.TextCNN(
                sequence_length = x_train.shape[1],
                num_classes = n_classes,
                vocab_size = (len_vocabulary),
                embedding_size = FLAGS.embedding_dim,
                filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters = FLAGS.num_filters)

            # define training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # output directory for models and summaries
            root_location = fh.get_root_location('data/convolutional_runs/')

            timestamp = datetime.datetime.now().isoformat()
            out_dir = fh.link_paths(root_location, timestamp)
            print("Writing to {}\n".format(out_dir))

            # summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = fh.join_paths(fh.link_paths(out_dir, 'summaries'), 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

            # dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = fh.join_paths(fh.link_paths(out_dir, 'summaries'), 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

            # checkpointing
            checkpoint_dir = fh.link_paths(out_dir, 'checkpoints')
            checkpoint_prefix = fh.join_paths(checkpoint_dir, 'model')

            saver = tf.train.Saver(tf.all_variables())

            # write vocabulary
            try:
                vocab_processor.save(os.path.join(out_dir, "vocab"))
            except:
                pass

            # initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                    }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # training loop. For each batch…
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def train_testing_convolution(data_frame, text_vectorizer, class_vectorizer):
    save_standard_sets = True
    root_location = fh.get_root_location('data/convolutional_outcome/')

    sets_location = fh.join_paths(root_location, "model_sets")
    checkpoint_path = fh.join_paths(root_location, "model_checkpoints")
    model_path = fh.link_paths(checkpoint_path, 'convolution_model')
    weights_path = fh.link_paths(checkpoint_path, 'convolution_weights')

    # get sets
    model_name = text_vectorizer+'/'+class_vectorizer+'/NN'
    standard_results = ch.apply_df_vectorizer(data_frame, text_vectorizer, class_vectorizer, model_name)
    train_data, test_data, train_labels, test_labels, classes, n_classes, vocab_processor, len_vocabulary = standard_results
    train_data, val_data, train_labels, val_labels = ch.get_train_test_from_data(train_data, train_labels)

    # save sets
    # ch.save_sets(sets_location, train_data, test_data, val_data, train_labels, test_labels, val_labels,
    #           [classes, n_classes, vocab_processor, len_vocabulary])

    # this is for test
    train_data, test_data, val_data, train_labels, test_labels, val_labels, _ = ch.load_sets(sets_location)

    # it could be that a label is only in the test/data data, might be a problem
    sequence_length = train_data.shape[1]
    # define the model
    model = pmh.get_cnn_test(len_vocabulary, n_classes, sequence_length)

    # calculates metrics with validating data
    model, val_predictions = pmh.run_cnn_test(model,
                                   train_data, train_labels, val_data, val_labels, val_data, val_labels,
                                   model_path, weights_path, True)
    binary_val_predictions = mh.get_binary_0_5(val_predictions)
    print(val_labels.shape)
    print(val_predictions.shape)
    # display validation metrics
    metrics = mh.get_sequential_metrics(val_labels, val_predictions, binary_predictions)
    mh.display_sequential_metrics('validation convolution sequence', metrics)


def test_testing_convolution(text_vectorizer, class_vectorizer, classif_level, classif_type, dataset_location):
    load_standard_sets = True
    root_location = fh.get_root_location('data/convolutional_outcome/')

    sets_location = fh.join_paths(root_location, "model_sets")
    checkpoint_path = fh.join_paths(root_location, "model_checkpoints")
    model_path = fh.link_paths(checkpoint_path, 'convolution_model')
    weights_path = fh.link_paths(checkpoint_path, 'convolution_weights')

    # load sets and settings
    model_name = text_vectorizer+'/'+class_vectorizer+'/NN'
    # train_data, test_data, val_data, train_labels, test_labels, val_labels, settings = ch.load_sets(sets_location, '2020-01-08T22:28:44.757410')
    # classes, n_classes, vocab_processor, len_vocabulary = settings

    # only for test
    train_data, test_data, val_data, train_labels, test_labels, val_labels, _ = ch.load_sets(sets_location)

    # it could be that a label is only in the test/data data, might be a problem
    sequence_length = train_data.shape[1]
    # calculates metrics with testing data
    model, predictions = pmh.run_cnn_test(_,
                                   train_data, train_labels, test_data, test_labels, val_data, val_labels,
                                   model_path, weights_path, False)
    binary_predictions = mh.get_binary_0_5(predictions)

    # display testing metrics
    metrics = mh.get_sequential_metrics(test_labels, predictions, binary_predictions)
    mh.display_sequential_metrics('testing convolution sequence', metrics)

    classifier_name, layers = ch.get_sequential_classifier_information(model)
    ch.save_results(classifier_name+' Test', metrics, layers, model_name, classif_level, classif_type, dataset_location)
    # i can also draw a graph over time of accuracy and loss

def second_attempt_from_web(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, dataset_location):

    root_location = fh.get_root_location('data/convolutional_outcome/')

    # imdb = keras.datasets.imdb
    # (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # explore data
    # print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    # print("how data looks like: ", train_data[0]) # [1, 14, 34...]
    # print("how labels looks like: ", train_labels[0]) # 0

    # preprocess data
    # A dictionary mapping words to an integer index
    # word_index = imdb.get_word_index()

    # The first indices are reserved, thus it increases all the indices by 3
    # word_index = {k : (v + 3) for k, v in word_index.items()}
    # word_index["<PAD>"] = 0
    # word_index["<START>"] = 1
    # word_index["<UNK>"] = 2  # unknown
    # word_index["<UNUSED>"] = 3

    # make both the train and the test dataset the same length
    # train_data = keras.preprocessing.sequence.pad_sequences(train_data,
    #                                                         value=word_index["<PAD>"],
    #                                                         padding='post',
    #                                                         maxlen=256)

    # test_data = keras.preprocessing.sequence.pad_sequences(test_data,
    #                                                        value=word_index["<PAD>"],
    #                                                        padding='post',
    #                                                        maxlen=256)
    # print(train_data[0]) # [1 14 34 0 0 0] - with little difference: PAD,START,UNK,UNUSED

    model_name = text_vectorizer+'/'+class_vectorizer+'/NN'
    standard_results = ch.apply_df_vectorizer(data_frame, text_vectorizer, class_vectorizer, model_name)
    train_data, test_data, train_labels, test_labels, classes, n_classes, vocab_processor, len_vocabulary = standard_results

    train_data, val_data, train_labels, val_labels = ch.get_train_test_from_data(train_data, train_labels)

    # print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    # print("how data looks like: ", train_data[0]) # [1 14 34 0 0 0]
    # print("how labels looks like: ", train_labels[0:5]) # list of lists [[0 1 0 0 0 1 0 1], ...]

    model = pmh.get_text_convolutional_from_web(len_vocabulary, n_classes)
    metrics, predictions = pmh.run_text_cnn_model(model, train_data, train_labels, test_data, test_labels)

    classifier_name, layers = ch.get_sequential_classifier_information(model)
    mh.display_convolutional_metrics(classifier_name, metrics[0], metrics[1], metrics[2], test_labels, predictions)

    ch.save_results(classifier_name, metrics, layers, model_name, classif_level, classif_type, dataset_location)
    # i can also draw a graph over time of accuracy and loss

def third_attempt_from_web(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, dataset_location):
    # fashion_mnist = keras.datasets.fashion_mnist
    # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # print("###  images  ###")
    # print(test_images)
    # print(type(test_images))
    # print(test_images.shape)
    # print(test_images[0])

    # print(test_labels)
    # print(type(test_labels))
    # print(test_labels.shape)
    # print(test_labels[0])

    model_name = text_vectorizer+'/'+class_vectorizer+'/CNN'
    standard_results = ch.apply_df_vectorizer(data_frame, text_vectorizer, class_vectorizer, model_name)
    train_data, test_data, train_labels, test_labels, classes, n_classes, vocab_processor, len_vocabulary = standard_results

    # print("###  texts  ###")
    # print(test_data)
    # print(type(test_data))
    # print(test_data.shape)
    # print(test_data[0])

    # print(test_labels)
    # print(type(test_labels))
    # print(test_labels.shape)
    # print(test_labels[0])

    # preprocess data
    # train_images = train_images / 255.0
    # test_images = test_images / 255.0
    # it reduced all in the range 0--1

    # normalize the values

    train_data = np.reshape(train_data, [train_data.shape[0], 1, train_data.shape[1]])
    test_data = np.reshape(test_data, [test_data.shape[0], 1, test_data.shape[1]])

    print(train_labels)
    print(train_labels.shape)

    for index in range(train_labels.shape[1]):
        temp_train_labels = train_labels[:, index] # estimating the first class
        temp_test_labels = test_labels[:, index] # estimating the first class

        print(temp_train_labels)

        model = pmh.get_image_convolutional_from_web(train_data, n_classes)

        metrics, predictions = pmh.run_image_cnn_model(model, train_data, temp_train_labels, test_data, temp_test_labels)

        print("predictions: ", predictions)

        classifier_name, layers = ch.get_sequential_classifier_information(model)
        mh.display_convolutional_metrics(classifier_name, metrics[0], metrics[1], metrics[2], temp_test_labels, predictions)

    ch.save_results(classifier_name, metrics, layers, model_name, classif_level, classif_type, dataset_location)

def fourth_attemp_from_web(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, dataset_location):
    model_name = text_vectorizer+'/'+class_vectorizer+'/CNN'
    standard_results = ch.apply_df_vectorizer(data_frame, text_vectorizer, class_vectorizer, model_name)
    train_data, test_data, train_labels, test_labels, classes, n_classes, vocab_processor, len_vocabulary = standard_results

    # TODO: is it useful?
    # test_labels = np_utils.to_categorical(test_labels, n_classes)

    model = pmh.get_fourth_attempt_model_from_web(train_data, n_classes)

    y_train_predclass, y_test_predclass, train_metrics, test_metrics, train_predictions, test_predictions = pmh.run_fourth_attempt_model(model, train_data, train_labels, test_data, test_labels)

    # mh.display_convolution_metrics_fourth_attempt(train_labels, test_labels, y_train_predclass, y_test_predclass)
    # mh.display_convolutional_metrics(classifier_name, train_metrics[0], train_metrics[1], train_metrics[2], train_labels, train_predictions)
    mh.display_convolutional_metrics(classifier_name, test_metrics[0], test_metrics[1], test_metrics[2], test_labels, test_predictions)

    classifier_name, layers = ch.get_sequential_classifier_information(model)
    ch.save_results(classifier_name, test_metrics, layers, model_name, classif_level, classif_type, dataset_location)

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
        source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_classification/cleaned/B/']
        # source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_classification/cleaned/B - 1500 patents/']
        # source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_classification/cleaned/B - 1500 patents/', '/Users/elio/Desktop/Patent-Classification/data/test_classification/cleaned/B/']

    text_vectorizer = 'standard'
    class_vectorizer = 'multi_label'
    classif_level = 'description_claim_abstract_title'
    classif_type = 'subclasses'

    patent_ids, temp_df, classifications_df = txth.load_data(source_path)
    data_frame, classif_level, classif_type = txth.get_final_df(patent_ids, temp_df)

    # train and save the model
    # train_testing_convolution(data_frame, text_vectorizer, class_vectorizer)

    # load and test the model
    test_testing_convolution(text_vectorizer, class_vectorizer, classif_level, classif_type, source_path)

    # first_attempt_based_on_text_classification_paper(data_frame, text_vectorizer, class_vectorizer) # not efficent 400 step
    # second_attempt_from_web(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path) # it should switch each class in fit and predict! - but it was written without the association - try to increase the epochs
    # worked on text data
    # third_attempt_from_web(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path) # it should switch each class in fit and predict! - i dont like the reshape
    # worked on image data
    # fourth_attemp_from_web(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path) # i should test it, for sure something is wrong such as the method to_categoricals ----- shapes problem.....

    print("end cnn classification step")

# TODO:
# the problem is tuning the model, how many (if we need it) conv-layers and which are
#
# TODO:
# normalize values out of text vectorizer, such as for images they divided by 255.0 (max value?) - DONE