# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import coverage_error
from sklearn.metrics import classification_report

# from tensorflow import keras
# from keras.callbacks import Callback

#########################################################################################################################
# MACHINE LEARNING:
#########################################################################################################################
# basic metrics:

def get_accuracy_score(y_true, y_pred):
    return round(accuracy_score(y_true, y_pred), 5)

def get_precision_score(y_true, y_pred, average):
    if average:
        return round(precision_score(y_true, y_pred, average=average), 5)
    return round(precision_score(y_true, y_pred), 5)

def get_recall_score(y_true, y_pred, average):
    if average:
        return round(recall_score(y_true, y_pred, average=average), 5)
    return round(recall_score(y_true, y_pred), 5)

def get_f1_score(y_true, y_pred, average):
    if average:
        return round(f1_score(y_true, y_pred, average=average), 5)
    return round(f1_score(y_true, y_pred), 5)

def get_cross_val_score(algorithm, vectorized_data, classifications, scoring):
    return cross_val_score(algorithm, vectorized_data, classifications, cv=5, scoring=scoring)

def calculate_recall_curve_precision_score(y_test_unfolded, y_score_unfolded, average, y_test, y_score):
    # A "micro-average": quantifying score on all classes jointly
    precision, recall, _ = precision_recall_curve(y_test_unfolded, y_score_unfolded)
    if average:
        average_precision = average_precision_score(y_test, y_score, average=average)
    else:
        average_precision = average_precision_score(y_test, y_score)
    return precision, recall, average_precision

#########################################################################################################################
# micro and macro average metrics:

def display_directly_metrics(algorithm, accuracy, precision, recall, f1):
    print('@@@  ', algorithm ,'  @@@')

    print('test accuracy : {0:0.5f}'.format(accuracy))
    print('test precision : {0:0.5f}'.format(precision))
    print('test recall : {0:0.5f}'.format(recall))
    print('test f1 : {0:0.5f}'.format(f1))

def display_metrics(algorithm, y_true, y_pred, average, accuracy):
    try:
        # this should be changed by comparing all the possibilities for specified text (i can use the original dataframe!)
        print(str(average)+"-average quality numbers")
        precision = get_precision_score(y_true, y_pred, average)
        recall = get_recall_score(y_true, y_pred, average)
        f1 = get_f1_score(y_true, y_pred, average)

        display_directly_metrics(algorithm, accuracy, recall, precision, f1)
        return accuracy, precision, recall, f1
    except:
        return -1, -1, -1, -1

def calculate_metrics_for_crossvalidation(algorithm, vect_data, classification):
    cross_results_accuracy = get_cross_val_score(algorithm, vect_data, classification, 'accuracy')
    cross_results_precision = get_cross_val_score(algorithm, vect_data, classification, 'precision')
    cross_results_recall = get_cross_val_score(algorithm, vect_data, classification, 'recall')

    return cross_results_accuracy, cross_results_recall, cross_results_precision

def calculate_metrics_with_recall_curve(y_score, y_train, y_test, y_train_pred, y_pred):
    print("METRICS: fitting and predicting for all classes at same time")

    print('TRAINING accuracy score: {0:0.2f}'.format(get_accuracy_score(y_train.ravel(), y_train_pred.ravel())))

    precision = dict()
    recall = dict()
    average_precision = dict()

    precision['micro'], recall['micro'], average_precision['micro'] = calculate_recall_curve_precision_score(y_test.ravel(), y_score.ravel(), 'micro', y_test, y_score)

    print('\nAverage recall score, micro-averaged over all classes: {0:0.2f}'.format(np.mean(recall['micro'])))
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision['micro']))

    precision['macro'], recall['macro'], average_precision['macro'] = calculate_recall_curve_precision_score(y_test.ravel(), y_score.ravel(), 'macro', y_test, y_score)

    print('Average recall score, macro-averaged over all classes: {0:0.2f}'.format(np.mean(recall['macro'])))
    print('Average precision score, macro-averaged over all classes: {0:0.2f}'.format(average_precision['macro']))

def calculate_metrics(algorithm, y_true, y_pred):
    print('\n###  calculating_metrics  ###')
    try:
        if y_true.shape[1] > 1 or y_pred.shape[1] > 1:
            y_true_unrolled = y_true.ravel()
            y_pred_unrolled = y_pred.ravel()
    except:
        print('exception in calculate metrics - line 110 metrics_helper.py')
        pass

    accuracy = get_accuracy_score(y_true, y_pred)

    none_average_metrics = display_metrics(algorithm, y_true, y_pred, None, accuracy)
    binary_average_metrics = display_metrics(algorithm, y_true, y_pred, 'binary', accuracy)
    micro_average_metrics = display_metrics(algorithm, y_true, y_pred, 'micro', accuracy)
    macro_average_metrics = display_metrics(algorithm, y_true, y_pred, 'macro', accuracy)

    print('\n### Unrolled Predictions ###')

    accuracy = get_accuracy_score(y_true_unrolled, y_pred_unrolled)

    none_average_metrics = display_metrics(algorithm, y_true_unrolled, y_pred_unrolled, None, accuracy)
    binary_average_metrics = display_metrics(algorithm, y_true_unrolled, y_pred_unrolled, 'binary', accuracy)
    micro_average_metrics = display_metrics(algorithm, y_true_unrolled, y_pred_unrolled, 'micro', accuracy)
    macro_average_metrics = display_metrics(algorithm, y_true_unrolled, y_pred_unrolled, 'macro', accuracy)

    return none_average_metrics, binary_average_metrics, micro_average_metrics, macro_average_metrics

def calculate_manual_metrics(algorithm, y_true, y_pred):
    print('### Calculating Manual Metrics ###')
    a, b = y_true.T.copy(order = 'C'), y_pred.T.copy(order = 'C')

    count = 0
    count_tp, count_fp, count_tn, count_fn = 0, 0, 0, 0
    temp_count_tp, temp_count_fp, temp_count_tn, temp_count_fn = 0, 0, 0, 0
    temp_precision, temp_recall = [], []
    for y_test_el, y_pred_el in zip(np.nditer(a), np.nditer(b)):
        # print(y_test_el, y_pred_el, end=', ')
        count += 1
        if y_pred_el == 1 and y_test_el == 1:
            temp_count_tp += 1
        if y_pred_el == 1 and y_test_el == 0:
            temp_count_fp += 1
        if y_pred_el == 0 and y_test_el == 0:
            temp_count_tn += 1
        if y_pred_el == 0 and y_test_el == 1:
            temp_count_fn += 1
        if count % y_true.shape[0] == 0:
            # print('')
            if temp_count_tp+temp_count_fp != 0:
                precision = temp_count_tp/(temp_count_tp+temp_count_fp)
            else:
                precision = 0
            if temp_count_tp+temp_count_fn != 0:
                recall = temp_count_tp/(temp_count_tp+temp_count_fn)
            else:
                recall = 0
            temp_precision.append(precision)
            temp_recall.append(recall)

            count_tp += temp_count_tp
            count_fp += temp_count_fp
            count_tn += temp_count_tn
            count_fn += temp_count_fn

            temp_count_tp, temp_count_fp, temp_count_tn, temp_count_fn = 0, 0, 0, 0

    print('\nglobal - true positives : ', count_tp, 'true negatives : ', count_tn, 'false positives : ', count_fp, 'false negatives : ', count_fn)
    print('per class - precision values to average: ', temp_precision)
    print('per class - recall values to average: ', temp_recall)

    try:
        accuracy = (count_tp+count_tn)/(count_tp+count_tn+count_fp+count_fn)
    except:
        accuracy = 0
    try:
        precision = count_tp/(count_tp+count_fp)
    except:
        precision = 0
    try:
        recall = count_tp/(count_tp+count_fn)
    except:
        recall = 0
    try:
        f1_score = 2*(precision*recall)/(precision+recall)
    except:
        f1_score = 0

    average_precision = sum(temp_precision)/len(temp_precision)
    average_recall = sum(temp_recall)/len(temp_recall)
    try:
        average_f1_score = 2*(average_precision*average_recall)/(average_precision+average_recall)
    except:
        average_f1_score = 0

    print('test accuracy : {0:0.5f}'.format(accuracy))
    print('micro test precision : {0:0.5f},'.format(precision), ' macro test precision : {0:0.5f}'.format(average_precision))
    print('micro test recall : {0:0.5f},'.format(recall), ' macro test recall : {0:0.5f}'.format(average_recall))
    print('micro test f1_score : {0:0.5f},'.format(f1_score), ' macro test f1_score : {0:0.5f}'.format(average_f1_score))
    return [-1,-1,-1,-1], [-1,-1,-1,-1], [accuracy, precision, recall, f1_score], [accuracy, average_precision, average_recall, average_f1_score]

#########################################################################################################################
# manual calculations :

def get_predictions_distribution(y_test, y_pred):
    tpfn, true_positives, false_positives = 0, 0, 0
    # this is the number of correct 1 over all the correct classifications (so, recall: TP over TP+TN) ######
    # true_positives and true_positives + false_negatives
    indexes_classifications = np.where(y_test.values == 1)
    if len(indexes_classifications[0]) > 0:
        for i in indexes_classifications[0]:
            if y_pred[i] == y_test.values[i]:
                true_positives += 1
            tpfn += 1

    # false_positives
    false_positives = tpfn - true_positives
    return true_positives, false_positives, tpfn

#########################################################################################################################
# DEEP LEARNING :
#########################################################################################################################
# CONVOLUTIONAL - LSTM:

# TODO: tune?
# binarizer: over a certain threshold it classify as 1 under as 0
get_binary_0_5 = lambda x: 1 if x > 0.05 else 0
get_binary_0_5 = np.vectorize(get_binary_0_5)

metrics_graph_ranges = {
    'sections': {'min':0, 'max': 0.5},
    'classes': {'min':0, 'max': 0.05},
    'subclasses': {'min':0, 'max': 0.05}
}

def get_binary_classification(predictions, threshold):
    binary_predictions = lambda prediction : 1 if prediction > threshold else 0
    return np.vectorize(binary_predictions)

def display_sequential_metrics(algorithm, metrics):
    print('###  calculating_metrics  ###')
    print('@@@  ' + algorithm + '  @@@')
    print("Over all labels - Coverage error: {:.3f}, Average labels: {:.3f}".format(
        metrics['coverage_error'], metrics['average_num_of_labels']))
    print("Percentage - Top 1: {:.3f}, Top 3: {:.3f}, Top 5: {:.3f}".format(
        metrics['top_1'], metrics['top_3'], metrics['top_5']))
    print("Macro - precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(
        metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro']))
    print("Micro - precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(
        metrics['precision_micro'], metrics['recall_micro'], metrics['f1_micro']))

# will be replaced
def display_convolutional_metrics(algorithm, loss, accuracy, mse, y_true, y_pred):
    print('###  calculating_metrics  ###')
    print('@@@  ', algorithm ,'  @@@')

    print('Test loss: ', loss)
    print('Test accuracy: ', accuracy)
    print('Test MSE: ', mse)

    print("\n")
    print("prediction 1: ", y_pred[6])
    print("argmax prediction 1: ", np.argmax(y_pred[6]))
    print("y_true 1: ", y_true[6])
    print("\n")

    unique, counts = np.unique(np.argmax(y_pred, axis=1), return_counts=True)
    print("argmax distribution in predictions: ", dict(zip(unique, counts)))

    unique, counts = np.unique(y_true, return_counts=True)
    print("distribution in TEST: ", dict(zip(unique, counts)))

    # print("y_true: ", y_true)

# never used
def display_convolution_metrics_fourth_attempt(y_train_true, y_true, y_train_pred, y_pred):
    print(y_train_true.shape)
    print(y_train_pred.shape)
    print(y_true.shape)
    print(y_pred.shape)

    print("nnDeep Neural Network - Train accuracy:")
    print(round(accuracy_score(y_train_true[0], y_train_pred[0]), 3))
    print("nDeep Neural Network - Test accuracy:")
    print(round(accuracy_score(y_test, y_pred), 3))
    print("nDeep Neural Network - Train Classification Report")
    print(classification_report(y_train_true, y_train_pred))
    print("nDeep Neural Network - Test Classification Report")
    print(classification_report(y_true, y_pred))

def get_top_N_percentage(y_score, y_true, max_N=3):
    """
    Get percentage of correct labels that are in the top N scores
    """
    num_all_true = 0
    num_found_in_max_N = 0
    for i in range(y_score.shape[0]):
        y_score_row = y_score[i, :]
        y_true_row = y_true[i, :]
        desc_score_indices = np.argsort(y_score_row)[::-1]
        true_indices = np.where(y_true_row ==1)[0]

        num_true_in_row = len(true_indices)
        num_all_true += num_true_in_row
        for i, score_index in enumerate(desc_score_indices):
            # only iterate through the score list till depth N, but make sure you also account for the case where
            # the number of true labels for the current row is higher than N
            if i >= max_N and i >= num_true_in_row:
                break
            if score_index in true_indices:
                num_found_in_max_N += 1
    return float(num_found_in_max_N) / num_all_true

def get_sequential_metrics(y_true, y_pred, y_binary_pred):
    """
    create the metrics object containing all relevant metrics
    """
    metrics = {}
    metrics['total_positive'] = np.sum(np.sum(y_binary_pred))

    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    metrics['y_binary_pred'] = y_binary_pred

    metrics['coverage_error'] = coverage_error(y_true, y_pred)
    metrics['average_num_of_labels'] = round(float(np.sum(np.sum(y_true, axis=1))) / y_true.shape[0], 2)

    metrics['average_precision_micro'] = average_precision_score(y_true, y_binary_pred, average='micro')
    metrics['average_precision_macro'] = average_precision_score(y_true, y_binary_pred, average='macro')

    metrics['precision_micro'] = precision_score(y_true, y_binary_pred, average='micro')
    metrics['precision_macro'] = precision_score(y_true, y_binary_pred, average='macro')
    metrics['recall_micro'] = recall_score(y_true, y_binary_pred, average='micro')
    metrics['recall_macro'] = recall_score(y_true, y_binary_pred, average='macro')
    metrics['f1_micro'] = f1_score(y_true, y_binary_pred, average='micro')
    metrics['f1_macro'] = f1_score(y_true, y_binary_pred, average='macro')

    # only calculate those for cases with a small number of labels (sections only)
    if y_true.shape[1] < 100:
        precision_scores = np.zeros(y_true.shape[1])
        for i in range(0, y_true.shape[1]):
            precision_scores[i] = precision_score(y_true[:,i], y_binary_pred[:,i])
        metrics['precision_scores_array'] = precision_scores.tolist()

        recall_scores = np.zeros(y_true.shape[1])
        for i in range(0, y_true.shape[1]):
            recall_scores[i] = recall_score(y_true[:,i], y_binary_pred[:,i])
        metrics['recall_scores_array'] = recall_scores.tolist()

        f1_scores = np.zeros(y_true.shape[1])
        for i in range(0, y_true.shape[1]):
            f1_scores[i] = f1_score(y_true[:,i], y_binary_pred[:,i])
        metrics['f1_scores_array'] = f1_scores.tolist()

    metrics['top_1'] = get_top_N_percentage(y_pred, y_true, max_N=1)
    metrics['top_3'] = get_top_N_percentage(y_pred, y_true, max_N=3)
    metrics['top_5'] = get_top_N_percentage(y_pred, y_true, max_N=5)
    return metrics

class MetricsCNNCallback():
# class MetricsCNNCallback(Callback):
    """
    Callback called by keras after each epoch. Records the best validation loss and periodically checks the
    validation metrics
    """
    def __init__(self, val_data, val_labels, patience):
        # super(EarlyStoppingAtMinLoss, self).__init__()
        self.val_data = val_data
        self.val_labels = val_labels

        self.patience = patience

        self.best_val_loss = None
        self.best_weights = None

    def on_train_begin(self, logs={}):
        self.epoch_index = 0

        self.wait = 0
        self.stopped_epoch = 0

        self.metrics_dict = {}
        self.best_val_loss = np.Inf
        self.best_weights = np.Inf
        self.best_validation_metrics = None

        self.losses = []
        self.val_losses = []

        self.val_predictions = None
        self.binary_predictions = None
        self.val_loss, self.val_acc, self.val_mse = 0, 0, 0
        self.validation_metrics = {}

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_index += 1

        actual_val_loss = logs.get('val_loss')
        loss = logs.get('loss')
        self.losses.append(loss)
        self.val_losses.append(actual_val_loss)

        if np.less(actual_val_loss, self.best_val_loss):
            self.best_val_loss = actual_val_loss
            self.best_weights = self.model.get_weights()
            print('Found lower val loss for epoch {} => {}'.format(self.epoch_index, round(actual_val_loss, 5)))
        else:
            self.wait += 1
            if np.less(self.patience, self.wait):
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

def get_data_files(base_location, classif_type, level, data_type):
    """
    get the files to load the data from for a certain classification type, level and data type
    """
    data_file = os.path.join(base_location, data_type_file_dict[data_type].format(level))
    labels_file = os.path.join(base_location, labels_type_file_dict[data_type].format(classif_type))
    return data_file, labels_file

def get_data(data_file, labels_file, mmap=False):
    """
    load np data with a certain mmap configuration
    """
    mmap_mode = None
    if mmap == True:
        mmap_mode = "r"
    X_data = np.load(data_file, mmap_mode=mmap_mode)
    y_data = np.load(labels_file, mmap_mode=mmap_mode)
    return X_data, y_data

class MetricsCallback():
# class MetricsCallback(Callback):
    """
    Callback called by keras after each epoch. Records the best validation loss and periodically checks the
    validation metrics
    """
    def __init__(self, base_load_directory, classifications_type, level, batch_size, is_mlp=False):
        MetricsCallback.EPOCHS_BEFORE_VALIDATION = 10
        MetricsCallback.GRAPH_MIN = metrics_graph_ranges[classifications_type]['min']
        MetricsCallback.GRAPH_MAX = metrics_graph_ranges[classifications_type]['max']
        self.base_load_directory = base_load_directory
        self.classifications_type = classifications_type
        self.level = level
        self.batch_size = batch_size
        self.is_mlp = is_mlp

    def on_train_begin(self, logs={}):
        self.epoch_index = 0
        self.val_loss_reductions = 0
        self.metrics_dict = {}
        self.best_val_loss = np.iinfo(np.int32).max
        self.best_weights = None
        self.best_validation_metrics = None

        self.losses = []
        self.val_losses = []
        self.fig = plt.figure(figsize=(12,6), dpi=80)
        self.ax = plt.subplot(111)

    def on_epoch_end(self, epoch, logs={}):
        QUEUE_SIZE = 100
        self.epoch_index += 1
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        loss_line, = self.ax.plot(range(1,self.epoch_index+1), self.losses, 'g-', label='Training Loss')
        val_loss_line, = self.ax.plot(range(1,self.epoch_index+1), self.val_losses, 'r-', label='Validation Loss')
        self.ax.legend(handles=[loss_line, val_loss_line])
        self.ax.set_ylim((MetricsCallback.GRAPH_MIN, MetricsCallback.GRAPH_MAX))
        self.fig.canvas.draw()
        if logs['val_loss'] < self.best_val_loss:
            self.val_loss_reductions += 1
            self.best_val_loss = logs['val_loss']
            self.best_weights = self.model.get_weights()
            print('\r    \r') # to remove the previous line of verbose output of model fit
            # time.sleep(0.1)
            print('Found lower val loss for epoch {} => {}'.format(self.epoch_index, round(logs['val_loss'], 5)))
            if self.val_loss_reductions % MetricsCallback.EPOCHS_BEFORE_VALIDATION == 0:

                print('Validation Loss Reduced {} times'.format(self.val_loss_reductions))
                print('Evaluating on Validation Data')
                Xv_file, yv_file = get_data_files(self.base_load_directory, self.classifications_type, self.level, 'validation') # creates the file paths
                Xv, yv = get_data(Xv_file, yv_file, mmap=True) # load the files as ndarray
                yvp = self.model.predict_generator(generator=batch_generator(Xv_file, yv_file,
                                                   self.batch_size, is_mlp=self.is_mlp, validate=True),
                                                   max_q_size=QUEUE_SIZE,
                                                   val_samples=yv.shape[1])
                yvp_binary = get_binary_0_5(yvp)
                print('Generating Validation Metrics')
                validation_metrics = get_sequential_metrics(yv, yvp, yvp_binary)
                print("****** Validation Metrics: Cov Err: {:.3f} | Top 3: {:.3f} | Top 5: {:.3f} | F1 Micro: {:.3f} | F1 Macro: {:.3f}".format(
                    validation_metrics['coverage_error'], validation_metrics['top_3'], validation_metrics['top_5'],
                    validation_metrics['f1_micro'], validation_metrics['f1_macro']))
                self.metrics_dict[self.epoch_index] = validation_metrics
