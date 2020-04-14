# -*- coding: utf-8 -*-
import sys
import os
import stat
import glob
import numpy as np
import pandas as pd
import multiprocessing
import random
import matplotlib as plt
from tqdm import tqdm
import csv
csv.field_size_limit(sys.maxsize)
import collections
import time

# we use OneVsRestClassifier for multi-label prediction
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

sys.path.append(sys.path[0][:sys.path[0].rfind('/')])
from main_scripts import directory
directory.append_working_directory() # append the script path for the server

from helpers import tool_helper as th
from helpers import txt_data_helper as txth
from helpers import classification_helper as ch
from helpers import metrics_helper as mh
from helpers import word_model_helper as wmh
from helpers import predicting_model_helper as pmh

script_key = "classify"

def get_suitable_path():
    current_directory = os.getcwd()
    dirlist = os.listdir(current_directory)
    if 'chakrm' in dirlist:
        ssh_source_dir = '/chakrm/workspace/2019-MastersProject/'
        sys.path.append(os.path.abspath('..')+ssh_source_dir)
    else:
        sys.path.append(os.path.abspath('..'))

def overview_models(data_frame, classif_level, classif_type, source_path):
    vect_data, patent_ids, vectorizer = ch.apply_tfidf_vectorizer_fit_transform(data_frame)

    benchmark = []

    # word2vec model, svm, decision tree, random forest, hidden markov model, k-nearest
    for algorithm in [pmh.get_SVC(),
                      pmh.get_decision_tree(),
                      pmh.get_kneighbors(),
                      pmh.get_logistic()
                      ]:
        classifier_name, parameters = ch.get_classifier_information(str(algorithm))
        print('###  ', classifier_name ,'  ###')
        cross_results = mh.get_cross_val_score(algorithm, vect_data, data_frame['classification'], 'accuracy')
        tmp = pd.DataFrame.from_dict(cross_results).mean(axis=0)
        tmp = tmp.append(pd.Series([classifier_name], index=['algorithm']))
        benchmark.append(tmp)
    ch.save_results_cross_validation('cross_validation', benchmark, ['cv=5'], classif_level, classif_type, source_path)

def overview_multilabel_models(data_frame):
    vect_data, patent_ids, vectorizer = ch.apply_tfidf_vectorizer_fit_transform(data_frame)

    ################################################# text: from text to sparse binary matrix [[0, 1, 0],[1, 0, 1]]

    # temp_text = ch.apply_count_vectorizer(data_frame)

    ################################################# classification: from text to sparse binary matrix [[0, 1, 0],[1, 0, 1]]

    temp_classification, classes, n_classes = ch.apply_multilabel_binarizer(data_frame)

    benchmark_dataframe = pd.DataFrame(columns=['algorithm', 'accuracy', 'recall', 'precision', 'class'])

    for algorithm in [pmh.get_SVC(),
                      pmh.get_decision_tree(),
                      pmh.get_kneighbors(),
                      pmh.get_logistic(),
                      pmh.get_random_forest_classifier(),
                      pmh.get_linear_SVC(),
                      pmh.get_multinomialNB()
                      ]:
        classifier_name, parameters = ch.get_classifier_information(str(algorithm))
        print('###  ', classifier_name ,'  ###')
        for i in range(n_classes):
            unique, counts = np.unique(temp_classification[:, i], return_counts=True)
            if len(counts) > 1 and counts[1] > 1:
                mean_accuracy, mean_precision, mean_recall = mh.calculate_metrics_for_crossvalidation(algorithm, vect_data, temp_classification[:, i])

                mean_accuracy = np.append(mean_accuracy, pd.Series([classifier_name], index=['Algorithm']))
                mean_precision = np.append(mean_precision, pd.Series([classifier_name], index=['Algorithm']))
                mean_recall = np.append(mean_recall, pd.Series([classifier_name], index=['Algorithm']))

                benchmark_dataframe.loc[benchmark_dataframe.shape[0] + 1] = [classifier_name, mean_accuracy, mean_recall, mean_precision, classes[i]]

    print("algorithm ", " accuracy ", " recall ", " precision")
    for index, row in benchmark_dataframe.iterrows():
        print(row[0], " ",  row[1].flatten()[0], " ", row[2].flatten()[0], " ", row[3].flatten()[0], " ",row[4])

    path_to_csv = ch.get_csv_path('cross_validation_multiclass')
    ch.write_dataframe_as_csv(benchmark_dataframe, path_to_csv)

def improved_logistic_regression(train_tagged, test_tagged, model_dbow, logreg, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path):
    classifier_name_0, parameters_0 = ch.get_classifier_information(str(model_dbow))
    classifier_name_1, parameters_1 = ch.get_classifier_information(str(logreg))

    model_dmm = wmh.train_doc2vec_with_tagged_data(train_tagged.values)

    classifier_name_2, parameters_2 = ch.get_classifier_information(str(model_dmm))

    y_train, X_train = wmh.vec_for_learning(model_dmm, train_tagged)
    y_test, X_test = wmh.vec_for_learning(model_dmm, test_tagged)

    y_pred = pmh.fit_predict_functions(logreg, X_train, y_train, X_test)

    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    merged_model = wmh.get_concatenated_doc2vec(model_dbow, model_dmm)

    y_train, X_train = wmh.vec_for_learning(merged_model, train_tagged)
    y_test, X_test = wmh.vec_for_learning(merged_model, test_tagged)

    y_pred = pmh.fit_predict_functions(logreg, X_train, y_train, X_test)

    model_name = '[all classes predictions]'+classifier_name_0+'/'+classifier_name_2+'/'+classifier_name_1
    list_metrics = mh.calculate_metrics(model_name, y_test, y_pred)
    none_average, binary_average, micro_average, macro_average = list_metrics

    ch.save_results(classifier_name_1, list_metrics, parameters_1, model_name, classif_level, classif_type, source_path)

def apply_doc2vec_logistic_regression(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path):
    baseline_name = '[all classes predictions]'+text_vectorizer+'/'+class_vectorizer
    results_model_dbow = ch.apply_doc2vec_separated_train_test(data_frame, baseline_name)
    X_train, y_train, X_test, y_test, model_dbow, train_tagged, test_tagged = results_model_dbow

    logreg = pmh.get_logistic()

    classifier_name_0, parameters_0 = ch.get_classifier_information(str(model_dbow))
    classifier_name_1, parameters_1 = ch.get_classifier_information(str(logreg))

    y_pred = pmh.fit_predict_functions(logreg, X_train, y_train, X_test)

    model_name = baseline_name+classifier_name_0+'/'+classifier_name_1
    list_metrics = mh.calculate_metrics(model_name, y_test, y_pred)
    none_average, binary_average, micro_average, macro_average = list_metrics

    ch.save_results(classifier_name_1, list_metrics, parameters_1, model_name, classif_level, classif_type, source_path)

    # baseline_name = '[each class predictions]'+text_vectorizer+'/'+class_vectorizer
    # vectorizer_results = ch.apply_df_vectorizer(data_frame, 'doc2vec', 'multi_label', baseline_name)
    # X_train, X_test, y_train, y_test, classes, n_classes, vocab_processor, len_vocabulary = vectorizer_results
    # model_name = baseline_name+'/'+classifier_name_0+'/'+classifier_name_1
    #
    # for i in range(n_classes):
    #     unique, counts = np.unique(y_train[:, i], return_counts=True)
    #     if len(counts) > 1 and counts[1] > 1:
    #         y_pred = pmh.fit_predict_functions(custom_pipeline, X_train, y_train[:, i], X_test)
    #         print('###  ', classes[i] ,'  ###')
    #         list_metrics = mh.calculate_metrics(model_name, y_test[:, i], y_pred)
    #         none_average, binary_average, micro_average, macro_average = list_metrics
    #
    #         ch.save_results(classifier_name_1, list_metrics, parameters_1, model_name, classif_level, classif_type, source_path)

    # improvement
    improved_logistic_regression(train_tagged, test_tagged, model_dbow, logreg, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path)

def apply_word2vec_extratrees(data_frame, classif_level, classif_type, source_path):
    data_frame['text'] = data_frame.apply(lambda row : th.tokenize_complex_text(row['text']), axis=1)
    data_frame['classification'] = data_frame.apply(lambda row : th.tokenize_complex_text(row['classification']), axis=1)

    df_single_classification = ch.get_list_each_text_a_different_classification(data_frame)

    x = df_single_classification['text']
    y = df_single_classification['classification']

    X_train, X_test, y_train, y_test = ch.get_train_test_from_data(x, y)

    model_w2v = wmh.get_word2vec_model(X_train)

    etree_w2v = Pipeline([
        ("word2vec vectorizer", wmh.MeanEmbeddingVectorizer(model_w2v)),
        ("extra trees", pmh.get_extra_tree())])
    etree_w2v_tfidf = Pipeline([
        ("word2vec vectorizer", wmh.TfidfEmbeddingVectorizer(model_w2v)),
        ("extra trees", pmh.get_extra_tree())])

    # NB!!!: the model does not support multi targets, so i duplicate the sources and give them different targets
    y_pred = pmh.fit_predict_functions(etree_w2v_tfidf, X_train, y_train, X_test)

    classifier_name_0 = 'Word2Vec/MeanEmbeddingVectorizer'
    classifier_name_1, parameters_1 = ch.get_extratree_classifier_information(str(etree_w2v))
    model_name = '[all classes predictions]'+classifier_name_0+'/'+classifier_name_1

    # this should be changed by comparing all the possibilities for specified text (i can use the original dataframe!)
    list_metrics = mh.calculate_metrics(model_name, y_test, y_pred)
    none_average, binary_average, micro_average, macro_average = list_metrics

    ch.save_results(classifier_name_1, list_metrics, parameters_1, model_name, classif_level, classif_type, source_path)

    # NB!!!: the model does not support multi targets, so i duplicate the sources and give them different targets
    y_pred = pmh.fit_predict_functions(etree_w2v, X_train, y_train, X_test)

    classifier_name_2 = 'Word2Vec/TfidfEmbeddingVectorizer'
    model_name = '[all classes predictions]'+classifier_name_2+'/'+classifier_name_1

    # this should be changed by comparing all the possibilities for specified text (i can use the original dataframe!)
    list_metrics = mh.calculate_metrics(model_name, y_test, y_pred)
    none_average, binary_average, micro_average, macro_average = list_metrics

    ch.save_results(classifier_name_1, list_metrics, parameters_1, model_name, classif_level, classif_type, source_path)

def apply_naive_bayes(X_train_tfidf, y_train, X_test_tfidf, y_test, classif_level, classif_type, source_path):
    naive = pmh.get_multinomialNB()

    classifier_name, parameters = ch.get_classifier_information(str(naive))

    y_pred = pmh.fit_predict_functions(naive, X_train_tfidf, y_train, X_test_tfidf)

    model_name = '[all classes predictions]label_encoder/tfidf/'+classifier_name
    list_metrics = mh.calculate_metrics(model_name, y_test, y_pred)
    none_average, binary_average, micro_average, macro_average = list_metrics

    ch.save_results(classifier_name, list_metrics, parameters, model_name, classif_level, classif_type, source_path)

def apply_svm(X_train_tfidf, y_train, X_test_tfidf, y_test, classif_level, classif_type, source_path):
    svm = pmh.get_SVC()

    classifier_name, parameters = ch.get_classifier_information(str(svm))

    y_pred = pmh.fit_predict_functions(svm, X_train_tfidf, y_train, X_test_tfidf)

    model_name = '[all classes predictions]label_encoder/tfidf/'+classifier_name
    list_metrics = mh.calculate_metrics(model_name, y_test, y_pred)
    none_average, binary_average, micro_average, macro_average = list_metrics

    ch.save_results(classifier_name, list_metrics, parameters, model_name, classif_level, classif_type, source_path)

def apply_multilabel_label_encoder_tfidf_classification(data_frame, classif_level, classif_type, source_path):
    data_frame = ch.get_list_each_text_a_different_classification(data_frame)

    temp_text, patent_ids, vectorizer = ch.apply_tfidf_vectorizer_fit_transform(data_frame)

    X_train_tfidf, X_test_tfidf, y_train, y_test = ch.get_train_test_from_data(temp_text, data_frame['classification'])

    y_train = ch.apply_label_encoder(y_train)
    y_test = ch.apply_label_encoder(y_test)

    apply_naive_bayes(X_train_tfidf, y_train, X_test_tfidf, y_test, classif_level, classif_type, source_path)
    apply_svm(X_train_tfidf, y_train, X_test_tfidf, y_test, classif_level, classif_type, source_path)

def apply_onevsrest(X_train, y_train, X_test, y_test, classes, baseline_name, source_path):
    # X_train.sort_indices() # SVC needs this line in addition
    custom_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(pmh.get_logistic(), n_jobs=-1)),
                    # ('clf', OneVsRestClassifier(pmh.get_SVC(), n_jobs=-1)),
                    # ('clf', OneVsRestClassifier(pmh.get_multinomialNB(), n_jobs=-1)),
                    # ('clf', OneVsRestClassifier(pmh.get_decision_tree(), n_jobs=-1)),
                    # ('clf', OneVsRestClassifier(pmh.get_kneighbors(), n_jobs=-1)),
                    # ('clf', OneVsRestClassifier(pmh.get_linear_SVC(), n_jobs=-1)),
                    # ('clf', OneVsRestClassifier(pmh.get_random_forest_classifier(), n_jobs=-1)),
                    # ('clf', OneVsRestClassifier(pmh.get_SGD_classifier(), n_jobs=-1)),
                ])

    classifier_name, parameters = ch.get_complex_classifier_information(str(custom_pipeline), 3, 1, 4, 0)
    model_name = '[each class predictions]'+baseline_name+classifier_name

    accuracies = []
    for _class in classes:
        print('**Processing {} texts...**'.format(_class))

        y_pred = pmh.fit_predict_functions(custom_pipeline, X_train, y_train[_class], X_test)

        accuracies.append(mh.get_accuracy_score(y_test[_class], y_pred))

        list_metrics = mh.calculate_metrics(model_name, y_test[_class], y_pred)
        none_average, binary_average, micro_average, macro_average = list_metrics

        ch.save_results(classifier_name, list_metrics, parameters, model_name, classif_level, classif_type, source_path)

        print('Pipeline score on training {}'.format(custom_pipeline.score(X_train, y_train[_class])))

        true_positives, false_positives, tpfn = mh.get_predictions_distribution(y_test[_class], y_pred)

    model_name = '[all classes predictions]'+baseline_name+classifier_name
    if tpfn == 0 or (true_positives + false_positives) == 0:
        mh.display_directly_metrics(model_name, 0, 0, 0, -1)
    else:
        precision = true_positives/(true_positives+false_positives)
        recall = true_positives/tpfn
        mh.display_directly_metrics(model_name, precision, recall, 2*(precision*recall)/(precision+recall), -1)

def apply_binary_relevance(X_train, y_train, X_test, y_test, baseline_name, source_path):
    classifier = ch.get_binary_relevance(pmh.get_gaussianNB())

    classifier_name, parameters = ch.get_complex_classifier_information(str(classifier), 1, 1, 2, 0)

    y_pred = pmh.fit_predict_functions(classifier, X_train, y_train, X_test)

    model_name = '[all classes predictions]'+baseline_name+classifier_name
    list_metrics = mh.calculate_metrics(model_name, y_test, y_pred)
    none_average, binary_average, micro_average, macro_average = list_metrics

    ch.save_results(classifier_name, list_metrics, parameters, model_name, classif_level, classif_type, source_path)

def apply_classifier_chain(X_train, y_train, X_test, y_test, baseline_name, source_path):
    classifier = ch.get_classifier_chain(pmh.get_logistic())

    classifier_name, parameters = ch.get_complex_classifier_information(str(classifier), 1, 1, 2, 0)

    y_pred = pmh.fit_predict_functions(classifier, X_train, y_train, X_test)

    model_name = '[all classes predictions]'+baseline_name+classifier_name
    list_metrics = mh.calculate_metrics(model_name, y_test, y_pred)
    none_average, binary_average, micro_average, macro_average = list_metrics

    ch.save_results(classifier_name, list_metrics, parameters, model_name, classif_level, classif_type, source_path)

def apply_label_powerset(X_train, y_train, X_test, y_test, baseline_name, source_path):
    classifier = ch.get_label_powerset(pmh.get_logistic())

    classifier_name, parameters = ch.get_complex_classifier_information(str(classifier), 1, 1, 2, 0)

    y_pred = pmh.fit_predict_functions(classifier, X_train, y_train, X_test)

    model_name = '[all classes predictions]'+baseline_name+classifier_name
    list_metrics = mh.calculate_metrics(model_name, y_test, y_pred)
    none_average, binary_average, micro_average, macro_average = list_metrics

    ch.save_results(classifier_name, list_metrics, parameters, model_name, classif_level, classif_type, source_path)

def apply_adapted_algorithm(X_train, y_train, X_test, y_test, baseline_name, source_path):
    classifier = pmh.get_MLkNN()

    classifier_name, parameters = ch.get_complex_classifier_information(str(classifier), 0, 0, 1, 0)

    X_train, y_train, X_test = th.get_lil_matrices(X_train, y_train, X_test)

    y_pred = pmh.fit_predict_functions(classifier, X_train, y_train, X_test)

    model_name = '[all classes predictions]'+baseline_name+classifier_name
    list_metrics = mh.calculate_metrics(model_name, y_test, y_pred)
    none_average, binary_average, micro_average, macro_average = list_metrics

    ch.save_results(classifier_name, list_metrics, parameters, model_name, classif_level, classif_type, source_path)

def apply_multi_label_classification(data_frame, text_vectorizer, class_vectorizer, classif_type, classif_level, source_path):
    ################################################# classification: from text to sparse binary matrix [[0, 1, 0],[1, 0, 1]]
    baseline_name = text_vectorizer+'/'+class_vectorizer+'/onevsrest/'
    vectorizer_results = ch.apply_df_vectorizer(data_frame, text_vectorizer, class_vectorizer, '[both]'+baseline_name)
    X_train, X_test, y_train, y_test, classes, n_classes, vocab_processor, len_vocabulary = vectorizer_results
    y_train = pd.DataFrame(y_train, columns = classes)
    y_test = pd.DataFrame(y_test, columns = classes)

    ###################################################### text: from text to sparse binary matrix [[0, 1, 0],[1, 0, 1]]

    # temp_text = ch.apply_count_vectorizer(data_frame)

    ###################################################### text: from text to sparse binary matrix [[0, 1, 0],[1, 0, 1]]

    apply_onevsrest(X_train, y_train, X_test, y_test, classes, baseline_name, source_path)
    # apply_binary_relevance(X_train, y_train, X_test, y_test, baseline_name, source_path) # needs lots of patents!
    # apply_classifier_chain(X_train, y_train, X_test, y_test, baseline_name, source_path) # needs lots of patents!
    # apply_label_powerset(X_train, y_train, X_test, y_test, baseline_name, source_path) # best one
    apply_adapted_algorithm(X_train, y_train, X_test, y_test, baseline_name, source_path)

def apply_multi_label_classification_without_pipeline(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path):
    baseline_name = text_vectorizer+'/'+class_vectorizer+'/onevsrest/'
    vectorizer_results = ch.apply_df_vectorizer(data_frame, text_vectorizer, class_vectorizer, '[both]'+baseline_name)
    X_train, X_test, y_train, y_test, classes, n_classes, vocab_processor, len_vocabulary = vectorizer_results
    # len_vocabulary = 57335
    # len_vocabulary = 34736

    print(X_train[0])

    print('len_vocabulary: ', len_vocabulary, ' num_classes: ', n_classes)

    # Run classifier
    classifier = OneVsRestClassifier(pmh.get_logistic()) # here
    # classifier = OneVsRestClassifier(pmh.get_SVC()) # here
    # classifier = OneVsRestClassifier(pmh.get_multinomialNB()) # ERROR WORD2VEC/DOC2VEC (X has negative value)
    # classifier = OneVsRestClassifier(pmh.get_decision_tree()) # here
    # classifier = OneVsRestClassifier(pmh.get_kneighbors()) # here
    # classifier = OneVsRestClassifier(pmh.get_linear_SVC())
    # classifier = OneVsRestClassifier(pmh.get_random_forest_classifier()) # here
    # classifier = OneVsRestClassifier(pmh.get_SGD_classifier())

    train_predictions = np.ndarray(shape=(n_classes, y_train.shape[0]), dtype=int)
    predictions = np.ndarray(shape=(n_classes, y_test.shape[0]), dtype=int)
    ###
    precision = dict()
    recall = dict()
    average_precision = dict()

    classifier_name, parameters = ch.get_complex_classifier_information(str(classifier), 1, 1, 2, 0)

    second_training = False
    just_once = True
    another_try = False # single train and estimation instead of multi-train-estimation steps (it performs better with svm and logistic)

    if not another_try:
        for _ in range(1):
            for i in range(n_classes):
                if second_training:
                    if classifier_name in ['DecisionTreeClassifier', 'KNeighborsClassifier', 'MultinomialNB', 'RandomForestClassifier']:
                        # do not provide the second metrics
                        break
                    elif just_once:
                        # fit again with the whole set of classes - should be better
                        classifier.fit(X_train, y_train)
                        y_score = classifier.decision_function(X_test)

                        precision[i], recall[i], average_precision[i] = mh.calculate_recall_curve_precision_score(y_test[:, i], y_score[:, i], None, y_test[:, i], y_score[:, i])

                        just_once = False
                else:
                    predictions[i] = pmh.fit_predict_functions(classifier, X_train, y_train[:, i], X_test)
                    train_predictions[i] = classifier.predict(X_train)
                print('**Processing classes {0:0.2f} % ...**'.format(((i+1)/n_classes)*100))
            second_training = True
            break

        predictions = predictions.transpose()
        print('transposed')
    else:
        predictions = pmh.fit_predict_functions(classifier, X_train, y_train, X_test)
        train_predictions = classifier.predict(X_train)

    # metrics
    model_name = '[each class predictions]'+baseline_name+classifier_name
    manual_metrics = mh.calculate_manual_metrics(model_name, y_test, predictions)
    none_average, binary_average, micro_average, macro_average = manual_metrics

    # metrics
    list_metrics = mh.calculate_metrics(model_name, y_test, predictions)
    none_average, binary_average, micro_average, macro_average = list_metrics

    ch.save_results(classifier_name, list_metrics, parameters, model_name, classif_level, classif_type, source_path)

    if not just_once:
        print('not just once')
        train_predictions = classifier.predict(X_train)
        predictions = classifier.predict(X_test)

        # metrics
        mh.calculate_metrics_with_recall_curve(y_score, y_train, y_test, train_predictions, predictions)

        # metrics
        model_name = '[all classes predictions]'+baseline_name+classifier_name
        list_metrics = mh.calculate_metrics(model_name, y_test, predictions)
        none_average, binary_average, micro_average, macro_average = list_metrics

        ch.save_results(classifier_name, list_metrics, parameters, model_name, classif_level, classif_type, source_path)

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
        # source_path = ['/home/chakrm/storage/cleaned/EP-patents-extract/2000/', '/home/chakrm/storage/cleaned/EP-patents-extract/2001/', '/home/chakrm/storage/cleaned/EP-patents-extract/2002/', '/home/chakrm/storage/cleaned/EP-patents-extract/2003/', '/home/chakrm/storage/cleaned/EP-patents-extract/2004/']
        # source_path = ['/home/chakrm/storage/cleaned/EP-patents-extract/2005/']
        # source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_classification/cleaned/B/']

    start = time.time()

    # text: tfidf, doc2vec, word2vec, load_data/tfidf, load_data/doc2vec
    # class: multi_label
    text_vectorizer = 'load_data/doc2vec'
    class_vectorizer = 'multi_label'
    classif_level = 'description_claim_abstract_title' # description, claim, abstract, title
    classif_type = 'subclasses' # sectors, classes, subclasses

    # if there is the '/' into the text_vectorizer, you are supposed to load data from csv
    index = text_vectorizer.find('/')
    if index == -1:
        print(source_path)
        # first option: here you read the directory with the patents
        patent_ids, temp_df, classifications_df = txth.load_data(source_path)
        data_frame, classif_level, classif_type = txth.get_final_df(patent_ids, temp_df, classif_type)

        # these lines are useful for storing the data_frames and loading them in a separate step
        #
        # # data_frame = ch.further_preprocessing_phase(data_frame)

        # # data_frame, _ = ch.reduce_amount_of_classes(data_frame, classifications_df)

        # csvfile = 'training_2000_2004_cleaned_abstract_2.csv'
        # ch.save_data_frame(script_key, data_frame, csvfile)

        # source_path = ['/home/chakrm/storage/cleaned/EP-patents-extract/2005/']
        # patent_ids, temp_df, classifications_df = txth.load_data(source_path)
        # test, classif_level, classif_type = txth.get_final_df(patent_ids, temp_df, classif_type)

        # # test = ch.further_preprocessing_phase(test)

        # # test, _ = ch.reduce_amount_of_classes(test, classifications_df)

        # csvfile = 'testing_2005_cleaned_abstract_2.csv'
        # ch.save_data_frame(script_key, test, csvfile)
        #
        # here you can stop the iteration
    else:
        try:
            # second option: here you specify the csv
            csvfile = 'training_2000_2004_cleaned_2.csv'
            # csvfile = 'training_2000_2004_cleaned_abstract_2.csv'
            train, train_classifications_df = ch.load_data_frame(script_key, csvfile)

            csvfile = 'testing_2005_cleaned_2.csv'
            # csvfile = 'testing_2005_cleaned_abstract_2.csv'
            test, classifications_df = ch.load_data_frame(script_key, csvfile)

            # train, classes_to_remove = ch.reduce_amount_of_classes(train, train_classifications_df)
            # test, _ = ch.reduce_amount_of_classes(test, classes_to_remove)

            print('loaded training set: ', train.shape)
            print('loaded testing set: ', test.shape)

            data_frame = [train, test]
        except:
            # third option: here you specify one csv
            csvfile = 'data_frame_cleaned_2.csv'
            data_frame, classifications_df = ch.load_data_frame(script_key, csvfile)

            train, test = ch.get_train_test_from_dataframe(data_frame)

            data_frame = [train, test]

    if False:
        print('before: \n', data_frame['text'])

        if index != -1:
            data_frame = pd.concat([train, test])
        data_frame = ch.further_preprocessing_phase(data_frame)

        print('after: \n', data_frame['text'])

        csvfile = 'training_2000_2004_cleaned_abstract.csv'
        # csvfile = 'data_frame_cleaned_2.csv'
        ch.save_data_frame(script_key, data_frame, csvfile)

    # data_frame['classification'] = data_frame['classification'].apply(lambda classcode : ch.shrink_to_sectors(th.tokenize_text(classcode)))
    # classification_df = pd.DataFrame(columns=['class', 'count'])
    # data_frame['classification'].apply(lambda classcode : th.calculate_class_distribution(classcode, classification_df))
    # print('sectors distribution : \n', classification_df)

    # overview_models(data_frame, classif_level, classif_type, source_path) # # cross_validation_one_class
    # overview_multilabel_models(data_frame) # cross_validation_all_classes - tfidf - multi_label

    # apply_doc2vec_logistic_regression(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path)
    # doc2vec - one prediction a patent - improvement/double_doc2vec - possibility to multi label classification # # not binary

    # wmh.weird_application_word2vec(data_frame) # # does not work
    # apply_word2vec_extratrees(data_frame, classif_level, classif_type, source_path) # uses single_clssification approach but it has cons... model learns one classification for text and another for the same text # # not binary

    # apply_multilabel_label_encoder_tfidf_classification(data_frame, classif_level, classif_type, source_path) # tfidf/naive bayes, tfidf/svm # # duplicate the text with different classes
    # apply_multi_label_classification(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path) # onevsrest - logistic, svm, naive, decisiontree; binary_relevance - naive bayes; classifier_chain - logistic; label_powerset - logistic; adapted_algorithm - mlknn
    apply_multi_label_classification_without_pipeline(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path)
    # all text vectorizer and label vectorizer, onevsrest, all models - reference!

    print("end classification step, time: ", time.time()-start)

# A1 structure: kind, classcode, applicant, abstract
# B1 structure: kind, respective A1, classcode, claim, description

# NB:
# logistic has slighly better accuracy but significantly worse recall!!!
# there are some patents that have the classification as three capitals not four, for instance: F16, C23...
#
# ONE CLASS - LOGISTIC
# the real recall of the method with sectors (logistic) is 0.12 - 1600 patents
# the real recall of the method with sectors and half class (1 digit) (logistic) is 0.068 - 1600 patents
# the real recall of the method with sectors and class (2 digits) (logistic) is 0.017 - 1600 patents
# the real recall of the mothod with subclass (4 characters) (logistic) is 0.003 - 1600 patents
#
# MULTICLASS - LOGISTIC
# the real recall of the method with sectors (logistic) is 0.17 - 1600 patents
# the real recall of the method with sectors and half class (1 digit) (logistic) is 0.10 - 1600 patents
# the real recall of the method with sectors and class (2 digits) (logistic) is 0.021 - 1600 patents
# the real recall of the mothod with subclass (4 characters) (logistic) is 0.0038 - 1600 patents
# MULTICLASS - SVM
# the real recall of the method with sectors (logistic) is 0.44 - 1600 patents
# the real recall of the method with sectors and half class (1 digit) (logistic) is 0.33 - 1600 patents
# the real recall of the method with sectors and class (2 digits) (logistic) is 0.17 - 1600 patents
# the real recall of the mothod with subclass (4 characters) (logistic) is 0.054 - 1600 patents
# MULTICLASS - DECISION TREE
# the real recall of the method with sectors (logistic) is 0.52 - 1600 patents
# the real recall of the method with sectors and half class (1 digit) (logistic) is 0.46 - 1600 patents
# the real recall of the method with sectors and class (2 digits) (logistic) is 0.34 - 1600 patents
# the real recall of the mothod with subclass (4 characters) (logistic) is 0.26 - 1600 patents
# MULTICLASS - KNEIGHBOR CLASSIFIER
# the real recall of the method with sectors (logistic) is 0.57 - 1600 patents
# the real recall of the method with sectors and half class (1 digit) (logistic) is 0.46 - 1600 patents
# the real recall of the method with sectors and class (2 digits) (logistic) is 0.36 - 1600 patents
# the real recall of the mothod with subclass (4 characters) (logistic) is 0.20 - 1600 patents
# MULTICLASS - LINEAR SVM
# the real recall of the method with sectors (logistic) is 0.44 - 1600 patents
# the real recall of the method with sectors and half class (1 digit) (logistic) is 0.33 - 1600 patents
# the real recall of the method with sectors and class (2 digits) (logistic) is 0.18 - 1600 patents
# the real recall of the mothod with subclass (4 characters) (logistic) is 0.07 - 1600 patents
# MULTICLASS - RANDOM FOREST
# the real recall of the method with sectors (logistic) is 0.27 - 1600 patents
# the real recall of the method with sectors and half class (1 digit) (logistic) is 0.15 - 1600 patents
# the real recall of the method with sectors and class (2 digits) (logistic) is 0.08 - 1600 patents
# the real recall of the mothod with subclass (4 characters) (logistic) is 0.025 - 1600 patents
# MULTICLASS - SGDC CLASSIFIER (mix between svm, logistic regression)
# the real recall of the method with sectors (logistic) is 0.39 - 1600 patents
# the real recall of the method with sectors and half class (1 digit) (logistic) is 0.27 - 1600 patents
# the real recall of the method with sectors and class (2 digits) (logistic) is 0.12 - 1600 patents
# the real recall of the mothod with subclass (4 characters) (logistic) is 0.029 - 1600 patents
#
# the real top recall for sectors: 0.57/0.53 - KNEIGHBOR CLASSIFIER/DECISION TREE CLASSIFIER - TFIDF
# the real top recall for sectors: 0.54/0.53 - KNEIGHBOR CLASSIFIER/LINEAR SVM - WORD2VEC
# the real top recall for sectors: 0.51 - LINEAR SVC - DOC2VEC
#
# the real top recall for subclasses: 0.17/0.23 - KNEIGHBOR CLASSIFIER/DECISION TREE CLASSIFIER - TFIDF # S.L.O.W.
# the real top recall for subclasses: 0.17/0.06 - KNEIGHBOR CLASSIFIER/LINEAR SVM - WORD2VEC
# the real top recall for subclasses: 0.17 - LINEAR SVC - DOC2VEC
#