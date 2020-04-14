# -*- coding: utf-8 -*-
import sys
import os
import csv
import datetime
import pandas as pd

import fasttext

sys.path.append(os.path.abspath('..'))
from helpers import folder_helper as fh
from helpers import tool_helper as th
from helpers import txt_data_helper as txth
from helpers import classification_helper as ch
from helpers import metrics_helper as mh
from helpers import predicting_model_helper as pmh
from helpers import word_model_helper as wmh

script_key = "fasttext classify"

def preprocessing_data_for_fasttext(data_frame, text_vectorizer, class_vectorizer):
    root_location = fh.get_root_location('data/fasttext_outcome/')

    data_frame['text'] = data_frame['text'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)

    model_name = text_vectorizer+'/'+class_vectorizer+'/FastText'
    try:
        X_train, X_test, Y_train, Y_test, _, _, _, _ = ch.apply_df_vectorizer(data_frame, text_vectorizer, class_vectorizer, model_name)

        X_train, X_val, Y_train, Y_val = ch.get_train_test_from_data(X_train, Y_train)

        # self.data_vectors = pd.DataFrame(columns=range(vectors_size), index=range(corpus_size))
        if not isinstance(X_train, pd.DataFrame):
            train = pd.DataFrame(data=X_train)
            test = pd.DataFrame(data=X_test)
            val = pd.DataFrame(data=X_val)
            # test_labels = pd.DataFrame(columns=[''])
        else:
            train = X_train
            test = X_test
            val = X_val

        train.loc[:, 1] = Y_train
        test.loc[:, 1] = Y_test
        val.loc[:, 1] = Y_val

        train.drop(columns=['patent_id'], inplace=True)
        test.drop(columns=['patent_id'], inplace=True)
        val.drop(columns=['patent_id'], inplace=True)

        data_frame.to_csv(fh.link_paths(root_location, 'dataframe.csv'), index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")

        train.to_csv(fh.link_paths(root_location, 'training set.csv'), index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
        test.to_csv(fh.link_paths(root_location, 'testing set.csv'), index=False, sep=',', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
    except:
        print('a problem occurred while trying to store the dataframes')

        X_train, X_test, Y_train, Y_test, _, _, _, _ = ch.apply_df_vectorizer(data_frame, text_vectorizer, class_vectorizer, model_name)

        X_train, X_val, Y_train, Y_val = ch.get_train_test_from_data(X_train, Y_train)

        val = pd.DataFrame({'text': X_val, 'classification': Y_val})
        train = pd.DataFrame({'text': X_train, 'classification': Y_train})
        test = pd.DataFrame({'text': X_test, 'classification': Y_test})

        data_frame.to_csv(fh.link_paths(root_location, 'dataframe.csv'), index=False, sep=' ', header=False,quoting=csv.QUOTE_NONE,quotechar="",escapechar=" ")

        val.to_csv(fh.link_paths(root_location, 'validating set.csv'), index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="",escapechar=" ")
        train.to_csv(fh.link_paths(root_location, 'training set.csv'), index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="",escapechar=" ")
        test.to_csv(fh.link_paths(root_location, 'testing set.csv'), index=False, sep=',', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")

def apply_fasttext(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, dataset_location):
    root_location = fh.get_root_location('data/fasttext_outcome/')

    fasttext_location = fh.join_paths(root_location, 'fasttext_models/')
    word2vec_location = fh.join_paths(root_location, 'word2vec_models/')
    timestamp = datetime.datetime.now().isoformat()

    #############################

    # unsupervised model

    # model = fasttext.train_unsupervised(input=fh.link_paths(root_location, 'training set.csv'),
    #                                   autotuneValidationFile=fh.link_paths(root_location, 'training set.csv'))

    #############################

    # supervised model

    # model = fasttext.train_supervised(input=fh.link_paths(root_location, 'training set.csv'),
    #                                   autotuneValidationFile=fh.link_paths(root_location, 'validating set.csv'), verbose=3, autotuneDuration=5000)

    #############################

    model, parameters = pmh.get_fasttext(fh.link_paths(root_location, 'training set.csv'))

    pmh.save_fasttext_model(model, fh.link_paths(fasttext_location, 'model '+timestamp+'.bin'))
    # pmh.load_fasttext_model(fh.link_paths(fasttext_location, 'model '+timestamp+'.bin'))

    test_labels, predictions, results = pmh.predict_test_fasttext(model, fh.link_paths(root_location, 'testing set.csv'))

    # print('predicted labels {}, probabilities of the labels {}'.format(predictions[0], predictions[1]))

    result_top_15 = model.test(fh.link_paths(root_location, 'testing set.csv'), k=15)
    result_top_8 = model.test(fh.link_paths(root_location, 'testing set.csv'), k=8)
    result_top_5 = model.test(fh.link_paths(root_location, 'testing set.csv'), k=5)
    result_top_3 = model.test(fh.link_paths(root_location, 'testing set.csv'), k=3)
    result_top_1 = model.test(fh.link_paths(root_location, 'testing set.csv'), k=1)

    classifier_name = ch.get_fasttext_classifier_information(str(model))
    model_name = text_vectorizer+'/'+class_vectorizer+'/'+classifier_name

    mh.display_directly_metrics('k=-1 '+classifier_name, -1, results[1], results[2], 2*(results[1]*results[2])/(results[1]+results[2]))
    mh.display_directly_metrics('k= 15 '+classifier_name, -1, result_top_15[1], result_top_15[2], 2*(result_top_15[1]*result_top_15[2])/(result_top_15[1]+result_top_15[2]))
    mh.display_directly_metrics('k= 8 '+classifier_name, -1, result_top_8[1], result_top_8[2], 2*(result_top_8[1]*result_top_8[2])/(result_top_8[1]+result_top_8[2]))
    mh.display_directly_metrics('k= 5 '+classifier_name, -1, result_top_5[1], result_top_5[2], 2*(result_top_5[1]*result_top_5[2])/(result_top_5[1]+result_top_5[2]))
    mh.display_directly_metrics('k= 3 '+classifier_name, -1, result_top_3[1], result_top_3[2], 2*(result_top_3[1]*result_top_3[2])/(result_top_3[1]+result_top_3[2]))
    mh.display_directly_metrics('k= 1 '+classifier_name, -1, result_top_1[1], result_top_1[2], 2*(result_top_1[1]*result_top_1[2])/(result_top_1[1]+result_top_1[2]))

    ch.save_results(classifier_name, results, parameters, model_name, classif_level, classif_type, dataset_location)

    manual_metrics = mh.calculate_manual_metrics(model_name, test_labels, predictions)
    none_average, binary_average, micro_average, macro_average = manual_metrics

    # print(model.test_label()) # path is missing

    # list_ = model.words
    # new_list = []

    # for token in list_:
    #     if len(token) in [0,1,2]:
    #         new_list.append(token)
    #     elif len(token) > 29:
    #         new_list.append(token)

    # # print(new_list)
    # print(len(new_list))

    print(model.labels)

def shrink_to_sectors(classcode_list):
    new_list = []
    for class_ in classcode_list:
        if class_[0] not in new_list:
            new_list.append(class_[0])
    return ' '.join(element for element in new_list)

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
        # source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_classification/cleaned/B - 500 patents/']

    text_vectorizer = 'None'
    # None - doesn't work
    # tfidf - no error it estimates but not the suitable number of variables
    # word2vec - no error but doesn not estimate
    # doc2vec - no error but does not estimate
    class_vectorizer = 'fasttext'
    classif_level = 'description_claim_abstract_title'
    classif_type = 'subclasses'

    patent_ids, temp_df, classifications_df = txth.load_data(source_path)
    data_frame, classif_level, classif_type = txth.get_final_df(patent_ids, temp_df, classif_type)

    data_frame = ch.further_preprocessing_phase(data_frame)
    # data_frame['classification'] = data_frame['classification'].apply(lambda classcode : shrink_to_sectors(th.tokenize_text(classcode)))

    preprocessing_data_for_fasttext(data_frame, text_vectorizer, class_vectorizer)

    apply_fasttext(data_frame, text_vectorizer, class_vectorizer, classif_level, classif_type, source_path)

    print("end fasttext classification step")

# TODO list:
# try all the state-of-the art and maybe we can combine two of them.
#
# little trouble: i have to save two CSVs (one for testing one for training) in order to feed the model with them. It doesn't work directly with dataframes. is it normal?
#
# quite big trouble: the only way to specify the model for learning the word representation (cbow or skipgram) is using the unsupervised learning with the argument "model". There isn't the possibility to add it to the supervised. if i apply the vectorization before it works with impressively much less time.
#
# k means for? predictions
#
# gensim ha a fasttext implementation
#
# edited the shrink to sectors and label algorithm