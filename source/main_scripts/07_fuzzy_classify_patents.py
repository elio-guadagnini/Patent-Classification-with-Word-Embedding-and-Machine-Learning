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
from helpers import lstm_doc2vec_helper as ldh

script_key = "fuzzy classify"



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

    import pandas as pd
    phrases = [ ('first word here', 'again here 1', 'class1') ,
                 ('is not the', 'you don\'t 2', 'class2') ,
                 ('second in abstract', 'have the 3', 'class3') ,
                 ('but is the', 'same distribution 1', 'class4') ,
                 ('first here for', 'you had 3', 'class5') ,
                 ('sure this is', 'before here 2', 'class6')  ]
    test_df = pd.DataFrame(phrases, columns=['abstract', 'claim', 'classification'])

    text_vectorizer = 'word2vec'
    class_vectorizer = 'multi_label'
    classif_level = 'description_claim_abstract_title'
    classif_type = 'subclasses'

    patent_ids, temp_df, classifications_df = txth.load_data(source_path)

    temp_df = test_df

    # abstract claim description id
    # print(temp_df)
    # print(patent_ids)
    # print(classifications_df)

    helper_doc2vec = wmh.Dov2VecHelper()
    abstract_temp_df = helper_doc2vec.label_sentences(temp_df['abstract'], 'Abstract')
    claim_temp_df = helper_doc2vec.label_sentences(temp_df['claim'], 'Claim')
    # description_temp_df = helper_doc2vec.label_sentences(temp_df['description'], 'Description') # here if it is None will raise an exception

    # print(temp_df['abstract'])
    # print(temp_df['claim'])

    abstract_dbow_model = wmh.train_doc2vec(abstract_temp_df, '/Users/elio/Desktop/temp/model_path/abstract')
    claim_dbow_model = wmh.train_doc2vec(claim_temp_df, '/Users/elio/Desktop/temp/model_path/claim')
    # description_dbow_model = wmh.train_doc2vec(temp_df['description'], '/Users/elio/Desktop/temp/model_path/description')

    abstract_vectors_dbow = helper_doc2vec.get_vectors(abstract_dbow_model, len(abstract_temp_df), 150, 'Abstract')
    claim_vectors_dbow = helper_doc2vec.get_vectors(claim_dbow_model, len(claim_temp_df), 150, 'Claim')
    # description_vectors_dbow = helper_doc2vec.get_vectors(description_model_dbow, len(description_temp_df), 150, 'Description')

    # print(type(abstract_vectors_dbow))
    # print(type(claim_vectors_dbow))
    # # print(type(description_vectors_dbow))
    # print(temp_df['abstract'].shape)
    # print(temp_df['claim'].shape)

    Y, classes, n_classes = ch.apply_classification_vectorizer(class_vectorizer, temp_df)

    # print(abstract_vectors_dbow)
    # print(claim_vectors_dbow)

    # print(Y)
    # print(classes)
    # print(n_classes)

    vectors = np.concatenate((abstract_vectors_dbow, claim_vectors_dbow), axis=1)

    # print(vectors)
    # print(vectors.shape)

    data_frame = pd.DataFrame(columns=['text', 'classification'])
    for index in range(vectors.shape[0]):
        data_frame.loc[data_frame.shape[0]+1] = [vectors[index, :], Y[index, :]]

    # TODO: i should have defined the data_frame
    # here i have 150+150+150 for three texts (abstract, claim, description)
    print(data_frame)

    X_train, X_test, y_train, y_test = ch.get_train_test_from_dataframe(data_frame)

    print("end fuzzy classifition step")

# add agorithm to display conv metrics, removed a del in 414 classification helper

# TODO list:
# try all the state-of-the art and maybe we can combine two of them.
#
# get_lstm_shapes, try with validation data. change batch size. change testing parameters according to training results
#
# fuzzy with splitted vectorizer (abstract, description, claims) and another model for estimating the classes