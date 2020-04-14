import sys
import os
from collections import namedtuple
import csv
csv.field_size_limit(sys.maxsize)
import pandas as pd
from tqdm import tqdm
import multiprocessing
import datetime

from sklearn import utils
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

ssh_source_dir = '/chakrm/workspace/2019-MastersProject/'
# sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('..')+ssh_source_dir)

from helpers import classification_helper as ch
from helpers import folder_helper as fh
from helpers import tool_helper as th
from helpers import txt_data_helper as txth
from helpers import word_model_helper as wmh
from helpers import lstm_readers_helper as lrh

script_key = "sequential_doc2vec"

GLOBAL_VARS = namedtuple('GLOBAL_VARS', ['MODEL_NAME', 'DOC2VEC_MODEL_NAME'])

def get_doc2vec_model(model_path):
    return Doc2Vec.load(model_path)

def create_tuple_array(data_frame, ids, text_batch_size=10000):
    sentences = []
    for index, row in enumerate(data_frame.iterrows()):
        temp_text = row[1]['text'].split(" ")
        len_line_array = len(temp_text)
        curr_batch_iter = 0

        if text_batch_size == 10000:
            sentences.append(LabeledSentence(words=temp_text, tags=[ids[index]]))
        else:
            while curr_batch_iter < len_line_array:
                sentences.append(LabeledSentence(words=temp_text[curr_batch_iter:curr_batch_iter+text_batch_size],
                                                 tags=[ids[index]]))
                curr_batch_iter += text_batch_size
    return tuple(sentences)

def utils_shuffle_rows(list_):
    return utils.shuffle(list_)

def train_doc2vec(data_frame, patent_ids, classif_level, classif_type):
    root_location = fh.get_root_location("data/lstm_outcome/")
    doc2vec_model_save_location = fh.join_paths(root_location, "doc2vec_model/")

    preprocessed_location = fh.join_paths(root_location, "preprocessed_data/separated_datasets/")
    training_preprocessed_files_prefix = fh.join_paths(preprocessed_location, "training_docs_data_preprocessed/")
    validation_preprocessed_files_prefix = fh.join_paths(preprocessed_location, "validation_docs_data_preprocessed/")
    test_preprocessed_files_prefix = fh.join_paths(preprocessed_location, "test_docs_data_preprocessed/")

    vocab_path = fh.join_paths(doc2vec_model_save_location, "vocab_model")

    training_docs_iterator = create_tuple_array(data_frame, patent_ids, text_batch_size=10000)

    #####
    tagged_data = training_docs_iterator
    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=1, vector_size=200, window=2, negative=10, sample=1e-8, hs=0, min_count=50,
                         alpha=0.25, min_alpha=0.05, dbow_words=0, seed=1234, concat=0, workers=cores)
    model_dbow.build_vocab([x for x in tqdm(tagged_data)])

    for epoch in range(30):
        # model_dbow.train(utils_shuffle_rows([x for x in tqdm(tagged_data)]), total_examples=len(tagged_data), epochs=1)
        model_dbow.train(utils_shuffle_rows([x for x in tqdm(tagged_data)]), total_examples=len(tagged_data), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    date = datetime.datetime.now().isoformat()
    model_dbow.save(fh.link_paths(vocab_path, 'doc2vec_vocab_30_epochs'))
    #####

    params = wmh.get_parameters_lstm_doc2vec()
    GLOBAL_VARS.DOC2VEC_MODEL_NAME, placeholder_model_name, doc2vec_model = wmh.get_lstm_doc2vec(params, classif_level, classif_type)

    # yields a list of sentences id, text as a tuple or (id, tuple)
    # training_docs_iterator = lrh.BatchWrapper(training_preprocessed_files_prefix, text_batch_size=10000, level=classif_level,
    #                                       level_type=classif_type)
    doc2vec_model.build_vocab(documents=training_docs_iterator, progress_per=params[13])
    doc2vec_model.save(fh.link_paths(vocab_path, "doc2vec_vocab"))

    DOC2VEC_ALPHA_DECREASE = wmh.set_alpha_parameters_lstm_doc2vec(doc2vec_model)
    start_epoch = 1

    # for epoch in range(1, params[11] + 1):
    #     GLOBAL_VARS.MODEL_NAME = placeholder_model_name.format(epoch)
    #     doc2vec_folder_path = fh.join_paths(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME)
    #     if fh.ensure_exists_path_location(fh.link_paths(doc2vec_folder_path, "doc2vec_model")):
    #         start_epoch = epoch

    # if start_epoch > 1:
    #     GLOBAL_VARS.MODEL_NAME = placeholder_model_name.format(start_epoch)
    #     doc2vec_folder_path = fh.join_paths(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME)
    #     # if a model of that epoch already exists, we load it and proceed to the next epoch
    #     doc2vec_model = Doc2Vec.load(fh.link_paths(doc2vec_folder_path, "doc2vec_model"))
    #     start_epoch += 1

    ## The Actual Training
    for epoch in range(start_epoch, params[11] + 1):
        print("### epoch "+str(epoch)+" ###")
        # set new filename/path to include the epoch
        GLOBAL_VARS.MODEL_NAME = placeholder_model_name.format(epoch)
        doc2vec_folder_path = fh.join_paths(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME)
        # train the doc2vec model
        # training_docs_iterator = lrh.BatchWrapper(training_preprocessed_files_prefix, text_batch_size=10000, level=classif_level,
        #                                 level_type=classif_type) # yields a list of sentences id, text as a tuple or (id, tuple)

        doc2vec_model.train(documents=training_docs_iterator, total_examples=len(training_docs_iterator),
                            report_delay=params[12], epochs=params[10])
        doc2vec_model.alpha -= DOC2VEC_ALPHA_DECREASE  # decrease the learning rate
        doc2vec_model.min_alpha = doc2vec_model.alpha  # fix the learning rate, no decay
        doc2vec_model.save(fh.link_paths(doc2vec_folder_path, "doc2vec_model"))

    if epoch != params[11]:
        print("still training epochs missing: " + str(epoch))
        sys.exit(1)

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

    text_vectorizer = 'None'
    class_vectorizer = 'multi_label'
    classif_level = 'description'
    classif_type = 'subclasses'

    # patent_ids, temp_df, classifications_df = txth.load_data(source_path)
    # data_frame, classif_level, classif_type = txth.get_final_df(patent_ids, temp_df)

    csvfile = 'training_2000_2004_cleaned_2.csv'
    csvfile = 'training_2000_2004_cleaned_abstract_2.csv'
    # save_data_frame(script_key, data_frame, source_path)
    classif_level, classif_type = 'description_claim_abstract_title', 'subclasses'
    training_df, classification_df = ch.load_data_frame(script_key, csvfile)

    csvfile = 'testing_2005_cleaned_2.csv'
    csvfile = 'testing_2005_cleaned_abstract_2.csv'
    testing_df, classification_df = ch.load_data_frame(script_key, csvfile)

    data_frame = pd.concat([training_df, testing_df])
    patent_ids = data_frame['patent_id'].tolist()

    train_doc2vec(data_frame, patent_ids, classif_level, classif_type)

    print("end training doc2vec model...")
