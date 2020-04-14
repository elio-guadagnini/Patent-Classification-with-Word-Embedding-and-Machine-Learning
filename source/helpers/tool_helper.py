# -*- coding: utf-8 -*-
import sys
import os
import glob
import numpy as np
from scipy.sparse import lil_matrix

sys.path.append(os.path.abspath('..'))
from helpers import folder_helper as fh

#########################################################################################################################
# get started tools:

def source_path_warnings():
    print("\nWARNING!!!! Check out the source_path.")
    print("You may have entered a source path including the patent folder, avoid it.")
    print("Don't forget to avoid also the final / character!")
    return [], 0

def handle_complete_args(source_path, folder_level):
    source_path = fh.link_paths(source_path, '*')
    source_path = fh.get_list_files(source_path, None)
    source_path = list(map(lambda path : path + '/', source_path))
    print("source path: %s" % source_path)
    folder_level = int(folder_level)
    print("folder destination level: %s" % folder_level)

    if len(source_path) == 0 or source_path[len(source_path)-1][-5:-1] == '.xml':
        return source_path_warnings()
    return source_path, folder_level

def handle_partial_args(source_path):
    source_path = fh.link_paths(source_path, '*')
    source_path = fh.get_list_files(source_path, None)
    source_path = list(map(lambda path : path + '/', source_path))
    print("source path: %s" % source_path)

    if len(source_path) == 0 or source_path[len(source_path)-1][-5:-1] == '.xml':
        return th.source_path_warnings()
    else:
        folder_level = source_path[0].count('/')-1
        print("folder destination level: %s" % folder_level)
        return source_path, folder_level

#########################################################################################################################
# filename tools:

def get_region_doc_number(patent_data):
    return patent_data["patent-country"] + patent_data["patent-doc-number"]

def get_us_filename(region_filename):
    return region_filename + '.xml'

def get_eu_filename(path_filename):
    index = path_filename.rfind('/')
    return path_filename[index+1:len(path_filename)]

def get_patent_id(string):
    index = string.rfind('/')
    return string[index+1:-4]

#########################################################################################################################
# lambda functions:

def get_node_value(node):
    # node text or None
    return node.text if node is not None else None

def get_flat_list(list_of_lists):
    return [element for elements in list_of_lists for element in elements]

def get_flat_super_list(list_of_list_of_lists):
    return [element for super_elements in list_of_list_of_lists for elements in super_elements for element in elements]

def unique_list(list_):
    temp_list = []
    [temp_list.append(element) for element in list_ if element not in temp_list]
    return temp_list

def get_string_from_list(list_, linking_string):
    return linking_string.join(element for element in list_ if isinstance(element, str))

def handle_ending_node(node, marker):
    text = list(map(lambda correct_subnode : get_node_value(correct_subnode), filter(lambda sub_node : sub_node.tag == marker, node)))
    return text.remove(None) if None in text else text

def handle_class(class_, index, index_2):
    classcode = get_node_value(class_)
    classcode = classcode.replace(" ", "")
    classcode = classcode.replace("/", "")
    if len(classcode) >= 4:
        return classcode[index:index_2]
    return ""

def handle_class_node(node, index, index_2, marker):
    text = list(map(lambda correct_subnode : handle_class(correct_subnode, index, index_2), filter(lambda sub_node : sub_node.tag == marker, node)))
    return text.remove(None) if None in text else text

def handle_citation(citation):
    text = get_node_value(citation)
    index = text.rfind('-')
    if index != -1:
        text = text.replace(" ", "")
        return text[index+1:]
    return text

def handle_citation_subnode(sub_node, marker_1, marker_2):
    text = list(map(lambda correct_cit : handle_citation(correct_cit), filter(lambda cit : cit.tag == marker_1 or cit.tag == marker_2, sub_node)))
    return text.remove(None) if None in text else text

def handle_citation_node(node, marker_1, marker_2):
    text = list(map(lambda correct_subnode : handle_citation_subnode(correct_subnode, 'text', 'date'), filter(lambda sub_node : sub_node.tag == marker_1 or sub_node.tag == marker_2, node)))
    return text.remove(None) if None in text else text

#########################################################################################################################
# word tools:

def to_uppercase(text):
    return text.upper()

def to_lowercase(text):
    return text.lower()

def tokenize_text(text):
    return text.split()

def tokenize_complex_text(text):
    return list(map(lambda correct_word : correct_word, filter(lambda word : len(word) >= 2, tokenize_text(text))))

# def tokenize_complex_text(text):
#     tokens = []
#     # for sent in nltk.sent_tokenize(text):
#     #    print("sent: ", sent)
#     for word in tokenize_text(text):
#         if len(word) < 2:
#             continue
#         tokens.append(word)
#     return tokens

def tokenize_complex_text_in_set(text):
    return set(map(lambda word : word, tokenize_text(text)))

# def tokenize_complex_text_in_set(text):
#     tokens = {""}
#     # for sent in nltk.sent_tokenize(text):
#     #     print("sent: ", sent)
#     for word in tokenize_text(text):
#         # if len(word) < 2:
#         #     continue
#         if "" in tokens:
#             tokens = {word}
#         else:
#             tokens.add(word)
#     return tokens

#########################################################################################################################
# checking files:

def check_us_infomation(patent_data, tag):
    if tag not in patent_data.keys() or patent_data[tag] == None:
        patent_data[tag] = ""

def check_xml_variables(patent_data, tag_list):
    for tag in tag_list:
        check_us_infomation(patent_data, tag)

def check_none_information(string):
    return string if string is not None else ""

def check_variables(variable_list):
    return list(map(lambda var : check_none_information(var), variable_list))

#########################################################################################################################
# stop words:

def load_english_stop_words():
    return ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

#########################################################################################################################
# utils for reading txts:

def cut_down(classification, index, top_classes, only_top_classes):
    temp_list = tokenize_text(classification)
    if only_top_classes:
        new_classification_list = list(set(filter(lambda element : element[:index] in top_classes, temp_list)))
        new_classification = get_string_from_list(new_classification_list, ' ')
    else:
        new_classification_list = list(set(map(lambda element : element[:index], temp_list)))
        new_classification = get_string_from_list(new_classification_list, ' ')
    return new_classification

# def cut_down(classification, index, top_classes, only_top_classes):
#     temp_list = tokenize_text(classification)
#     if only_top_classes:
#         new_classification = ""
#         for element in temp_list:
#             if element[:index] in top_classes:
#                 if not element[:index] in new_classification:
#                     if new_classification != "":
#                         new_classification = new_classification + " " + element[:index]
#                     else:
#                         new_classification = element[:index]
#     else:
#         for i, element in enumerate(temp_list):
#             if i != 0:
#                 if not element[:index] in new_classification:
#                     new_classification = new_classification + " " + element[:index]
#             else:
#                 new_classification = element[:index]
#     return new_classification

def calculate_class_distribution(classification, classifications_df):
    for _class in tokenize_text(classification):
        if classifications_df['class'].str.contains(_class).any():
            index = classifications_df.index[classifications_df['class'] == _class]
            classifications_df.loc[index[0], ['count']] += 1
        else:
            classifications_df.loc[classifications_df.shape[0] + 1] = [_class, 1]
    return classifications_df

#########################################################################################################################
# utils data:

def get_set_from_index(index_set, data_set):
    set_ = np.ndarray(shape=(index_set.shape[0], data_set.shape[1]))
    for index, data_index in enumerate(index_set['text'].values):
        set_[index-1] = data_set[data_index-1]
    return set_, index_set['patent_id']

def get_lil_matrices(x_train, y_train, x_test):
    x_train = lil_matrix(x_train).toarray()
    y_train = lil_matrix(y_train).toarray()
    x_test = lil_matrix(x_test).toarray()
    return x_train, y_train, x_test

def handle_single_text(ft_model, text, window_length, n_features):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    words = text.split()
    window = words[-window_length:]

    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x

def get_vectors_from_dataframe(ft_model, df, window_length, n_features):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in enumerate(df['text'].values):
        x[i, :] = handle_single_text(ft_model, comment, window_length, n_features)

    return x
