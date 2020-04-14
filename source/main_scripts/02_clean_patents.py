# -*- coding: utf-8 -*-
import sys
import time
import pandas as pd
import glob
import os
import stat
import string
import re
import csv
from tqdm import tqdm
import xml.etree.cElementTree as et
from collections import Counter

# from stemming.porter2 import stem
import krovetz

# from spacy.lang.en import English
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer

sys.path.append(os.path.abspath('..'))
from helpers import folder_helper as fh
from helpers import tool_helper as th
from helpers import txt_data_helper as txth
from helpers import xml_data_helper as xmlh

script_key = "clean"

def get_alternative_text(patent_document, marker):
    text = list(map(lambda sub_node : th.get_node_value(sub_node), filter(lambda node: node.tag == marker, patent_document)))
    if text:
        if None in text:
            text.remove(None)
        return th.get_string_from_list(text, " ")
    return ""

def get_text(patent_document, marker):
    for node in patent_document:
        if node.tag == marker:
            # print("original: ", th.get_node_value(node))
            return th.get_node_value(node)
    return ""

def remove_numbers(text):
    return re.sub('[^a-zA-Z]+', ' ', text)

def remove_puntuation(text):
    return re.sub(r'[^\w\s]','', text)

# def remove_alphabetic(words):
#     return [word for word in words if word.isalpha()]

def remove_alternative_stop_words(tokens):
    stop_words = th.load_english_stop_words()
    # print("alternative: ", list(map(lambda word_ : word_, filter(lambda word : word not in stop_words, tokens))))
    return list(map(lambda word_ : word_, filter(lambda word : word not in stop_words, tokens)))

def remove_stop_words(tokens):
    stop_words = th.load_english_stop_words()
    # print("original: ", [word for word in tokens if word not in stop_words])
    return [word for word in tokens if word not in stop_words]

# def lemmatization_algorithm(words):
#     lemmatizer = WordNetLemmatizer()
#     return [lemmatizer.lemmatize(word) for word in words]

# def stemming_algorithm(words): # through PorterStemming algorithm
#     porter = PorterStemmer()
#     return [porter.stem(word) for word in words]

def krovetz_alternative_stemming_algorithm(tokens):
    ks = krovetz.PyKrovetzStemmer()
    # print("alternative: ", list(map(lambda word : ks.stem(word), tokens)))
    return list(map(lambda word : ks.stem(word), tokens))

def krovetz_stemming_algorithm(tokens):
    ks = krovetz.PyKrovetzStemmer()
    # print("original: ", [ks.stem(word) for word in tokens])
    return [ks.stem(word) for word in tokens]

# moved to krovetz
# def porter_stemming_algorithm_without_nltk(tokens):
#    return [stem(word) for word in tokens]

# def remove_words_on_occurrences(words, occurrences):
#     remove_list = []
#     count = 0
#     count_tot = 0
#     occurrences_words = Counter(words)
#     for word, value in occurrences_words.items():
#         if value <= occurrences:
#             remove_list.append(word)
#             count += 1
#         count_tot += 1
#     # print("to be removed ", count)
#     # print("out of ", count_tot)
#     return remove_list

# def remove_numeric_words(words):
#     new_words = []
#     for word in words:
#         if not word[0].isdigit():
#             new_words.append(word)
#     return new_words

# def remove_words_on_length(words, suitable_length):
#     new_words = []
#     for word in words:
#         if len(word) >= suitable_length:
#             new_words.append(word)
#     return new_words

def clean_text(text):
    text = remove_numbers(text)

    # stop words
    # pre-processing (NLTK)

    # remove punctuation and numbers word
    text = remove_puntuation(text)

    # normalizing step - convert to lower case
    text = th.to_lowercase(text)

    # split into tokens by white spaces
    # tokens = nltk.word_tokenize(text)
    tokens = th.tokenize_text(text)

    # # is it useful to tag the text?
    # tagged = nltk.pos_tag(tokens)
    # # identify named entities
    # entities = nltk.chunk.ne_chunk(tagged)

    # remove remaining tokens that are not alphabetic
    # words = remove_alphabetic(tokens)

    # remove stop words
    tokens = remove_alternative_stop_words(tokens)

    # lemmatization of words
    # lemmatized = lemmatization_algorithm(words_without_stops)

    # stemming of words
    tokens = krovetz_alternative_stemming_algorithm(tokens)
    # words = stemming_algorithm(words)
    # tokens = porter_stemming_algorithm_without_nltk(tokens)

    # drop words that do not have at least 4 occcurences
    # to_be_removed = remove_words_on_occurrences(stemmed, 4)
    # new_stemmed = list(set(stemmed) - set(to_be_removed))

    # drop words that start with numeric character
    # new_stemmed = remove_numeric_words(new_stemmed)

    # not do the mispelling
    # tokens = remove_word_on_length(tokens, 2)

    return tokens

def a1_parser(patent_document):
    # abstract = get_text(patent_document, 'abstract')
    abstract = get_alternative_text(patent_document, 'abstract')
    if abstract != None:
        cleaned_abstract = clean_text(abstract)
    else:
        cleaned_abstract = abstract

    # citations = get_text(patent_document, 'citations')
    citations = get_alternative_text(patent_document, 'citations')

    return get_alternative_text(patent_document, 'class-code'), get_alternative_text(patent_document, 'applicant'), cleaned_abstract, citations

def b1_parser(patent_document):
    classcode, applicant, cleaned_abstract, citations = a1_parser(patent_document)

    # claim = get_text(patent_document, 'claim')
    claim = get_alternative_text(patent_document, 'claim')
    if claim != None:
        cleaned_claim = clean_text(claim)
    else:
        cleaned_claim = claim

    temp = get_text(patent_document, 'description')
    description = get_alternative_text(patent_document, 'description')
    if temp != description:
        print("different")
    if description != None:
        cleaned_description = clean_text(description)
    else:
        cleaned_description = description
    return classcode, applicant, cleaned_abstract, citations, cleaned_claim, cleaned_description

def eu_parser(patent_document, attributes, filename, kind, destination_path, mixed_destination_path, csv_dataframe, patent_data):
    if kind == 'A1': # bibliografy - index
        classcode, applicant, abstract, citations = a1_parser(patent_document)

        variable_list = [classcode, applicant, abstract, citations]
        variable_list = th.check_variables(variable_list)

        txth.write_eu_a1_text_patent(destination_path, filename, kind, variable_list[0], variable_list[1], variable_list[2], variable_list[3])

        # checking the third dataset
        if xmlh.is_in_csv_alternative(filename[:-8], csv_dataframe, 'file'):
        # if xmlh.is_in_csv(filename[:-8], csv_eu_path):
            patent_data.loc[patent_data.shape[0] + 1] = [filename[:-8], variable_list[1], variable_list[3]]

    elif kind == 'B1': # text - data
        id_respective_document = attributes['id_respective_document']
        classcode, _, abstract, citations, claim, description = b1_parser(patent_document)

        variable_list = [classcode, abstract, claim, description, citations]
        variable_list = th.check_variables(variable_list)

        txth.write_eu_b1_text_patent(destination_path, filename, kind, id_respective_document, variable_list[0], variable_list[1], variable_list[2], variable_list[3], variable_list[4])

        # creating the third dataset
        idx = patent_data.index[patent_data['filename'] == id_respective_document[:-4]].tolist()
        if len(idx) > 0:
            kind = 'A1B1'
            applicant = patent_data.loc[idx]['applicant'].iloc[0]
            citations = patent_data.loc[idx]['citations'].iloc[0]
            txth.write_eu_mix_text_patent(mixed_destination_path, filename, kind, classcode, applicant, abstract, claim, description, citations)

def us_parser(patent_document, filename, kind, destination_path):
    classcode, applicant, abstract, citations, claim, description = b1_parser(patent_document)

    variable_list = [classcode, applicant, abstract, claim, description, citations]
    variable_list = th.check_variables(variable_list)

    txth.write_us_text_patent(destination_path, filename, kind, variable_list[0], variable_list[1], variable_list[2],
                                variable_list[3], variable_list[4], variable_list[5])

def clean_patents(source_path, destination_path, mixed_destination_path, csv_eu_path):
    """ clean_patents """
    patent_data = pd.DataFrame(columns=['filename', 'applicant', 'citations'])
    csv_dataframe = xmlh.load_index_csv(csv_eu_path)
    for index, path in enumerate(source_path):
        files = fh.get_list_files(path, 'xml')
        for i in tqdm(range(len(files))):
            try:
                path_filename = files[i]
                # print("file: ", path_filename)
                parsed_xml = et.parse(path_filename)
                patent_document = parsed_xml.getroot()

                # country, date, doc_n, dtd_ver, file, id, kind, lang, status
                attributes = patent_document.attrib
                lang = attributes['lang'].upper()
                kind = attributes['kind'].upper()

                if lang == 'EN':
                    filename = attributes['file']
                    region = filename[:2].upper()
                    if region == 'EP':
                        eu_parser(patent_document, attributes, filename, kind, destination_path[index], mixed_destination_path[index], csv_dataframe, patent_data)
                    elif region == 'US':
                        us_parser(patent_document, filename, kind, destination_path[index])
            except:
                print("WARNING!!!! Check out the patent: ", path_filename)
                continue

if __name__ == '__main__':
    try:
        if len(sys.argv) == 3:
            # source_path = 'test_clean/*/directories - and inside all the patents'
            source_path = sys.argv[1]
            folder_level = sys.argv[2]

            # here the source_path must be passed as a string of the root directory of all data folders!
            source_path, folder_level = th.handle_complete_args(source_path)
        elif len(sys.argv) == 2:
            # source_path = 'test_clean/*/directories - and inside all the patents'
            source_path = sys.argv[1]

            # here the source_path must be passed as a string of the root directory of all data folders!
            source_path, folder_level = th.handle_partial_args(source_path)
        else:
            print(usage())
            sys.exit(1)
    except:
        # A1 folders before the B1 ones!
        # FOLDER LEVEL - indicates at which level you want to create the new folders:
        # less it is, higher is the level of the folders -> 1 - folder_a/__HERE__/, 2 - folder_a/folder_b/__HERE__/
        # suggestion: (n of slashes)-1, taking into account that the path must end with the slash

        source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/parsed/B/', '/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/parsed/A/']
        folder_level = source_path[0].count('/')-1

    try:
        start = time.time()

        csv_eu_path, csv_us_path = xmlh.get_csv_path(script_key)

        destination_path = fh.get_destination_path(source_path, folder_level, script_key)
        mixed_destination_path = fh.get_destination_path(source_path, folder_level, "clean_mix")

        clean_patents(source_path, destination_path, mixed_destination_path, csv_eu_path)

        print("\nend of cleaning")
        end = time.time()
        print("time: ", end-start)
    except:
        print("ERROR: during calculating destination path. Check it out!")

# to do list:
# should i save abstract, citations in B1? remember to edit the classify script! otherwise it is the same as a1b1 (mix)
#
# Keeping only words of length of at least 2? commented