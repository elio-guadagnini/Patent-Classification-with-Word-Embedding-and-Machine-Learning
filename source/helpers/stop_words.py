# -*- coding: utf-8 -*-
import sys
print(sys.version_info)
import xml.etree.cElementTree as et
import pandas as pd
import glob
import os
import stat
import string
import re
from spacy.lang.en import English
from collections import Counter
import time
from tqdm import tqdm

from stemming.porter2 import stem

import nltk
from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer

import string

def to_lowercase(text):
    return text.lower()

def remove_punctuation_2(text):
    return re.sub(r'[^\w\s]','', text)

# def remove_punctuation(text):
#     table = str.maketrans('', '', string.punctuation)
#     print(table)
#     return [w.translate(table) for w in text]

def tokenize_text(text):
    return text.split()

def remove_stop_words(tokens, language):
    stop_words = stopwords.words(language)
    return [word for word in tokens if word not in stop_words]

def stemming(tokens):
    return [stem(word) for word in tokens]

if __name__ == '__main__':
    test = 'Hello everyone this is a test of how NLTK promps work, let\'s hope that everything works. Because otherwise, we are fucked'
    test = to_lowercase(test)
    test = remove_punctuation_2(test)
    test = tokenize_text(test)
    test = remove_stop_words(test, 'english')
    print(test)

    test = stemming(test)
    print(test)