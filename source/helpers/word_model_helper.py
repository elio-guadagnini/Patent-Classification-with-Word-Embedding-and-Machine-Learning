# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import collections
import multiprocessing
import datetime
from copy import deepcopy
from tqdm import tqdm

from sklearn import utils

import gensim
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

# import fasttext

sys.path.append(os.path.abspath('..'))
from helpers import tool_helper as th
from helpers import folder_helper as fh
from helpers import classification_helper as ch

#########################################################################################################################

def get_tagged_document(row):
    return TaggedDocument(words=th.tokenize_complex_text(row['text']), tags=row.classification)

#########################################################################################################################
# doc2vec tools:

def utils_shuffle_rows(list_):
    return utils.shuffle(list_)

def train_doc2vec(tagged_data, model_path):
    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=1, vector_size=200, window=2, negative=10, sample=1e-7, hs=0, min_count=4,
                         alpha=0.25, min_alpha=0.05, dbow_words=0, seed=1234, concat=0, workers=cores)
    model_dbow.build_vocab([x for x in tqdm(tagged_data)])

    for epoch in range(30):
        # model_dbow.train(utils_shuffle_rows([x for x in tqdm(tagged_data)]), total_examples=len(tagged_data), epochs=1)
        model_dbow.train(utils_shuffle_rows([x for x in tqdm(tagged_data)]), total_examples=len(tagged_data), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    date = datetime.datetime.now().isoformat()
    model_dbow.save(fh.link_paths(model_path, 'doc2vec_model '+date))
    return model_dbow

# for dataframe structure
def train_alternative_doc2vec(tagged_data, model_path):
    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=1, vector_size=200, window=2, negative=10, sample=1e-8, hs=0, min_count=50,
                         alpha=0.25, min_alpha=0.05, dbow_words=0, seed=1234, concat=0, workers=cores)
    model_dbow.build_vocab([row[0] for index, row in tqdm(tagged_data.iterrows(), total=tagged_data.shape[0])])

    for epoch in range(30):
        model_dbow.train(utils_shuffle_rows([row[0] for index, row in tqdm(tagged_data.iterrows(), total=tagged_data.shape[0])]), total_examples=len(tagged_data), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    date = datetime.datetime.now().isoformat()
    model_dbow.save(fh.link_paths(model_path, 'doc2vec_model_for_data_frame '+date))
    return model_dbow

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

def get_concatenated_doc2vec(model_dbow, model_dmm):
    return ConcatenatedDoc2Vec([model_dbow, model_dmm])

import pandas as pd
class Dov2VecHelper():
    def __init__(self):
        print('###  doc2vec_vectorizer  ###')
        self.labeled = []
        self.vectors = []
        self.data_labeled = pd.DataFrame(columns=['data'])
        self.data_vectors = pd.DataFrame(columns=['data'])

    def label_sentences(self, corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the post.
        """
        self.labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            self.labeled.append(TaggedDocument(th.tokenize_text(v), [label]))
        return self.labeled

    def alternative_label_sentences(self, corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the post.
        """
        self.data_labeled = pd.DataFrame(columns=['data'])
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            self.data_labeled.loc[self.data_labeled.shape[0] + 1] = [TaggedDocument(th.tokenize_text(v), [label])]
        return self.data_labeled

    def get_vectors(self, model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        self.vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            self.vectors[i] = model.docvecs[prefix]
        return self.vectors

    def alternative_get_vectors(self, model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        self.data_vectors = pd.DataFrame(columns=range(vectors_size), index=range(corpus_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            self.data_vectors.loc[i] = model.docvecs[prefix] # every one has size 150
        return self.data_vectors

def set_alpha_parameters_lstm_doc2vec(model):
    model.alpha = 0.025
    model.min_alpha = 0.025
    DOC2VEC_ALPHA_DECREASE = 0.001
    return DOC2VEC_ALPHA_DECREASE

def get_parameters_lstm_doc2vec():
    DOC2VEC_SIZE = 200
    DOC2VEC_WINDOW = 2
    DOC2VEC_MAX_VOCAB_SIZE = None
    DOC2VEC_SAMPLE = 1e-3
    DOC2VEC_TYPE = 1
    DOC2VEC_HIERARCHICAL_SAMPLE = 0
    DOC2VEC_NEGATIVE_SAMPLE_SIZE = 10
    DOC2VEC_CONCAT = 0
    DOC2VEC_MEAN = 1
    DOC2VEC_TRAIN_WORDS = 0
    DOC2VEC_EPOCHS = 1 # we do our training manually one epoch at a time
    DOC2VEC_MAX_EPOCHS = 8
    REPORT_DELAY = 20 # report the progress every x seconds
    REPORT_VOCAB_PROGRESS = 100000 # report vocab progress every x documents

    DOC2VEC_EPOCH = 8

    MIN_WORD_COUNT = 100
    DOC2VEC_SEED = 1234
    NUM_CORES = 16

    return [DOC2VEC_SIZE, DOC2VEC_WINDOW, DOC2VEC_MAX_VOCAB_SIZE, DOC2VEC_SAMPLE, DOC2VEC_TYPE, DOC2VEC_HIERARCHICAL_SAMPLE,
            DOC2VEC_NEGATIVE_SAMPLE_SIZE, DOC2VEC_CONCAT, DOC2VEC_MEAN, DOC2VEC_TRAIN_WORDS, DOC2VEC_EPOCHS, DOC2VEC_MAX_EPOCHS,
            REPORT_DELAY, REPORT_VOCAB_PROGRESS, DOC2VEC_EPOCH, MIN_WORD_COUNT, DOC2VEC_SEED, NUM_CORES]

def set_parameters_lstm_doc2vec(out_path, classif_level, classif_type):
    params = get_parameters_lstm_doc2vec()

    placeholder_model_name = 'doc2vec_size_{}_w_{}_type_{}_concat_{}_mean_{}_trainwords_{}_hs_{}_neg_{}_vocabsize_{}_model_{}'.format(
                                    params[0],
                                    params[1],
                                    'dm' if params[4] == 1 else 'pv-dbow',
                                    params[7], params[8],
                                    params[9],
                                    params[5], params[6],
                                    str(params[2]),
                                    classif_level + '_' + classif_type)

    doc2vec_model_name = placeholder_model_name
    placeholder_model_name = fh.link_paths(placeholder_model_name, "epoch_{}")
    doc2vec_model_name_epoch = placeholder_model_name.format(params[14])

    fh.create_folder(fh.link_paths(out_path, doc2vec_model_name_epoch))
    return doc2vec_model_name, doc2vec_model_name_epoch

def get_lstm_doc2vec(params, classif_level, classif_type):
    placeholder_model_name = 'doc2vec_size_{}_w_{}_type_{}_concat_{}_mean_{}_trainwords_{}_hs_{}_neg_{}_vocabsize_{}_model_{}'.format(
                                    params[0],
                                    params[1],
                                    'dm' if params[4] == 1 else 'pv-dbow',
                                    params[7], params[8],
                                    params[9],
                                    params[5],params[6],
                                    str(params[2]),
                                    classif_level + '_' + classif_type)

    doc2vec_model_name = placeholder_model_name
    placeholder_model_name = placeholder_model_name + "_epoch_{}"

    doc2vec_model = Doc2Vec(size=params[0], window=params[1], min_count=params[15],
                            max_vocab_size=params[2],
                            sample=params[3], seed=params[16], workers=params[17],
                            # doc2vec algorithm dm=1 => PV-DM, dm=2 => PV-DBOW, PV-DM dictates CBOW for words
                            dm=params[4],
                            # hs=0 => negative sampling, hs=1 => hierarchical softmax
                            hs=params[5], negative=params[6],
                            dm_concat=params[7],
                            # would train words with skip-gram on top of cbow, we don't need that for now
                            dbow_words=params[9],
                            iter=params[10])
    return doc2vec_model_name, placeholder_model_name, doc2vec_model

#########################################################################################################################
# word2vec tools:

def get_word2vec_model(data_, out_path):
    cores = multiprocessing.cpu_count()
    # results_word2vec_sg_1_size_150_hs_1_cbow_mean_1_negative_5_min_count_1_lpha_0.25_min_alpha_0
    # results_word2vec_sg_0_size_150_hs_1_cbow_mean_1_negative_12_min_count_1_lpha_0.25_min_alpha_0
    model_dbow = Word2Vec(data_, sg=1, size=200, hs=0, cbow_mean=1, negative=10, min_count=50, alpha=.25, min_alpha=.0,
                          seed=1234, sample=1e-8, compute_loss=True, window=2, workers=cores)
    # model_dbow.build_vocab([x for x in tqdm(data_)])

    for epoch in range(30):
        model_dbow.train(utils_shuffle_rows([x for x in tqdm(data_)]), total_examples=len(data_), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    date = datetime.datetime.now().isoformat()
    model_dbow.save(fh.link_paths(out_path, 'word2vec_model ' + date))
    return model_dbow

class Word2VecHelper():
    def __init__(self):
        print('###  word2vec_vectorizer  ###')
        self.tokens = []
        self.mean = []

        self.vectors = []

    def w2v_tokenize_text(self, text):
        self.tokens = []
        for word in th.tokenize_text(text):
            if len(word) < 2:
                continue
            self.tokens.append(word)
        return self.tokens

    def word_averaging(self, wv, words):
        all_words = set()
        self.mean = []

        for word in words:
            if isinstance(word, np.ndarray):
                self.mean.append(word)
            elif word in wv.vocab:
                self.mean.append(wv.syn0norm[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)

        if not self.mean:
            print("POSSIBLE ERROR IN WORD2VECHELPER: cannot compute similarity with no input %s", words)
            return np.zeros(wv.vector_size,)

        self.mean = gensim.matutils.unitvec(np.array(self.mean).mean(axis=0)).astype(np.float32)
        return self.mean

    def  word_averaging_list(self, wv, text_list):
        return np.vstack([self.word_averaging(wv, post) for post in text_list])

    def reduce_dimensions(model, X_data):
        # num_dimensions = 2  # final num dimensions (2D, 3D, etc)

        vectors = [] # positions in vector space
        new_element = []
        # labels = [] # keep track of words to label our data again later
        for element in X_data:
            print("element: ", element)
            for word in element:
                new_element.append(model.wv[word])
                # labels.append(word)
            vectors.append(new_element)

        # convert both lists into numpy vectors for reduction
        vectors = np.asarray(vectors)
        # labels = np.asarray(labels)

        # reduce using t-SNE
        # vectors = np.asarray(vectors)
        # tsne = TSNE(n_components=num_dimensions, random_state=0)
        # vectors = tsne.fit_transform(vectors)

        x_vals = [v[0] for v in vectors]
        # y_vals = [v[1] for v in vectors]
        return x_vals #, y_vals, labels

#########################################################################################################################
# word2vec tools:

def get_fasttext_word2vec(data_frame_path):
    # model: skipgram(based on nearby word of Word),cbow(based on context,bag-words contained in a fixed windows around Word)
    # epoch: 5-50 (how many times it loops over your data)
    # lr: 0.1-1 (larger it is, faster it converges to a solution)
    # minn: 3 (all substrings contained in a word between minn and maxn)
    # maxn: 6 (0 to avoid subwords)
    # dim: 100-300 (size of the vectors - larger it is, more information it has and needs)
    # thread: default 12 (threads of the job)
    return fasttext.train_unsupervised(input=data_frame_path, model='cbow', dim=300, minn=2, maxn=5, epoch=5, lr=0.2, thread=12) #dim 100 200 300, model cbow skipgram, minn

def save_fasttext_word2vec(model, bin_path):
    model.save_model(bin_path)

def load_fasttext_word2vec(bin_path):
    fasttext.load_model(bin_path)

def save_fasttext_word2vec_vectors(model_w2v, data_frame, root_location):
    window_length = 300
    n_features = model_w2v.get_dimension()
    vectors = th.get_vectors_from_dataframe(model_w2v, data_frame, window_length, n_features)

    np.savetxt(fh.link_paths(root_location, 'data_vectors.csv'), vectors.reshape((3,-1)), delimiter=',', fmt='%s', header='vectors with shape:'+str(vectors.shape))

def load_fasttext_word2vec_vectors(root_location):
    file = open(fh.link_paths(root_location, 'data_vectors.csv'))
    header = file.readline().split(':')[1][1:-2].split(',')
    tuple_ = (int(header[0]), int(header[1]), int(header[2]))

    vectors = np.loadtxt(root_location + 'data_vectors.csv', delimiter=',')

    return vectors.reshape(tuple_)

#########################################################################################################################
# to be reviewed tools:

# they work almost in the same way as tfidfvectorizer works! the difference lies in the sklearn own tokenization that we do not use anyway
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        tfidf = ch.get_tfidf_vectorizer(X)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

#########################################################################################################################
# useless tools:

def get_word2vec_vocabulary(words):
    word2vec = Word2Vec(words, min_count=1)
    return word2vec.wv.vocab

def handle_bool(row, classes):
    list_ = list(filter(lambda class_ : class_ not in classes, row[1]))
    return True if list_ else False

def class_sentences(train_test_set, classes):
    flag = list(map(lambda row : handle_bool(row, classes), train_test_set.iterrows()))[0]
    if flag:
        yield row[0]

# def class_sentences(train_test_set, classes):
#     for index, row in train_test_set.iterrows():
#         flag = True
#         for class_ in row[1]:
#             if class_ not in classes:
#                 flag = False
#         if flag:
#             yield row[0]

def docprob(docs, mods):
    # score() takes a list [s] of sentences here; could also be a sentence generator
    sentlist = [s for d in docs for s in d]
    # the log likelihood of each sentence in this review under each w2v representation
    llhd = np.array([m.score(sentlist, len(sentlist)) for m in mods])
    # now exponentiate to get likelihoods,
    lhd = np.exp(llhd - llhd.max(axis=0)) # subtract row max to avoid numeric overload
    # normalize across models (stars) to get sentence-star probabilities
    prob = pd.DataFrame((lhd/lhd.sum(axis=0)).transpose())
    # and finally average the sentence probabilities to get the review probability
    prob["doc"] = [i for i,d in enumerate(docs) for s in d]
    prob = prob.groupby("doc").mean()
    return prob

def test_docprob(docs, mods):
    # score() takes a list [s] of sentences here; could also be a sentence generator
    sentlist = [s for s in docs]
    # the log likelihood of each sentence in this review under each w2v representation
    llhd = np.array([m.score(sentlist, len(sentlist)) for m in mods])
    # now exponentiate to get likelihoods,
    lhd = np.exp(llhd - llhd.max(axis=0)) # subtract row max to avoid numeric overload
    # normalize across models (stars) to get sentence-star probabilities
    prob = pd.DataFrame((lhd/lhd.sum(axis=0)).transpose())
    # and finally average the sentence probabilities to get the review probability
    prob["doc"] = [i for i, d in enumerate(docs)]
    prob = prob.groupby("doc").mean()
    return prob

def weird_application_word2vec(data_frame):
    data_frame['text'] = data_frame.apply(lambda row : th.tokenize_complex_text(row['text']), axis=1)

    ################################################# classification: from text to sparse binary matrix [[0, 1, 0],[1, 0, 1]]
    temp_classification, classes, n_classes = ch.apply_multilabel_binarizer(data_frame)

    train, test = ch.get_train_test_from_dataframe(data_frame)

    ## create a w2v learner
    model = Word2Vec(
        workers=multiprocessing.cpu_count(), # use your cores
        iter=3, # iter = sweeps of SGD through the data; more is better
        hs=1, negative=0 # we only have scoring for the hierarchical softmax setup
        )
    vocabulary = model.build_vocab(class_sentences(train, classes))

    number_patents_per_classcode = []
    class_model = list(map(lambda i : deepcopy(model), classes))
    for index, i in enumerate(classes):
        slist = list(class_sentences(train, [i]))
        vocab = class_model[index].build_vocab(slist)
        number_patents_per_classcode.append([i, "classification (", len(slist), ")"])
        class_model[index].train(slist, total_examples=len(slist), epochs=model.epochs)

    print(number_patents_per_classcode)

    # model.save("word2vec.model")
    # Word2Vec.load("word2vec.model")

    # print(model.score(test))
    # sentences = list of tokens

    # get the probs (note we give docprob a list of lists of words, plus the models)
    # probs = test_docprob([row[0] for index, row in test.iterrows()], class_model)
    probs = test_docprob(list(map(lambda index, row : row[0], test.iterrows())), class_model)
    # print("probs ", probs)
    probpos = pd.DataFrame({"out-of-sample prob positive":probs[[3,4]].sum(axis=1),
                            "true classifications":[next(iter(row[1])) for index, row in test.iterrows()]}) # problem with multiclass because it does not match with the index length - so i get only the first
    probpos.boxplot("out-of-sample prob positive", by="true classifications", figsize=(12,5))

