import numpy
import os
import matplotlib.pyplot as plt
import re
from numpy import linalg
from sklearn.utils.extmath import randomized_svd
from collections import Mapping, defaultdict
from itertools import izip
import MySQLdb
from enum import Enum
import sys
import math
import datetime

from scipy.linalg.lapack import get_lapack_funcs

try:
    from scipy.linalg.basic import triu
except ImportError:
    from scipy.linalg.special_matrices import triu


class Document(object):
    def __init__(self, id, tokens):
        self.id = id
        self.tokens = tokens


class SVDEngine(object):
    def __init__(self, num_docs, num_words, num_features=2):
        self.min_iter = 1000
        self.max_iter = 3000
        self.num_docs = num_docs
        self.num_words = num_words
        self.num_features = num_features
        self.feature_init_low = -0.01
        self.feature_init_high = 0.01
        self.svd_u = None
        self.svd_v = None
        self.min_improvement = 0.0001
        self.learning_rate = 0.002
        self.regul_factor = 0.02

    def predict_value(self, document_id, word_id):
        return numpy.dot(self.svd_u[document_id, :], self.svd_v[:, word_id])

    def feature_training(self, documents_iterator):
        rmse = 2.0
        rmse_last = 2.0

        self.svd_u = numpy.random.uniform(low=-0.01, high=0.01, size=(self.num_docs, self.num_features))
        self.svd_v = numpy.random.uniform(low=-0.01, high=0.01, size=(self.num_words, self.num_features))
        self.svd_v = self.svd_v.T

        epoch = 0

        #while (epoch < self.min_iter or rmse_last - rmse >= self.min_improvement) and epoch < self.max_iter:
        while epoch < self.max_iter:
            squared_error = 0.0
            rmse_last = rmse
            num_values = 0

            for document_id, word_id, value in documents_iterator:
                num_values += 1

                predicted_value = self.predict_value(document_id, word_id)
                error = 1.0 * value - predicted_value
                squared_error += error * error

                for feature_id in numpy.arange(self.num_features):
                    document_feature = self.svd_u[document_id, feature_id]
                    word_feature = self.svd_v[feature_id, word_id]

                    self.svd_u[document_id, feature_id] += \
                        self.learning_rate * (error * word_feature - self.regul_factor * document_feature)
                    self.svd_v[feature_id, word_id] += \
                        self.learning_rate * (error * document_feature - self.regul_factor * word_feature)

            rmse = numpy.sqrt(squared_error / num_values)

            if epoch % 100 == 0:
                print epoch

            epoch += 1

        print "Last epoch: " + str(epoch)
        print "Last improvement: " + str(rmse_last - rmse)

    def save_complete_model_compressed(self, file_name):
        numpy.savez_compressed(file_name + '_model.npz', svd_u=self.svd_u, svd_v=self.svd_v)

    def load_complete_model_compressed(self, file_name):
        model = numpy.load(file_name + '_model.npz')
        self.svd_u = model['svd_u']
        self.svd_v = model['svd_v']

    def save_document_profile_batch(self, document_ids):
        document_profile_batch = []
        for document_id in document_ids:
            document_profile = {
                'id': document_id,
                'features': self.svd_u[document_id, :],
                'snapshot_time': datetime.datetime.utcnow()
            }

            document_profile_batch.append(document_profile)

        self.profile_db.document_profiles.insert(document_profile_batch)

    def save_word_profile_batch(self, word_ids):
        word_profile_batch = []
        for word_id in word_ids:
            word_profile = {
                'id': word_id,
                'features': self.svd_v[:, word_id],
                'snapshot_time': datetime.datetime.utcnow()
            }

            word_profile_batch.append(word_profile)

        self.profile_db.word_profiles.insert(word_profile_batch)

    def load_document_profiles(self):
        pass

    def load_word_profiles(self):
        pass

def svd_factorization(R, P, Q, K, steps=3000, alpha=0.002, beta=0.02):
    Q = Q.T
    min_err = sys.maxint
    for step in xrange(steps):
        for i, j, value in R:
            eij = value - numpy.dot(P[i, :], Q[:, j])
        #for i, doc in enumerate(R):
        #    for j in xrange(len(doc)):
        #        eij = doc[j] - numpy.dot(P[i, :], Q[:, j])
            for k in xrange(K):
                P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        e = calculate_error(R, P, Q, K)
        min_err = min(e, min_err)
        if step % 100 == 0:
            print "Error: " + str(e)
            print step

            # P, _ = qr_destroy([P])
            # Q, _ = qr_destroy([Q.T])
            # Q = Q.T

    return P, Q.T

def calculate_error(R, P, Q, K, beta=0.02):
    e = 0
    for i, j, value in R:
    #for i, doc in enumerate(R):
    #    for j in xrange(len(doc)):
        e = e + pow(value - numpy.dot(P[i, :], Q[:, j]), 2)
        for k in xrange(K):
            e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
    return e


def tokenize(document, lower=False):
    tokens = re.split('\W+', document)

    return [token.lower() if lower else token for token in tokens]


DOCS_HCI_GRAPHS = ["Human machine interface for lab abc computer applications",
                   "A survey of user opinion of computer system response time",
                   "The EPS user interface management system",
                   "System and human system engineering testing of EPS",
                   "Relation of user perceived response time to error measurement",
                   "The generation of random binary unordered trees",
                   "The intersection graph of paths in trees",
                   "Graph minors IV Widths of trees and well quasi ordering",
                   "Graph minors A survey"]


# "Trees are special cases of graphs so whatever works for a general graph works for trees",
# "In mathematics, and more specifically in graph theory, a tree"
# " is an undirected graph in which any two vertices are connected by exactly one path, paths"]


def iter_documents(top_dir):
    document_id = 0
    for root, dirs, files in os.walk(top_dir):
        for file_name in filter(lambda x: x.endswith('.txt'), files):
            document = open(os.path.join(root, file_name)).read()
            yield Document(id=document_id, tokens=tokenize(document, lower=True))
            document_id += 1


def iter_documents_hci_graphs():
    document_id = 0
    for doc in DOCS_HCI_GRAPHS:
        yield Document(id=document_id, tokens=tokenize(doc, lower=True))
        document_id += 1


DOCS_TWO_GROUPS = [
    "A B C",
    "A B C C C F F B B",
    "A A A A B B B B C C C C",
    "C C C C C A A A A A B B B B B",
    "D E D E D E D E",
    "E D E D E D E D E D",
    "D D E E"
]


def iter_documents_two_explicit_groups():
    document_id = 0
    for doc in DOCS_TWO_GROUPS:
        yield Document(id=document_id, tokens=tokenize(doc, lower=True))
        document_id += 1


def iter_documents_db_table(db, sql):
    cursor = db.cursor()
    cursor.execute(sql)

    for paper_row in cursor:
        document_id = paper_row[0]
        txt = paper_row[1]
        yield Document(id=document_id, tokens=tokenize(txt, lower=True))


class Dictionary(Mapping):
    def __len__(self):
        return len(self.token_to_id)

    def __getitem__(self, key):
        if len(self.id_to_token) != len(self.token_to_id):
            self.id_to_token = dict((v, k) for k, v in self.token_to_id.iteritems())
        return self.id_to_token[key]

    def keys(self):
        return list(self.token_to_id.values())

    def __iter__(self):
        return iter(self.keys())

    def __init__(self, documents=None):
        self.num_docs = 0
        self.token_to_id = dict()  # token -> token id
        self.id_to_token = dict()  # token id -> token
        self.doc_freqs = defaultdict(int)  # token id -> the number of documents this token appears in
        self.doc_db_id_to_local_id = dict()  # document id in the db -> local document id

        if documents is not None:
            self.add_documents(documents)

    def add_documents(self, documents):
        for document in documents:
            self.doc_to_bag_of_words(document, allow_update=True)

    def doc_to_bag_of_words(self, document, allow_update=False):
        token_freq_map = defaultdict(int)
        for token in document.tokens:
            token_freq_map[token] += 1

        if allow_update:
            self.num_docs += 1
            self.doc_db_id_to_local_id[document.id] = len(self.doc_db_id_to_local_id)
            for token, freq in token_freq_map.iteritems():
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)
                self.doc_freqs[self.token_to_id[token]] += 1

        bag_of_words = dict((self.token_to_id[token], freq)
                            for token, freq in token_freq_map.iteritems()
                            if token in self.token_to_id)

        return bag_of_words

    def filter_tokens(self, bad_ids):
        bad_ids = set(bad_ids)
        self.token_to_id = dict((token, token_id)
                                for token, token_id in self.token_to_id.iteritems()
                                if token_id not in bad_ids)
        self.doc_freqs = dict((token_id, freq)
                              for token_id, freq in self.doc_freqs.iteritems()
                              if token_id not in bad_ids)

        self.compactify()

    def compactify(self):
        idmap = dict(izip(self.token_to_id.itervalues(), xrange(len(self.token_to_id))))

        self.token_to_id = dict((token, idmap[token_id]) for token, token_id in self.token_to_id.iteritems())
        self.doc_freqs = dict((idmap[token_id], freq) for token_id, freq in self.doc_freqs.iteritems())
        self.id_to_token = dict()


class TfidfModel(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __getitem__(self, bag_of_words):
        tf_sum = sum([token_freq for (token_id, token_freq) in bag_of_words.iteritems()])
        return dict((token_id,
                     self.tf(token_freq, tf_sum) * self.idf(self.dictionary.doc_freqs[token_id],
                                                            self.dictionary.num_docs))
                    for (token_id, token_freq) in bag_of_words.iteritems())

    @staticmethod
    def tf(token_freq, tf_sum):
        return 1.0 * token_freq / tf_sum

    @staticmethod
    def idf(doc_freq, total_docs, log_base=2.0):
        return math.log(1.0 * total_docs / doc_freq, log_base)


class Corpus(object):
    def __init__(self, document_iterator, stop_words):
        self.document_iterator = document_iterator
        self.stop_list = stop_words
        self.dictionary = Dictionary(document_iterator())
        self.tfidf_model = TfidfModel(self.dictionary)
        stop_ids = [self.dictionary.token_to_id[stop_word] for stop_word in self.stop_list
                    if stop_word in self.dictionary.token_to_id]
        once_ids = [token_id for token_id, doc_freq in self.dictionary.doc_freqs.iteritems() if doc_freq == 1]
        self.dictionary.filter_tokens(stop_ids + once_ids)

    def __iter__(self):
        for document in self.document_iterator():
            # yield self.dictionary.doc_to_bag_of_words(tokens)
            #yield doc_to_vec(len(self.dictionary.items()), self.dictionary.doc_to_bag_of_words(document))
            #yield doc_to_vec(len(self.dictionary.items()),
            #                 self.tfidf_model[self.dictionary.doc_to_bag_of_words(document)])
            converted_document = self.dictionary.doc_to_bag_of_words(document)
            converted_document = self.tfidf_model[converted_document]

            word_count = len(self.dictionary.items())
            for word_id in xrange(word_count):
                if word_id in converted_document:
                    yield document.id, word_id, converted_document[word_id]
                else:
                    yield document.id, word_id, 0


            #for (word_id, value) in converted_document.iteritems():
            #    yield document.id, word_id, value


def doc_to_vec(term_count, term_freqs):
    doc_vector = [0] * term_count
    for (word_id, freq) in term_freqs.iteritems():
        doc_vector[word_id] = freq

    return doc_vector


class DataSourceType(Enum):
    local_nasa_elections = 1
    local_explicit_groups = 2
    local_hci_graphs = 3
    db_explicit_groups = 4


class FactorizationAlgorithm(Enum):
    gradient_descent = 1
    linear_svd = 2
    randomized_svd = 3
    gradient_descent_engine = 4

def select_corpus(data_source_type):
    doc_count = 0
    corpus = None
    if data_source_type == DataSourceType.local_nasa_elections:
        doc_count = 9
        corpus = Corpus(lambda: iter_documents('test_documents'),
                        stop_words=set('for a of the and to in'.split()))
    elif data_source_type == DataSourceType.local_explicit_groups:
        doc_count = 7
        corpus = Corpus(lambda: iter_documents_two_explicit_groups(), stop_words=set())
    elif data_source_type == DataSourceType.local_hci_graphs:
        doc_count = 9
        corpus = Corpus(lambda: iter_documents_hci_graphs(),
                        stop_words=set('for a of the and to in'.split()))
    elif data_source_type == DataSourceType.db_explicit_groups:
        db = MySQLdb.connect(host='localhost', user='sa',
                             passwd='1qaz@WSX', db='test')

        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM papers WHERE groupId = 1")
        doc_count = cursor.fetchone()[0]
        sql = "SELECT * FROM papers WHERE groupId = 1"
        # doc_count = cursor.rowcount

        corpus = Corpus(lambda: iter_documents_db_table(db, sql), stop_words=set())

    return doc_count, corpus


def select_factorization_algorithm(factorization_algo, corpus=None, doc_count=0, num_features=2):
    U, S, V = None, None, None
    if factorization_algo == FactorizationAlgorithm.linear_svd:
        A = [[]]
        for doc_id, word_id, value in corpus:
            if len(A) < doc_id + 1:
                A.append([])
            A[doc_id].append(value)
        U, S, V = linalg.svd(A)
    elif factorization_algo == FactorizationAlgorithm.randomized_svd:
        A = [[]]
        for doc_id, word_id, value in corpus:
            if len(A) < doc_id + 1:
                A.append([])
            A[doc_id].append(value)
        U, S, V = randomized_svd(numpy.array(A), n_components=num_features)
    elif factorization_algo == FactorizationAlgorithm.gradient_descent:
        N = doc_count
        M = len(corpus.dictionary.items())
        K = num_features

        P = numpy.random.uniform(low=-0.01, high=0.01, size=(N, K))
        Q = numpy.random.uniform(low=-0.01, high=0.01, size=(M, K))

        # P = numpy.full((N, K), 0.1)
        # Q = numpy.full((M, K), 0.1)

        U, V = svd_factorization(corpus, P, Q, K)
    elif factorization_algo == FactorizationAlgorithm.gradient_descent_engine:
        svd_engine = SVDEngine(num_docs=doc_count, num_words=len(corpus.dictionary.items()), num_features=2)
        svd_engine.feature_training(corpus)
        U, V = svd_engine.svd_u, svd_engine.svd_v

    return U, S, V


def run_factorization():
    doc_count, corpus = select_corpus(DataSourceType.local_explicit_groups)
    print corpus.dictionary.items()

    PP, S, QQ = select_factorization_algorithm(FactorizationAlgorithm.gradient_descent_engine,
                                               corpus=corpus, doc_count=doc_count, num_features=2)
    draw_plot(PP, QQ, doc_count)


def draw_plot(PP, QQ, doc_count):
    print PP
    x = PP[:, 0]
    print "X: " + str(x.tolist())
    y = PP[:, 1]
    print "Y: " + str(y.tolist())
    print "Dot product: " + str(numpy.dot(x.tolist(), y.tolist()))
    n = ['d' + str(i) for i in range(1, doc_count + 1)]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()
    # print QQ


run_factorization()
