import numpy
import os
import matplotlib.pyplot as plt
import re
import scipy.linalg.interpolative as sli
from IPython.lib.security import passwd
from numpy import linalg
from sklearn.utils.extmath import randomized_svd
from collections import Mapping, defaultdict
from itertools import izip
import MySQLdb
from enum import Enum
import sys

from scipy.linalg.lapack import get_lapack_funcs

try:
    from scipy.linalg.basic import triu
except ImportError:
    from scipy.linalg.special_matrices import triu


def svd_factorization(R, P, Q, K, steps=3000, alpha=0.002, beta=0.02):
    Q = Q.T
    min_err = sys.maxint
    for step in xrange(steps):
        for i, doc in enumerate(R):
            for j in xrange(len(doc)):
                #if doc[j] > 0:
                    eij = doc[j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        e = calculate_error(R, P, Q, K)
        min_err = min(e, min_err)
        if step % 100 == 0:
            print "Error: " + str(e)
            print step

        #P, _ = qr_destroy([P])
        #Q, _ = qr_destroy([Q.T])
        #Q = Q.T

    return P, Q.T

def qr_destroy(la):
    """
    Return QR decomposition of `la[0]`. Content of `la` gets destroyed in the process.

    Using this function should be less memory intense than calling `scipy.linalg.qr(la[0])`,
    because the memory used in `la[0]` is reclaimed earlier.
    """
    a = numpy.asfortranarray(la[0])
    del la[0], la # now `a` is the only reference to the input matrix
    m, n = a.shape
    # perform q, r = QR(a); code hacked out of scipy.linalg.qr
    geqrf, = get_lapack_funcs(('geqrf',), (a,))
    qr, tau, work, info = geqrf(a, lwork=-1, overwrite_a=True)
    qr, tau, work, info = geqrf(a, lwork=work[0], overwrite_a=True)
    del a # free up mem
    assert info >= 0
    r = triu(qr[:n, :n])
    if m < n: # rare case, #features < #topics
        qr = qr[:, :m] # retains fortran order
    gorgqr, = get_lapack_funcs(('orgqr',), (qr,))
    q, work, info = gorgqr(qr, tau, lwork=-1, overwrite_a=True)
    q, work, info = gorgqr(qr, tau, lwork=work[0], overwrite_a=True)
    assert info >= 0, "qr failed"
    assert q.flags.f_contiguous
    return q, r


def calculate_error(R, P, Q, K, beta=0.02):
    e = 0
    for i, doc in enumerate(R):
        for j in xrange(len(doc)):
            #if doc[j] > 0:
                e = e + pow(doc[j] - numpy.dot(P[i,:],Q[:,j]), 2)
                for k in xrange(K):
                    e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
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
        #"Trees are special cases of graphs so whatever works for a general graph works for trees",
        #"In mathematics, and more specifically in graph theory, a tree"
        #" is an undirected graph in which any two vertices are connected by exactly one path, paths"]


def iter_documents(top_dir):
    for root, dirs, files in os.walk(top_dir):
        for file_name in filter(lambda x: x.endswith('.txt'), files):
            document = open(os.path.join(root, file_name)).read()
            yield tokenize(document, lower=True)


def iter_documents_hci_graphs():
    for doc in DOCS_HCI_GRAPHS:
        yield tokenize(doc, lower=True)

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
    for doc in DOCS_TWO_GROUPS:
        yield tokenize(doc, lower=True)


def iter_documents_db_table(db, sql):
    cursor = db.cursor()
    cursor.execute(sql)

    for paper_row in cursor:
        txt = paper_row[0]
        yield tokenize(txt, lower=True)


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
        self.token_to_id = dict()  # token -> token id
        self.id_to_token = dict()  # token id -> token
        self.doc_freqs = defaultdict(int)  # token id -> the number of documents this token appears in

        if documents is not None:
            self.add_documents(documents)

    def add_documents(self, documents):
        for document in documents:
            self.doc_to_bag_of_words(document, allow_update=True)

    def doc_to_bag_of_words(self, document, allow_update=False):
        token_freq_map = defaultdict(int)
        for token in document:
            token_freq_map[token] += 1

        if allow_update:
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


class Corpus(object):
    def __init__(self, doc_iterator, stop_words):
        self.doc_iterator = doc_iterator
        self.stop_list = stop_words
        self.dictionary = Dictionary(doc_iterator())
        stop_ids = [self.dictionary.token_to_id[stop_word] for stop_word in self.stop_list
                    if stop_word in self.dictionary.token_to_id]
        once_ids = [token_id for token_id, doc_freq in self.dictionary.doc_freqs.iteritems() if doc_freq == 1]
        self.dictionary.filter_tokens(stop_ids + once_ids)

    def __iter__(self):
        for tokens in self.doc_iterator():
            #yield self.dictionary.doc_to_bag_of_words(tokens)
            yield doc_to_vec(len(self.dictionary.items()), self.dictionary.doc_to_bag_of_words(tokens))


def doc_to_vec(term_count, doc_as_bow):
    doc_vector = [0] * term_count
    for (word, freq) in doc_as_bow.iteritems():
        doc_vector[word] = freq

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

        corpus = Corpus(lambda: iter_documents_db_table(db, sql), stop_words=set())

    return doc_count, corpus


def select_factorization_algorithm(factorization_algo, corpus=None, doc_count=0, num_features=2):
    U, S, V = None, None, None
    if factorization_algo == FactorizationAlgorithm.linear_svd:
        A = []
        for doc in corpus:
            A.append(doc)
        U, S, V = linalg.svd(A)
    elif factorization_algo == FactorizationAlgorithm.randomized_svd:
        A = []
        for doc in corpus:
            A.append(doc)
        U, S, V = randomized_svd(numpy.array(A), n_components=num_features)
    elif factorization_algo == FactorizationAlgorithm.gradient_descent:
        N = doc_count
        M = len(corpus.dictionary.items())
        K = num_features

        P = numpy.random.uniform(low=-0.01, high=0.01, size=(N,K))
        Q = numpy.random.uniform(low=-0.01, high=0.01, size=(M,K))

        #P = numpy.full((N, K), 0.1)
        #Q = numpy.full((M, K), 0.1)

        U, V = svd_factorization(corpus, P, Q, K)

    return U, S, V


def run_factorization():
    doc_count, corpus = select_corpus(DataSourceType.local_hci_graphs)
    print corpus.dictionary.items()

    PP, S, QQ = select_factorization_algorithm(FactorizationAlgorithm.gradient_descent,
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
    #print QQ


run_factorization()


