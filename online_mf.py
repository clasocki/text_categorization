import numpy
import os
import matplotlib.pyplot as plt
from collections import Mapping, defaultdict
from itertools import izip
import re

def svd_factorization(R, P, Q, K, steps=5000, alpha=0.002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        if step % 100 == 0:
            print step
        for i, doc in enumerate(R):
            for j in xrange(len(doc)):
                #if R[i][j] > 0:
                    eij = doc[j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

    return P, Q.T


def tokenize(document, lower=False):
    tokens = re.split('\W+', document)

    return [token.lower() if lower else token for token in tokens]

DOCS = ["Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
        "Trees are special cases of graphs so whatever works for a general graph works for trees",
        "In mathematics, and more specifically in graph theory, a tree"
        " is an undirected graph in which any two vertices are connected by exactly one path, paths"]


def iter_documents(top_dir):
    for doc in DOCS:
        yield tokenize(doc, lower=True)


#def iter_documents(top_dir):
#    for root, dirs, files in os.walk(top_dir):
#        for file_name in filter(lambda x: x.endswith('.txt'), files):
#            document = open(os.path.join(root, file_name)).read()
#            yield tokenize(document, lower=True)


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


class TextSubdirectoriesCorpus(object):
    def __init__(self, top_dir):
        self.top_dir = top_dir
        self.stop_list = set('for a of the and to in'.split())
        self.dictionary = Dictionary(iter_documents(top_dir))
        stop_ids = [self.dictionary.token_to_id[stop_word] for stop_word in self.stop_list
                    if stop_word in self.dictionary.token_to_id]
        once_ids = [token_id for token_id, doc_freq in self.dictionary.doc_freqs.iteritems() if doc_freq == 1]
        self.dictionary.filter_tokens(stop_ids + once_ids)

    def __iter__(self):
        for tokens in iter_documents(self.top_dir):
            yield doc_to_vec(len(self.dictionary.items()), self.dictionary.doc_to_bag_of_words(tokens))


def doc_to_vec(term_count, doc_as_bow):
    doc_vector = [0] * term_count
    for (word, freq) in doc_as_bow.iteritems():
        doc_vector[word] = freq

    return doc_vector


def run_factorization():
    doc_count = 9
    corpus = TextSubdirectoriesCorpus('test_documents')

    N = doc_count
    M = len(corpus.dictionary.items())
    K = 2

    P = numpy.random.uniform(low=-0.01, high=0.01, size=(N,K))
    Q = numpy.random.uniform(low=-0.01, high=0.01, size=(M,K))

    PP, QQ = svd_factorization(corpus, P, Q, K)
    print PP
    x = PP[:,0]
    y = PP[:, 1]
    n = ['d' + str(i) for i in range(1, doc_count + 1)]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()
    print QQ


run_factorization()


