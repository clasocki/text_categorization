import numpy
import matplotlib.pyplot as plt
from numpy import linalg
from sklearn.utils.extmath import randomized_svd
import MySQLdb
import sys
from helpers import DataSourceType, Document, FactorizationAlgorithm
from corpus import Corpus
from svd_engine import SVDEngine
import iter_documents


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


def select_corpus(data_source_type):
    doc_count = 0
    corpus = None
    if data_source_type == DataSourceType.local_nasa_elections:
        doc_count = 9
        corpus = Corpus(lambda: iter_documents.iter_documents('test_documents'),
                        stop_words=set('for a of the and to in'.split()))
    elif data_source_type == DataSourceType.local_explicit_groups:
        doc_count = 7
        corpus = Corpus(lambda: iter_documents.iter_documents_two_explicit_groups(), stop_words=set())
    elif data_source_type == DataSourceType.local_hci_graphs:
        doc_count = 9
        corpus = Corpus(lambda: iter_documents.iter_documents_hci_graphs(),
                        stop_words=set('for a of the and to in'.split()))
    elif data_source_type == DataSourceType.db_explicit_groups:
        db = MySQLdb.connect(host='localhost', user='sa',
                             passwd='1qaz@WSX', db='test')

        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM papers WHERE groupId = 1")
        doc_count = cursor.fetchone()[0]
        sql = "SELECT * FROM papers WHERE groupId = 1"
        # doc_count = cursor.rowcount

        corpus = Corpus(lambda: iter_documents.iter_documents_db_table(db, sql), stop_words=set())

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
    doc_count, corpus = select_corpus(DataSourceType.local_hci_graphs)
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
