try:
    import numpy
except:
    print "This implementation requires the numpy module."
    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                #if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        if calculate_error(R, P, Q, K, beta):
            break

    return P, Q.T


def calculate_error(R, P, Q, K, beta=0.02):
    e = 0
    for i in xrange(len(R)):
        for j in xrange(len(R[i])):
            #if R[i][j] > 0:
                e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                for k in xrange(K):
                    e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
    return e < 0.001

###############################################################################
corpus_bow = [
    [(5, 1), (8, 1), (10, 1)],
    [(2, 1), (5, 1), (6, 1), (7, 1), (9, 1), (11, 1)],
    [(2, 1), (4, 1), (7, 1), (10, 1)],
    [(2, 2), (4, 1), (8, 1)],
    [(7, 1), (9, 1), (11, 1)],
    [(3, 1)],
    [(1, 1), (3, 1)],
    [(0, 1), (1, 1), (3, 1)],
    [(0, 1), (1, 1), (6, 1)],
]

words = 12
docs = len(corpus_bow)
R = []

for doc in corpus_bow:
    doc_vec = [0] * words
    for (word, freq) in doc:
        doc_vec[word] = freq
    R.append(doc_vec)

R = numpy.array(R)

N = len(R)
M = len(R[0])
K = 2

P = numpy.random.uniform(low=-0.01, high=0.01, size=(N,K))
Q = numpy.random.uniform(low=-0.01, high=0.01, size=(M,K))

nP, nQ = matrix_factorization(R, P, Q, K)
print nP
print nQ