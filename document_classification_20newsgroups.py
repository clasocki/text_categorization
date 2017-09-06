"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time
(normalized) of each classifier.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause


import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
#import matplotlib
#matplotlib.use('GTK')
#matplotlib.use('Qt5Agg')

#import matplotlib.pyplot as plt
#import seaborn as sn

#plt.switch_backend('agg')

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

import gensim_tests
from semantic_model import SemanticModel, DocumentIterator, tokenize1
import pandas as pd
from nltk.corpus import reuters
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
import itertools

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

class Tokenizer(object):
    def __call__(self, doc):
        return tokenize1(doc).split()
# Benchmark classifiers
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
"""
###############################################################################
def benchmark(clf, clf_name, X_train, y_train, X_test, y_test, multilabel):
    #print('_' * 80)
    #print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    print pred
    #print clf.predict_proba(X_test).tolist()
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = None
    if not multilabel:
        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)
    else:
        precision = metrics.precision_score(y_test, pred, average='micro')
        recall = metrics.recall_score(y_test, pred, average='micro')
        f1 = 2.0 * precision * recall / (precision + recall)
        score = { 'precision': precision,
                  'recall': recall,
                  'f1' : f1
                }
    
        print("precision: %0.3f recall: %0.3f f1: %0.3f" % (precision, recall, f1))

    """ 
    if hasattr(clf, 'coef_'):
        #print("dimensionality: %d" % clf.coef_.shape[1])
        #print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        #print()
    
    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))
    """
    """
    cnf_matrix = metrics.confusion_matrix(y_test, pred)
    print(clf.classes_.tolist())
    print(cnf_matrix.tolist())
    
    df_cm = pd.DataFrame(cnf_matrix, index=clf.classes_, columns=clf.classes_)
    sn.heatmap(df_cm, annot=True)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation=0) 
    plt.savefig('conf_matrix' + clf_name.replace(" ", "") + '.png')
    """
    #print()
    #print metrics.classification_report(y_test, pred, target_names=clf.classes_)

    return clf_name, score, pred

class Tokenizer(object):
    def __call__(self, doc):
        return tokenize1(doc).split()

def svd(num_features):
    vectorizer = TfidfVectorizer(tokenizer=Tokenizer(), stop_words='english', min_df=1, max_df=0.5,
                                 use_idf=True, smooth_idf=True, sublinear_tf=True)
    svd_model = TruncatedSVD(n_components=num_features, algorithm='randomized',
                             n_iter=10, random_state=42)

    svd_transformer = Pipeline([('tfidf', vectorizer), 
                                ('svd', svd_model)])

    return svd_transformer
    #svd_matrix = svd_transformer.fit_transform(document_corpus)

    #return svd_matrix

def testClassifiers(X_train, y_train, X_test, y_test, multilabel):
    mlb = None
    if multilabel:
        mlb = MultiLabelBinarizer()
        mlb = mlb.fit(y_train)
        y_train = mlb.transform(y_train)
        y_test = mlb.transform(y_test)
        
    results = []
    for clf, clf_name in (
            #(RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            #(Perceptron(n_iter=50), "Perceptron"),
            #(PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            #(KNeighborsClassifier(n_neighbors=10), "kNN"),
            #(KNeighborsClassifier(n_neighbors=8, algorithm='brute', metric='cosine'), "kNN cosine"),
            #(RandomForestClassifier(n_estimators=100), "Random forest"),
            #(LinearSVC(penalty="l2", dual=False, tol=1e-3), "Linear SVC [l2]"),
            #(LinearSVC(penalty="l1", dual=False, tol=1e-3), "Linear SVC [l1]"),
            #(SGDClassifier(alpha=.0001, n_iter=50, penalty="l2"), "SGD Classifier [l2]"),
            #(SGDClassifier(alpha=.0001, n_iter=50, penalty="l1"), "SGD Classifier [l1]"),
            (SGDClassifier(alpha=.001, n_iter=50, penalty="elasticnet"), "SGD Classifier [elasticnet]"),
            #(NearestCentroid(), "Nearest Centroid"), #not suitable for multilabel
            #(Pipeline([
            #    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
            #    ('classification', LinearSVC(penalty="l2"))]), "Linear SVC [l1 based features]")
            ):
        #print('=' * 80)
        #print(name)
        if multilabel:
            clf = OneVsRestClassifier(clf)
        results.append(benchmark(clf, clf_name, X_train, y_train, X_test, y_test, multilabel))

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in xrange(len(results[0]))]
    if multilabel:
        preds = results[2]
        for i, pred in enumerate(preds):
            preds[i] = mlb.inverse_transform(pred)

    return results

    """
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.savefig('classifier_comparison.pdf')
    """

def reutersSet():
    documents = reuters.fileids()
     
    train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                                     documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                                    documents))
      
    train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

    train_labels = [reuters.categories(doc_id) for doc_id in train_docs_id]
    test_labels = [reuters.categories(doc_id) for doc_id in test_docs_id]

    multilabel = True

    return train_docs, test_docs, train_labels, test_labels, multilabel

def newsgroupsSet():
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
        'rec.motorcycles',
        'sci.electronics',
        'sci.med',
        'talk.politics.guns',
        'rec.autos'
    ]

    categories = None

    #remove = ('headers', 'footers', 'quotes')
    remove = ()

    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42,
                                    remove=remove)

    data_test = fetch_20newsgroups(subset='test', categories=categories,
                                   shuffle=True, random_state=42,
                                   remove=remove)
    # order of labels in `target_names` can be different from `categories`
    target_names = np.asarray(data_train.target_names)

    # split a training set and a test set
    y_train, y_test = data_train.target, data_test.target

    multilabel = False

    return data_train.data, data_test.data, y_train, y_test, multilabel

if __name__ == "__main__":

    # parse commandline arguments
    op = OptionParser()
    op.add_option("--report",
                  action="store_true", dest="print_report",
                  help="Print a detailed classification report.")
    op.add_option("--chi2_select",
                  action="store", type="int", dest="select_chi2",
                  help="Select some number of features using a chi-squared test")
    op.add_option("--confusion_matrix",
                  action="store_true", dest="print_cm",
                  help="Print the confusion matrix.")
    op.add_option("--top10",
                  action="store_true", dest="print_top10",
                  help="Print ten most discriminative terms per class"
                       " for every classifier.")
    op.add_option("--all_categories",
                  action="store_true", dest="all_categories",
                  help="Whether to use all categories or not.")
    op.add_option("--use_hashing",
                  action="store_true",
                  help="Use a hashing vectorizer.")
    op.add_option("--n_features",
                  action="store", type=int, default=2 ** 16,
                  help="n_features when using the hashing vectorizer.")
    op.add_option("--filtered",
                  action="store_true",
                  help="Remove newsgroup information that is easily overfit: "
                       "headers, signatures, and quoting.")


    def is_interactive():
        return not hasattr(sys.modules['__main__'], '__file__')

    # work-around for Jupyter notebook and IPython console
    argv = [] if is_interactive() else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    print(__doc__)
    op.print_help()
    print()


    ###############################################################################
    # Load some categories from the training set
    
    train_docs, test_docs, y_train, y_test, multilabel = newsgroupsSet()
    #train_docs, test_docs, y_train, y_test, multilabel = reutersSet()

    #y_train, y_test = [[x] for x in target_names[data_train.target]], [[x] for x in target_names[data_test.target]]

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    #vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=0.00, max_df=0.5,
    #                                 stop_words='english')
    #X_train = vectorizer.fit_transform(train_docs)
    #svd_transformer = svd(300)
    #X_train = svd_transformer.fit_transform(train_docs)
    semantic_model = gensim_tests.SemanticModel.build(lambda: (text for text in train_docs), 400, no_below=0.001 * len(train_docs), no_above=0.33)
    print "Model building time: " + str(time() - t0)
    X_train = np.asarray([semantic_model.inferProfile(x) for x in train_docs])
    
    #document_iterator = DocumentIterator(doc_filter="published = 1 and learned_category is not null", 
    #                                     document_batch_size=5000, db_window_size=5000)

    #iter_semantic_model = SemanticModel.load('experiments/165__reuters_propValSet_triples_pos_rmse_num_f=300-learn_r=0.001-regul_f=0.01-zero_w=3.0-doc_low=-0.01-doc_high=0.01-word_low=-0.01-word_high=-0.01-decay=2.0-min_df=0.001-max_df=0.33,term_w=log_normalization-iter=80/semantic_model.snapshot', document_iterator=None, word_profiles_in_db=False)
    #X_train = np.asarray([iter_semantic_model.inferProfile(x, num_iters=30, learning_rate=0.001, regularization_factor=0.01) for x in train_docs])

    """
    query = "SELECT profile, learned_category FROM pap_papers_view WHERE published = 1 and profile is not null and learned_category is not null"
    X_train, y_train = [], []

    rowmapper = lambda row: (np.asarray([float(value) for value in row['profile'].split(',')]), row['learned_category'])
    with LocalDocumentGenerator(query, rowmapper) as labeled_documents:
            for profile, label in labeled_documents:
                    X_train.append(profile)
                    y_train.append(label)
    """
    #X_train = np.asarray(X_train)
    #y_test = np.asarray(data_test.target_names)[data_test.target]

    duration = time() - t0
    #print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    #print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    #X_test = vectorizer.transform(test_docs)
    #X_test = svd_transformer.transform(test_docs)
    X_test = np.asarray([semantic_model.inferProfile(x) for x in test_docs])
    #X_test = np.asarray([iter_semantic_model.inferProfile(x, num_iters=30, learning_rate=0.001, regularization_factor=0.01) for x in test_docs])
    #print (X_test)
    duration = time() - t0
    #print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    # mapping from integer feature name to original token string
    """
    if opts.use_hashing:
        feature_names = None
    else:
        feature_names = vectorizer.get_feature_names()

    if opts.select_chi2:
        print("Extracting %d best features by a chi-squared test" %
              opts.select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        if feature_names:
            # keep selected feature names
            feature_names = [feature_names[i] for i
                             in ch2.get_support(indices=True)]
        print("done in %fs" % (time() - t0))
        print()

    if feature_names:
        feature_names = np.asarray(feature_names)
    """

    def trim(s):
        """Trim string to fit on terminal (assuming 80-column display)"""
        return s if len(s) <= 80 else s[:77] + "..."

    
    clf_names, score, training_time, test_time = testClassifiers(X_train, y_train, X_test, y_test, multilabel)

    scores = dict()
    for i, clf_name in enumerate(clf_names):
        scores[clf_name] = { 0: score[i] }
    df = pd.DataFrame(scores)
    df.to_csv('doc_class_tests.csv')
    print(df)


