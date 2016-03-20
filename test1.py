from gensim import corpora, models
#import logging
#logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
#                    level=logging.INFO)


def print_list(list_of_lists, msg, tuples=True):
    ''' prints a list of lists one by row with a message
    tuples tells if the list is made of tuples
    :param tuples:
    :param msg:
    :param list_of_lists: '''
    if msg:
        print msg
    if not tuples:
        for listW in list_of_lists:
            print listW
    else:
        for listP in list_of_lists:
            for (i, v) in listP:
                print "(%s, %.3f)" % (i, v),
            print
    print


def print_dic(d, msg):
    ''' prints a dictionary, one item (key - value) per line '''
    if msg:
        print msg
    for k in sorted(d.keys()):
        if d[k]:
            print "%d - %s " % (k, d[k])
    print


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
NUM_TOP = 2                                 # Number of topics for LSI

# --------------- INITIAL DOCUMENTS -----------------------------------
print_list(DOCS[0:9], "INITIAL TEXTS:", False)
texts = [[w for w in doc.split()] for doc in DOCS]
first_texts = texts[0:9]

dic = corpora.HashDictionary(first_texts)
print_dic(dic, "HASH DICTIONARY:")

corpus = [dic.doc2bow(text) for text in first_texts]
print_list(corpus, "CORPUS BAG-OF-WORDS:", False)

tfidf = models.TfidfModel(corpus)
print_list(tfidf[corpus], "TF-IDF")

lsi = models.LsiModel(tfidf[corpus], num_topics=NUM_TOP, id2word=dic)
print_list(lsi[tfidf[corpus]], "LSI (from TF-IDF):")
lsiBOW = models.LsiModel(corpus, num_topics=NUM_TOP, id2word=dic)
print_list(lsiBOW[corpus], "LSI (from BOW):")

# ----------------- NEW DOCUMENT -----------------------------------
print_list(DOCS[9:11], "----- NEW TEXT: -----", False)
new_texts = texts[9:11]
dic.add_documents(new_texts)
print_dic(dic, "UPDATED DICTIONARY:")

corpus_new = [dic.doc2bow(text) for text in new_texts]
corpus.extend(corpus_new)
print_list(corpus, "UPDATED CORPUS BAG-OF-WORDS:", False)

print_list(tfidf[corpus], "PREVIOUS TF-IDF OF UPDATED CORPUS:")
tfidf_new = models.TfidfModel(corpus)
print_list(tfidf_new[corpus], "NEW TF-IDF OF UPDATED CORPUS:")

lsi.add_documents(tfidf_new[corpus_new])
print_list(lsi[tfidf_new[corpus]], "UPDATED LSI (from TF-IDF):")
lsi_new = models.LsiModel(tfidf_new[corpus], num_topics=NUM_TOP, id2word=dic)
print_list(lsi_new[tfidf_new[corpus]], "NEW LSI (from TF-IDF):")
lsiBOW.add_documents(corpus_new)
print_list(lsiBOW[corpus], "UPDATED LSI (from BOW):")
lsi_newBOW = models.LsiModel(corpus, num_topics=NUM_TOP, id2word=dic)
print_list(lsi_newBOW[corpus], "NEW LSI (from BOW):")