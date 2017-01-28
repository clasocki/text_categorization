from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint
import logging
import matplotlib.pyplot as plt
import numpy
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey", ""]

additional_documents_trees = [
    "Trees are special cases of graphs so whatever works for a general graph works for trees",
    "In mathematics, and more specifically in graph theory, a tree"
    " is an undirected graph in which any two vertices are connected by exactly one path, paths"
]

additional_documents_elephants = [
    "Overview of African and Asian elephants with photos and videos",
    "Elephants are large mammals of the family Elephantidae and the order Proboscidea",
    "Once common throughout Africa and Asia elephant numbers fell dramatically in the 19th and 20th centuries"
]

#stop_list = set('for a of the and to in'.split())
stop_list = set()

class DictionaryBasedCorpus(object):
    def __init__(self, dictionary, corpus_source_file):
        self.dictionary = dictionary
        self.corpus_source_file = corpus_source_file

    def __iter__(self):
        for line in open(self.corpus_source_file):
            yield self.dictionary.doc2bow(line.lower().split())


class MyCorpus(object):
    def __init__(self, corpus_source_file):
        self.corpus_source_file = corpus_source_file
        self.dictionary = self.create_new_dictionary(self.corpus_source_file)

    def __iter__(self):
        for line in open(self.corpus_source_file):
            yield self.dictionary.doc2bow(line.lower().split())

    @staticmethod
    def create_new_dictionary(corpus_file):
        new_dictionary = corpora.Dictionary(line.lower().split() for line in open(corpus_file))
        stop_ids = [new_dictionary.token2id[stopword] for stopword in stop_list
                    if stopword in new_dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in new_dictionary.dfs.iteritems() if docfreq == 1]
        new_dictionary.filter_tokens(stop_ids + once_ids)

        #self.dictionary.filter_extremes(no_below=0)
        new_dictionary.compactify()
        print(new_dictionary.token2id)
        new_dictionary.save('tmp/simpleDict.dict')
        return new_dictionary

    def add_documents(self, new_documents_file):
        new_dictionary = self.create_new_dictionary(new_documents_file)
        self.dictionary = self.dictionary.merge_with(new_dictionary)



# Calling model[corpus] only creates a wrapper around the old corpus document
# stream actual conversions are done on-the-fly, during document iteration.
#  We cannot convert the entire corpus at the time of calling
# corpus_transformed = model[corpus], because that would mean storing the result
# in main memory, and that contradicts gensims objective of memory-indepedence.
# If you will be iterating over the transformed corpus_transformed multiple times,
# and the transformation is costly, serialize the resulting corpus to disk first
#  and continue using that.

#for vector in corpus:
#    print tfidf[vector]

corpus = MyCorpus('mycorpus.txt')
print "Korpus BOW:"
for vector in corpus:
    print vector

#corpora.MmCorpus.serialize('tmp/corpus.mm', corpus)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

print "Korpus tfidf"
for doc in corpus_tfidf:
    print doc

num_topics = 2
lsi = models.LsiModel(corpus_tfidf, id2word=corpus.dictionary, num_topics=num_topics)
corpus_lsi = lsi[corpus_tfidf]
#lsi = models.LsiModel(corpus, id2word=corpus.dictionary, num_topics=num_topics)
#corpus_lsi = lsi[corpus]
print "Topic 1:"
lsi.show_topic(1)

print "Dokumenty w korpusie lsi:"

doc_count = 0
PP = []
for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print doc
    doc_features_dict = {feature_id: feature_val for (feature_id, feature_val) in doc}
    doc_features_list = [doc_features_dict[i] if i in doc_features_dict else 0 for i in range(0, num_topics)]
    PP.append(doc_features_list)
    doc_count += 1

print "---------------"

for doc in documents:
    text_bow = corpus.dictionary.doc2bow(doc.lower().split())
    text_tfidf = tfidf[text_bow]
    profile = lsi[text_tfidf]
    print profile

"""
PP = numpy.array(PP)
x = PP[:, 0]
print "X: " + str(x.tolist())
y = PP[:, 1]
print "Y: " + str(y.tolist())
#z = PP[:, 2]
#print "Z: " + str(z.tolist())
print "Dot product: " + str(numpy.dot(x.tolist(), y.tolist()))
n = ['d' + str(i) for i in range(1, doc_count + 1)]
fig, ax = plt.subplots()
ax.scatter(x, y)
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.show()

lsi.save('tmp/model.lsi')
#lsi = models.LsiModel.load('tmp/model.lsi')



doc = "Human computer interaction"
vec_bow = corpus.dictionary.doc2bow(doc.lower().split())
vec_tfidf = tfidf[vec_bow]


vec_lsi = lsi[vec_tfidf]

print "Mapowanie nowego dokumentu"
print vec_lsi
new_corpus = DictionaryBasedCorpus(corpus.dictionary, 'newcorpus.txt')

print "Nowy korpus BOW:"
for vector in new_corpus:
    print vector

new_tfidf = models.TfidfModel(new_corpus)
new_corpus_tfidf = new_tfidf[new_corpus]

print "Nowy korpus tfidf"
for doc in new_corpus_tfidf:
    print doc

print "Macierz U przed dodaniem dokumentow"
print lsi.projection.u

lsi.add_documents(new_corpus_tfidf)
lsi.print_topics(2)
print "Macierz U po dodaniu dokumentow"
print lsi.projection.u
print "Macierz S"
print lsi.projection.s
print "Po dodaniu dokumentow do lsi liczba dokumentow w koorpusie lsi nie zmienila sie"
for doc in corpus_lsi:
    print doc
print "Koncept 0"
print lsi.show_topic(0)
print lsi.show_topic(1)

"""




