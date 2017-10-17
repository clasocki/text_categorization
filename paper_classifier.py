"""
Dependencies: gensim, nltk, sklearn, numpy
"""

from gensim import corpora, models
from nltk.corpus import stopwords
import re
import math
import numpy
import os
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.externals import joblib
import operator
import logging

class TupleFieldHandler(object): #Tuple(FieldHandler)
    """Converts a list of tuples between a db and python format.
    When writing to a db each tuple is separated by tupsep and each its element by elemsep.
    """
    def __init__(self, elemsep = ':', tupsep = ' ', null = ''):
        self.elemsep = elemsep
        self.tupsep = tupsep
        self.null = null

    def onWrite(self, pyval, oper):
        if not pyval: return self.null
        out = self.tupsep.join(self.elemsep.join(str(e) for e in tup) for tup in pyval)
        return out

    def onRead(self, dbval):
        if not dbval or dbval == self.null: return []
        l = dbval.split(self.tupsep)
        l = [tuple(tupstr.split(self.elemsep)) for tupstr in l]
        return l

def lognormalize(tf):
    return 1 + math.log(tf) if tf > 0 else 0

class SemanticModel(object):
    """With this class, a semantic model is built that conducts an analysis on relationships between documents and words in a provided corpus finding semantic concepts hidden in the corpus.
    """
    def __init__(self):
        """
        - lsi: latent semantic indexing model: gensim.models.lsimodel.LsiModel
        - tfidf: model converting bag-of-word corpus into a weighted tf-idf format: gensim.models.tfidfmodel.TfidfModel
        - dict: mapping between words in a corpus and their integer ids: gensim.corpora.dictionary.Dictionary
        - num_features: number of concepts in the lsi model
        """
        self.lsi = None
        self.tfidf = None
        self.dict = None
        self.num_features = None
        self.stopwords = stopwords.words('english')

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"\b-\b", "", text) 
        words = re.split(r'\W+', text)
        
        result = []
        for word in words:
            if word not in self.stopwords and not word.isdigit() and len(word) >= 2:
                result.append(word)

        return result

    def tokenizeTexts(self, texts):
        for text in texts:
            if text:
                yield self.tokenize(text)

    def createDictionary(self, tokenized_texts, no_below, no_above, keep_n):
        new_dictionary = corpora.Dictionary(tokenized_texts)
        new_dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)

        return new_dictionary

    def getTfIdfCorpus(self, texts):
        corpus = (self.dict.doc2bow(tokenized_text) for tokenized_text in self.tokenizeTexts(texts))
        
        tfidf_model = models.TfidfModel(corpus, dictionary=self.dict, wlocal=lognormalize, normalize=False)
        tfidf_corpus = tfidf_model[corpus]

        return tfidf_model, tfidf_corpus

    def getSerialPath(self, folder):
        return os.path.join(folder, 'semantic_model')

    @staticmethod
    def load(folder):
        """Load a semantic model from a provided folder.
           Returns an instance of SemanticModel.
        """
        model = SemanticModel()
        filepath = model.getSerialPath(folder)

        model.dict = corpora.Dictionary.load(filepath + '.dict')
        model.lsi = models.LsiModel.load(filepath + '.lsi')
        model.tfidf = models.TfidfModel.load(filepath + '.tfidf')
        model.num_features = model.lsi.num_topics

        return model

    def save(self, folder):
        """Save the semantic model to a provided folder. 
        """

        filepath = self.getSerialPath(folder)

        self.dict.save(filepath + '.dict')
        self.tfidf.save(filepath + '.tfidf')
        self.lsi.save(filepath + '.lsi')

    @staticmethod
    def build(texts, num_features, no_below=2, no_above=0.10, keep_n=300000):
        """Build a semantic model from a provided corpus of texts.
           -texts: an iterator to texts in a raw format that allows multiple iterations
           -num_features: the number of extracted topics in the latent semantic indexing process
           While building a dictionary of words in the texts, filter out all the words that appear in
           (1) less than no_below texts (absolute number) or
           (2) more than no_above texts (fraction of total corpus size, not absolute number).
           after (1) and (2), keep only the first keep_n most frequent words (or keep all if None).
        """

        model = SemanticModel()
        model.num_features = num_features

        model.dict = model.createDictionary(model.tokenizeTexts(texts), no_below, no_above, keep_n)
        model.tfidf, corpus_tfidf = model.getTfIdfCorpus(texts)
        model.lsi = models.LsiModel(corpus_tfidf, num_topics=num_features, id2word=model.dict)

        return model

    def inferProfile(self, text):
        """Infer a profile for a given text in a raw format.
        """
        tokenized_text = self.tokenize(text)
        text_bow = self.dict.doc2bow(tokenized_text)
        text_tfidf = self.tfidf[text_bow]
        profile = self.lsi[text_tfidf]

        profile = numpy.array([feature_val for feature_id, feature_val in profile])
        #profile = profile / numpy.linalg.norm(profile)
        return profile if profile.any() else numpy.zeros(self.num_features)

    def inferProfiles(self, texts):
        return numpy.array([self.inferProfile(text) for text in texts])

class RawTextCorpus(object):
    """Converts a multi-iterable corpus where each document has a fulltext property to a corpus of raw texts.
       Required for building SemanticModel that operates on raw texts only.
    """
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for pid, text in self.corpus:
            yield text

       
class PaperClassifier():
    def __init__(self):
        self.semantic_model = None
        self.classifier = None

    def prepareTrainingSet(self, trainset):
        """ -trainset: an iterable collection of tuples: (pid, raw text, disciplines) for papers with already assigned disciplines
        Returns a tuple: (train_profiles, profiles_dict, train_labels):
        -train_profiles: extracted profiles for each doc
        -train_labels: a list of known labels for each doc
        -profiles_dict: mappings between labels and lists of corresponding profiles
        """
        train_profiles, profiles_dict, train_labels, labeled_docs_ids = [], defaultdict(list), [], set()
        for pid, text, disciplines in trainset:
            profile = self.semantic_model.inferProfile(text)
            
            for discipline in disciplines:
                profiles_dict[discipline].append(profile)
            train_profiles.append(profile)
            train_labels.append(disciplines)
            labeled_docs_ids.add(pid)
        
        return train_profiles, train_labels, profiles_dict, labeled_docs_ids

    def fitClassifier(self, train_profiles, train_labels):
        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=10000)
        clf = OneVsRestClassifier(clf)
        mlb = MultiLabelBinarizer()
        
        mlb = mlb.fit(train_labels)
        train_labels = mlb.transform(train_labels)
        clf.fit(train_profiles, train_labels)

        return clf, mlb

    def allocateNewLabels(self, corpus, profiles_dict, classifier, label_binarizer, labeled_docs_ids,
            max_labels=2, min_confidence=0.8, max_closest_docs=0.05, max_new_allocs=0.05):
        """Iterates over corpus and assigns labels to documents with highest confidence.
        -corpus: an iterator to a full collection of documents in a corpus (allows multiple iteration), each document in the following format: (pid, raw text)		
        -profiles_dict: mappings between labels and lists of corresponding profiles, 
                        used to choose only labels for those documents which 
                        have high enough average distance to docs with the same already assigned label
        -label_binarizer: MultiLabelBinarizer: transforms between a string class format and a multi-label format
        -labeled_docs_ids: a set of pids for those papers that already have had labels assigned
        -max_labels: specifies how many labels maximum can be assigned to a document
        -min_confidence: only docs with highest confidence will have labels assigned
        -max_closest_docs: fraction of the total number of newly-labeled docs with a given class that can have labels assigned
        -max_new_allocs: specifies how many docs maximum can have labels assigned, 
                         (as a fraction of the total number of docs with already assigned labels)
        
        Returns: a generator of a list of tuples: (pid, profile, list of labels)
        """
        new_lbl_allocs = defaultdict(list)
        new_profiles = dict()
    
        #assign most confident labels based on confidence scores provided by the classfier
        for pid, text in corpus:
            if not text or pid in labeled_docs_ids: continue
            profile = self.semantic_model.inferProfile(text)
            if not profile.any(): continue
        
            probas = classifier.predict_proba([profile])[0]
            max_proba_ids = numpy.argsort(probas)[-max_labels:]

            any_confident_classes = False
            for max_proba_id in max_proba_ids:
                if probas[max_proba_id] < min_confidence: continue
                any_confident_classes = True
                class_ = label_binarizer.classes_[max_proba_id]
                avg_dist = pairwise_distances([profile], profiles_dict[class_], metric='cosine').sum() / float(len(profiles_dict[class_]))
                new_lbl_allocs[class_].append((avg_dist, pid))

            if any_confident_classes:
                new_profiles[pid] = profile
        
        #for each label narrow down the allocations to only those for which average distance to observations with known labels is small
        for lbl, allocs in new_lbl_allocs.iteritems():
            new_lbl_allocs[lbl] = sorted(new_lbl_allocs[lbl], key=operator.itemgetter(0))[: int(max_closest_docs * len(new_lbl_allocs[lbl]))]
       
        #undersampling is applied in order to prevent the labeled set from growing too fast and to prevent some classes
        #from exceeding substantially other classes sizes
        num_least_represented_classes = sorted([len(x) for x in new_lbl_allocs.values()])[:3]
        num_labeled_docs = sum([len(profiles) for profiles in profiles_dict.values()])
        max_num_new_allocs = int(min(numpy.sum(num_least_represented_classes), max_new_allocs * num_labeled_docs))
        
        lbls_per_doc = defaultdict(list)
        for lbl in new_lbl_allocs.keys():
            new_lbl_allocs[lbl] = new_lbl_allocs[lbl][:max_num_new_allocs]
    
            for avg_dist, pid in new_lbl_allocs[lbl]:
                lbls_per_doc[pid].append(lbl)
        
        return ((pid, new_profiles[pid], labels) for pid, labels in lbls_per_doc.iteritems())

    def propagateLabels(self, trainset, corpus, max_num_labeled_docs=20000, max_iter=10):
        """Iteratively allocates new labels to documents in corpus until a given number of documents is labeled.
        """
        train_profiles, train_labels, profiles_dict, labeled_docs_ids = self.prepareTrainingSet(trainset)
        num_labeled_docs = sum([len(profiles) for profiles in profiles_dict.values()])

        current_iter = 0
        while num_labeled_docs < max_num_labeled_docs and current_iter < max_iter:
            classifier, label_binarizer = self.fitClassifier(train_profiles, train_labels)
            new_allocations = self.allocateNewLabels(corpus, profiles_dict, classifier, label_binarizer, labeled_docs_ids)

            for pid, profile, lbls in new_allocations:
                labeled_docs_ids.add(pid)
                train_profiles.append(profile)
                train_labels.append(lbls)

                for lbl in lbls:
                    profiles_dict[lbl].append(profile)

                num_labeled_docs += 1
            
            current_iter += 1
        
        return train_profiles, train_labels
             
    @staticmethod
    def build(corpus, trainset):
	"""Builds a semantic model + document classifier on a given corpus of documents.
        - corpus: an iterator to a full collection of documents in a corpus (allows multiple iteration), each document in the following format: (pid, raw text)		
        - trainset: an iterator to a collection of labeled documents (allows multiple iteration), each document in the following format: (pid, raw text, disciplines)
        """
        paper_clf = PaperClassifier()
        paper_clf.semantic_model = SemanticModel.build(RawTextCorpus(corpus), num_features=400)
          
        train_profiles, train_labels = paper_clf.propagateLabels(trainset, corpus)
        classifier, label_binarizer = paper_clf.fitClassifier(train_profiles, train_labels)
        paper_clf.classifier = classifier
        paper_clf.label_binarizer = label_binarizer

        return paper_clf
	
    def classify(self, fulltext, min_discipline_contribution=0.8, max_num_disciplines=2):
        """Assigns semantic profile and disciplines to a given paper.
        -fulltext: a full text of a paper in a raw text format.
        -min_discipline_contribution: specifies a minimum percentage of how much a discipline contributes to a document to be taken into account in classification
        -max_num_disciplines: max number of disciplines assigned to a document
        Returns a pair: (profile, classes)
 	- profile: numpy vector
	- classes: list of (displines, weight) pairs, weight is an estimated contribution of the discipline to a given paper
	"""
        profile = self.semantic_model.inferProfile(fulltext)
        confidence_scores = self.classifier.predict_proba([profile])[0]
        confidence_scores /= confidence_scores.sum()
        sorted_conf_ids = numpy.argsort(confidence_scores)[-max_num_disciplines:]
        max_conf_ids = [max_id for max_id in sorted_conf_ids if confidence_scores[max_id] >= min_discipline_contribution]
        if not max_conf_ids:
            max_conf_ids = sorted_conf_ids[-1:].tolist()
        
        classes = self.label_binarizer.classes_[max_conf_ids]
        
        return profile, zip(classes, map(int, confidence_scores[max_conf_ids] * 100))

    def getSerialPath(self, folder):
        return os.path.join(folder, 'paper_classifier.clf')
 
    @staticmethod
    def load(folder):
        "Returns an instance of PaperClassifier."
        paper_clf = PaperClassifier()
        filepath = paper_clf.getSerialPath(folder)
        
        paper_clf.semantic_model = SemanticModel.load(folder)
        paper_clf.classifier = joblib.load(filepath)
        paper_clf.label_binarizer = joblib.load(filepath + '.lbin')

        return paper_clf

    def save(self, folder):
        """Saves all model data to several separate files located in a given folder
        """
        filepath = self.getSerialPath(folder)

        joblib.dump(self.classifier, filepath)
        joblib.dump(self.label_binarizer, filepath + '.lbin')
        self.semantic_model.save(folder)

############## Corpus helpers (for testing) ###############################

from paperity.environ import db
from paperity.content.paper import Paper

class Corpus(object):
    def __init__(self, db, doc_filter, english_only=False, db_window_size=10000):
        self.db = db
        self.doc_filter = doc_filter
        self.english_only = english_only
        self.db_window_size = db_window_size

    "A collection of strings (full texts of articles) that allows multiple iteration over all strings"
    def __iter__(self):
        db_row_count = self.db.select("SELECT COUNT(1) FROM pap_papers_view WHERE " + self.doc_filter)[0]['COUNT(1)']
        current_record_offset = 0
        
        while current_record_offset < db_row_count:
            query = "select p.* " + \
                    "from (select id from pap_papers_view where " + self.doc_filter + " " \
                    " order by id limit " + str(self.db_window_size) + " OFFSET " + str(current_record_offset) + \
                    ") q join pap_papers_view p on p.id = q.id"; 
            
            document_batch = Paper.selectRaw(query)
            
            #if current_record_offset > 20000: break;

            for doc in document_batch:
                doc.fulltext = ''
                if doc.title:
                    doc.fulltext += doc.title
                if doc.abstract:
                    doc.fulltext += " %s" % doc.abstract
                if doc.rawtext:
                    doc.fulltext += " %s" % doc.rawtext

                if self.english_only:
                    try:
                        detected_langs = langdetect.detect_langs(doc.fulltext)
                        detected_langs = sorted(detected_langs, key=lambda x: x.prob, reverse=True)
                    
                        if detected_langs and not (detected_langs[0].lang == 'en' and detected_langs[0].prob >= 0.8): continue

                    except Exception as e:
                        logging.error(e, exc_info=True)
                        continue

                yield doc

            current_record_offset += self.db_window_size


class LabeledCorpus(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for doc in self.corpus:
            yield doc.pid, doc.fulltext, [disc for disc, _ in doc.disciplines]

class UnlabeledCorpus(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for doc in self.corpus:
            yield doc.pid, doc.fulltext

################## Runner  #################################
 
if __name__ == "__main__":
    folder = 'paper_classifier_dumps'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #logging.basicConfig(filename=os.path.join(folder,'paper_classifier.log'), format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    #semantic_model = SemanticModel.load('paper_classifier_dumps')  
    
    trainset = LabeledCorpus(Corpus(db, doc_filter="published = 0 and is_test = 1"))
    full_corpus = UnlabeledCorpus(Corpus(db, doc_filter="is_test = 1"))
    test_set = LabeledCorpus(Corpus(db, doc_filter="published = 1 and is_test = 1"))

    paper_classifier = PaperClassifier.build(full_corpus, trainset)
    paper_classifier.save(folder)
    paper_classifier = PaperClassifier.load(folder)
    
    y_test, pred = [], []
    for pid, text, disciplines in test_set:
        profile, pred_disciplines = paper_classifier.classify(text)
        pred.append([c for c, _ in pred_disciplines])
        y_test.append(disciplines)

        print pred_disciplines

    y_test = paper_classifier.label_binarizer.transform(y_test)
    pred = paper_classifier.label_binarizer.transform(pred)

    from sklearn.metrics import classification_report
    print classification_report(y_test, pred, target_names=paper_classifier.label_binarizer.classes_)

