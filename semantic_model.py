from paperity.environ import db
from paperity.content.paper import Paper
from nifty.text import tokenize
import numpy
from collections import defaultdict
import json
import operator
from enum import Enum
import math
import sys
import numbers
#from sortedcontainers import SortedSet
import time
import itertools
import numba
import warnings
from sklearn.datasets import fetch_20newsgroups
import random
from nltk.corpus import stopwords
import re


#numpy.seterr(all='warn')
#warnings.filterwarnings('error')

STOP = stopwords.words('english')

def tokenize1(text):
    text = text.lower()
    text = re.sub(r"\b-\b", "", text) 
    words = re.split(r'\W+', text)
    
    result = []
    for word in words:
        if word not in STOP and not word.isdigit() and len(word) >= 2:
            result.append(word)

    return ' '.join(result)


class Document(object):
    def __init__(self, id, rawtext, tokenized_text):
        self.id = id
        self.rawtext = rawtext
        self.word_weights = None
        self.tokenized_text = tokenized_text
        self.profile = None

class DocumentIterator(object):
    def __init__(self, doc_filter, document_batch_size=None, db_window_size=None):
        self.current_record_offset = 0
        self.DOCUMENT_BATCH_SIZE = document_batch_size
        self.DB_WINDOW_SIZE = db_window_size
        self.doc_filter = doc_filter

    def getRandomDocumentsFromDb(self):
        query = "SELECT * FROM " + \
            "((SELECT * " + \
            "FROM pap_papers_view WHERE " + \
            self.doc_filter + " " + \
            "LIMIT " + str(self.DB_WINDOW_SIZE) + " OFFSET " + str(self.current_record_offset) + ")"

        db_row_count = db.select("SELECT COUNT(1) FROM pap_papers_view WHERE " + self.doc_filter)[0]['COUNT(1)']

        remaining_batch_size = self.DB_WINDOW_SIZE - db_row_count + self.current_record_offset
        
        if db_row_count > 0:
            self.current_record_offset = (self.current_record_offset + self.DB_WINDOW_SIZE) % db_row_count

        if remaining_batch_size > 0:
            query += " UNION " + \
            "(SELECT * " + \
            "FROM pap_papers_view WHERE " + \
            self.doc_filter + " " + \
            "LIMIT " + str(remaining_batch_size) + " OFFSET 0)"

        query += ") LIMITED " + \
            "ORDER BY RAND() " + \
            "LIMIT " + str(self.DOCUMENT_BATCH_SIZE)

        #query = "SELECT * FROM pap_papers_view WHERE published = 1 AND is_test = 1"
        #print query
        #document_batch = db.select(query)
        document_batch = Paper.selectRaw(query)    

        #print "Current record offset: " + str(self.current_record_offset)
        
        return document_batch

    def getAllInBatches(self):
        db_row_count = db.select("SELECT COUNT(1) FROM pap_papers_view WHERE " + self.doc_filter)[0]['COUNT(1)']
        
        while self.current_record_offset < db_row_count:
            print self.current_record_offset
            
            query = "SELECT * " + \
                "FROM pap_papers_view WHERE " + self.doc_filter + " " + \
                "LIMIT " + str(self.DB_WINDOW_SIZE) + " OFFSET " + str(self.current_record_offset)

            document_batch = Paper.selectRaw(query)

            yield self.processDocuments(document_batch)

            self.current_record_offset += self.DB_WINDOW_SIZE

    def getAll(self, convert=None):
        query = "SELECT * FROM pap_papers_view WHERE " + self.doc_filter

        all_documents = Paper.selectRaw(query)

        return self.processDocuments(all_documents, convert)
    
    def getAllByIds(self, document_ids, convert=None):
        query = "SELECT * FROM pap_papers_view WHERE " + self.doc_filter + " AND id in (" + ','.join(str(doc_id) for doc_id in document_ids) + ")"

        all_documents = Paper.selectRaw(query)

        return self.processDocuments(all_documents, convert)

    @staticmethod
    def getAll2():
        query = "SELECT * FROM pap_papers_view WHERE published = 1"

        all_documents = Paper.selectRaw(query)

        return self.processDocuments(all_documents)

    def saveDocumentProfilesToFile(self, documents, file_name):
        with open(file_name, 'w') as f:
            for document in all_documents:
                str_doc_profile = str(document.id) + '\t'
                str_doc_profile += ','.join(str(profile_element) for profile_element in document.profile) + '\n'
                f.write(str_doc_profile)

    def saveDocumentProfilesToDb(self, documents):
        for document in documents:
            db_document_id = document.id
            document_profile = document.profile
            str_document_profile = ','.join(str(profile_element) for profile_element in document_profile)

            #sql_update = "UPDATE pap_papers_3 SET profile = '" + \
            sql_update = "UPDATE pap_papers_view SET profile = '" + \
                str_document_profile + \
                "' WHERE id = " + str(db_document_id)

            db.query(sql_update)
            #document.update()

        db.commit()

    #@profile
    #@numba.jit(cache=True)
    def processDocuments(self, documents, convert):
        processed_docs = []            

        for document in documents:
            document.tokenized_text = tokenize(document.rawtext).split()
            document.profile = numpy.asarray(document.profile)
            document.word_weights = convert(document.id, document.tokenized_text) if convert else None

            if (convert and len(document.word_weights) > 0) or (not convert and len(document.tokenized_text) > 0):
                processed_docs.append(document)
     
        return processed_docs

    #@profile
    def batchIter(self, convert=None):
        while True:
            document_batch = self.getRandomDocumentsFromDb()
            if not document_batch:
                break

            yield self.processDocuments(document_batch, convert)
    """
    def docByDocIter(self, doc_filter=None):
        query = "SELECT rawtext,  FROM pap_papers_view WHERE " + doc_filter if doc_filter else self.doc_filter
        cursor = db.execute(query)
        for doc in cursor:
            tokenized_text = tokenize(document.rawtext).split()
            profile = numpy.asarray(document.profile)
            word_weights = self.calculateWordWeights(document.tokenized_text) if self.calculateWordWeights else None
            yield Document(


        return cursor.rowcount, cursor
    """

class InMemoryDocumentIterator(object):
    def __init__(self, data_set):
        self.data_set = data_set
	self.docs = dict((i, Document(i, text, tokenize(text).split())) for i, text in enumerate(data_set))

    def processDocuments(self, documents, convert):
        processed_docs = []            

        for document in documents:
            document.word_weights = convert(document.id, document.tokenized_text) if convert else None

            if (convert and len(document.word_weights) > 0) or not convert: #or (not convert and len(document.tokenized_text) > 0):
                processed_docs.append(document)
        random.shuffle(processed_docs) 
        return processed_docs
     
    def processDocuments2(self, documents, convert):
        doc_profiles, word_weights, doc_ids = [], [], []            
        current_doc = 0
        for document in documents:
            document.word_weights = convert(document.id, document.tokenized_text) if convert else None

            if (convert and len(document.word_weights) > 0) or not convert: #or (not convert and len(document.tokenized_text) > 0):
                doc_profiles.append(document.profile)
                for word_id, word_weight in document.word_weights:
                    word_weights.append((current_doc, word_id, word_weight))
                doc_ids.append(document.id)
                current_doc += 1

        random.shuffle(word_weights) 
        return numpy.asarray(doc_profiles), word_weights, doc_ids

    def updateProfiles(self, doc_profiles, doc_ids):
        for doc_profile, doc_id in zip(doc_profiles, doc_ids):
            self.docs[doc_id].profile = doc_profile

    def getAll(self, convert=None):
        return self.processDocuments(self.docs.values(), convert)

    def getAllByIds(self, ids, convert=None):
        return self.processDocuments((self.docs[i] for i in ids), convert)

    def getById(self, doc_id):
        return self.docs[doc_id]

    def batchIter(self, convert=None):
        while True:
            docs = self.processDocuments(self.docs.values(), convert) #required as each time new words are selected
            yield docs

#class CalculationHelper(object):
#@numba.jit("float32(float32[:],float32[:,:], int32)", nopython=True,cache=True)
@numba.jit(nopython=True,cache=True)
def predictValue(profile, word_profiles, word_id):
    return numpy.dot(profile, word_profiles[word_id, :])

#@numba.jit("void(float32[:],float32,float32,float32[:,:],int32,float32)",nopython=True,cache=True)
@numba.jit(nopython=True,cache=True)
def updateWordProfile(profile, learning_rate, error, word_profiles, word_id, regul_factor):
    word_profiles[word_id, :] += \
        learning_rate * (error * profile - regul_factor * word_profiles[word_id, :])

@numba.jit(nopython=True,cache=True)
def updateDocProfile(profile, learning_rate, error, word_profiles, word_id, regul_factor):
    profile += \
        learning_rate * (error * word_profiles[word_id, :] - regul_factor * profile) 

#@profile
@numba.jit(nopython=True,cache=True)
#@numba.jit(numba.types.Tuple((numba.float64, numba.int64))(numba.typeof([(2, 3)]), numba.float64[:], numba.float64[:, :], numba.float64, numba.float64, numba.boolean, numba.boolean, numba.typeof(set([3]))), nopython=True,cache=True)
def inferProfilesPerDocument(word_weights, document_profile, word_profiles, learning_rate, 
                             regul_factor, update_document_profiles, update_word_profiles):#, updated_word_ids):
    squared_error, num_values = 0, 0

    for word_id, weight in word_weights:
        predicted_value = numpy.dot(document_profile, word_profiles[word_id, :])
        error = 1.0 * weight - predicted_value
        original_document_profile = numpy.copy(document_profile)

        if update_document_profiles:
            document_profile += \
                learning_rate * (error * word_profiles[word_id, :] - regul_factor * original_document_profile) 

        if update_word_profiles:
            word_profiles[word_id, :] += \
                learning_rate * (error * original_document_profile - regul_factor * word_profiles[word_id, :])

        squared_error += error * error
        num_values += 1

        #if update_word_profiles:
        #    updated_word_ids.add(word_id)

    return squared_error, num_values

@numba.jit(nopython=True,cache=True)
def inferProfilesPerDocuments(word_weights, document_profiles, word_profiles, learning_rate, 
                             regul_factor, update_document_profiles, update_word_profiles):#, updated_word_ids):
    squared_error, num_values = 0, 0

    for doc_id, word_id, weight in word_weights:
        document_profile = document_profiles[doc_id]
        predicted_value = numpy.dot(document_profile, word_profiles[word_id, :])
        error = 1.0 * weight - predicted_value
        original_document_profile = numpy.copy(document_profile)

        if update_document_profiles:
            document_profile += \
                learning_rate * (error * word_profiles[word_id, :] - regul_factor * original_document_profile) 

        if update_word_profiles:
            word_profiles[word_id, :] += \
                learning_rate * (error * original_document_profile - regul_factor * word_profiles[word_id, :])

        squared_error += error * error
        num_values += 1

        #if update_word_profiles:
        #    updated_word_ids.add(word_id)

    return squared_error, num_values

@numba.jit(nopython=True,cache=True)
#@numba.jit(numba.types.Tuple((numba.float64, numba.float64))(numba.float64[:], numba.typeof([(2, 3)]), numba.float64[:,:], numba.float64), nopython=True,cache=True)
def calculateErrorPerDocument(doc_profile, word_weights, word_profiles, regularization_factor):
    rsse, num_values = 0.0, 0
    
    for word_id, value in word_weights:
        num_values += 1
        predicted_value = numpy.dot(doc_profile, word_profiles[word_id, :])
        error = 1.0 * value - predicted_value
        rsse += error ** 2 + regularization_factor * (numpy.linalg.norm(doc_profile) ** 2 + numpy.linalg.norm(word_profiles[word_id, :]) ** 2)

    return rsse, num_values

class SemanticModel(object):
    def __init__(self, document_iterator, num_features=10, file_name=None,
        term_freq_weight='log_normalization', use_idf = True,
        min_df=0.001, max_df=0.33, learning_rate=0.001, regularization_factor=0.01, 
        neg_weights=3.0, doc_prof_low=-0.01, doc_prof_high=0.01, word_prof_low=-0.01, word_prof_high=0.01, 
        limit_features=True, preanalyze_documents=True, tester=None, save_frequency=40, test_frequency=40, save_model=True,
        with_validation_set=False, save_to_db=True, decay=0.0):
        """
        :param num_features: number of features inferred from the document set
        :param file_name: the file used for the serialization
        :param term_freq_weight: the numerical statistic used as a term weighting factor in feature retrieval
        :param use_idf: determines if the inverse document frequency factor should be used
        :param min_df: float in range [0.0, 1.0] or int; float - represents a proportion of documents, words that have a document
            frequency strictly lower than the given threshold (min_df * self.num_docs) are ignored, int - absolute count
        :param max_df: float in range [0.0, 1.0] or int; float - represents a proportion of documents, words that have a document
            frequency strictly higher than the given threshold (max_df * self.num_docs) are ignored, int - absolute count
        :param limit_features: if True then removes too rare or too common features using min_df, max_df parameters
        :return:
        """
        self.num_docs = 0
        self.num_features = num_features
        self.file_name = file_name
        self.term_freq_weight = term_freq_weight
        self.tf = self.setUpTf()
        self.use_idf = use_idf
        self.idf = self.setUpIdf()
        self.min_df = min_df
        self.max_df = max_df
        self.learning_rate = learning_rate
        self.regularization_factor = regularization_factor
        self.neg_weights = neg_weights
        self.doc_prof_low = doc_prof_low
        self.doc_prof_high = doc_prof_high
        self.word_prof_low = word_prof_low
        self.word_prof_high = word_prof_high
        self.limit_features = limit_features
        self.preanalyze_documents = preanalyze_documents
        self.tester = tester
        self.save_frequency = save_frequency
        self.test_frequency = test_frequency
        self.save_model = save_model
        self.with_validation_set = with_validation_set
        self.validation_set = None
        self.save_to_db = save_to_db
        self.decay = decay
        self.word_profiles = numpy.random.uniform(low=-0.01, high=0.01, size=(1, self.num_features))
        self.document_iterator = document_iterator
        self.token_to_id = dict() # token -> token id
        self.id_to_token = dict()
        self.doc_freqs = defaultdict(int)  # token id -> the number of documents this token appears in
        self.last_active_id = -1 #describes the id of the last active word
        #self.updated_word_ids = set()
        self.word_id_to_document_ids = defaultdict(list)
        self.current_document_batch = None 

    def adaptiveLearningRate(self, decay, learning_rate, current_iter):
        return learning_rate * (1.0 / (1.0 + decay * learning_rate * current_iter))
  
    def inferProfile(self, rawtext, num_iters, learning_rate, regularization_factor, decay=0.0, zero_weights=None):
        words = tokenize(rawtext).split()
        profile = self.getInitialDocumentProfile()
        #word_weights = self.calculateRandomizedWordWeights(words, zero_weights=zero_weights)
        #word_weights = list(self.calculateTfIdf(words).iteritems())
        #f not word_weights: return numpy.empty(self.num_features)
        for current_iter in xrange(num_iters):
            word_weights = self.calculateRandomizedWordWeights(words, zero_weights=zero_weights)
            if not word_weights: return numpy.empty(self.num_features)

            inferProfilesPerDocument(word_weights, profile, self.word_profiles, self.adaptiveLearningRate(decay, learning_rate, current_iter), 
                                 regularization_factor, update_document_profiles=True, update_word_profiles=False)#, updated_word_ids=set([-1]))

        return profile

    #@profile
    def inferProfiles(self, documents, epoch, update_document_profiles=True, update_word_profiles=True, initialize_document_profiles=False, 
                      initialize_word_profiles=False, num_iters=1, print_stats=False):
        """
        Calculates profiles for documents
        :param documents: a list of unconverted documents
        :return:
        """    

        start_time = time.time()

        if initialize_document_profiles:
            documents = self.initializeDocumentProfiles(documents)  
    
        #if initialize_word_profiles and selected_word_ids is not None:
        #    self.initializeWordProfiles(selected_word_ids, len(selected_word_ids))

        if num_iters > 1:
            documents = list(documents)

        #self.updated_word_ids.add(-1)

        for current_iter in xrange(num_iters):
            squared_approx_error = 0.0
            num_words = 0

            for document in documents:
                err, num_w = inferProfilesPerDocument(document.word_weights, document.profile, self.word_profiles, 
                    self.adaptiveLearningRate(self.decay, self.learning_rate, epoch), self.regularization_factor, 
                    update_document_profiles, update_word_profiles)#, self.updated_word_ids)
            
                squared_approx_error += err
                num_words += num_w
           
            rmse = numpy.sqrt(squared_approx_error / num_words) if num_words > 0 else 0.0

        #self.updated_word_ids.remove(-1)            

        return (document.profile for document in documents)
    
    def inferProfiles2(self, documents, epoch, update_document_profiles=True, update_word_profiles=True, initialize_document_profiles=False, 
                      initialize_word_profiles=False, num_iters=1, print_stats=False):
        """
        Calculates profiles for documents
        :param documents: a list of unconverted documents
        :return:
        """    

        doc_profiles, word_weights, doc_ids = documents

        inferProfilesPerDocuments(word_weights, doc_profiles, self.word_profiles, 
            self.adaptiveLearningRate(self.decay, self.learning_rate, epoch), self.regularization_factor, 
            update_document_profiles, update_word_profiles)#, self.updated_word_ids)

        self.document_iterator.updateProfiles(doc_profiles, doc_ids)
            

    def createValidationSet(self):
        if not self.with_validation_set: return None

        validation_set = []
        num_words = self.last_active_id + 1
        total_training_set_size = self.num_docs * num_words
        sample_size = numpy.floor(0.2 * total_training_set_size)
        sample = numpy.random.choice(total_training_set_size, sample_size, replace=False)      
        
        doc_id = lambda i: i / num_words
        word_id = lambda i: i % num_words

        validation_set = defaultdict(set)
        for i in sample:
            validation_set[doc_id(i)].add(word_id(i))
        
        return validation_set

    def createProportionateValidationSet(self):
        if not self.with_validation_set: return None, None

        validation_set = []
        num_words = self.last_active_id + 1
        total_training_set_size = self.num_docs * num_words
        
        doc_converter = lambda doc_id, words: self.calculateTfIdf(words)
        positive_weights = []
        for document in self.document_iterator.getAll(convert=doc_converter):
            for word_id in document.word_weights.keys():
                positive_weights.append((document.id, word_id))

        positive_sample_size = numpy.floor(0.2 * len(positive_weights))
        zero_sample_size = numpy.floor(self.neg_weights * positive_sample_size)

        positive_sample = numpy.random.choice(len(positive_weights), positive_sample_size, replace=False)
        zero_sample = numpy.random.choice(total_training_set_size, zero_sample_size, replace=False)

        doc_id = lambda i: i / num_words
        word_id = lambda i: i % num_words

        doc_word_pairs = ((doc_id(i), word_id(i)) for i in zero_sample)
        validation_set = defaultdict(set)
        positive_validation_set = defaultdict(set)
       
        for i in positive_sample:
            validation_set[positive_weights[i][0]].add(positive_weights[i][1])
            positive_validation_set[positive_weights[i][0]].add(positive_weights[i][1])
         
        for i in zero_sample:
            validation_set[doc_id(i)].add(word_id(i))
            #positive_validation_set[doc_id(i)].add(word_id(i))
        
       
        return positive_validation_set, validation_set


    def calculateError(self, documents):
        total_rsse = 0.0
        num_values = 0

        for document in documents:
            rsse, num_w = calculateErrorPerDocument(document.profile, document.word_weights, self.word_profiles, self.regularization_factor)
            total_rsse += rsse
            num_values += num_w
        
        rmse = numpy.sqrt(total_rsse / num_values) if num_values > 0 else 0.0

        return rmse

    def update(self, documents, num_iters_full_retrain, num_iters_partial_retrain):
        """
        Updates the model based on a new batch of documents
        Profile matrices are resized if after the update the number of documents or words 
        exceeds the initial size of profile matrices ..
        :param documents: a batch of new documents
        :param num_iters if unum_iters is not None then initialize document profiles based on current model
        :return:
        """
        """
        self.current_document_batch = documents
        
        if num_iters_profile_inference:
            self.inferProfiles(documents, 
                num_iters=num_iters_profile_inference, 
                update_word_profiles=True, initialize_document_profiles=True)
        """
        start_time = time.time()
        #self.inferProfiles(documents, update_document_profiles=False, initialize_word_profiles=True, num_iters=num_iters_partial_retrain, print_stats=True)
        new_word_ids = self.updateStatisticsForNewDocuments(documents)       
        print "Limiting the number of words..."
        kept_word_ids, added_word_ids, removed_word_ids = self.limit_words(new_word_ids)
        print "Kept: " + str(len(kept_word_ids)) + ", added: " + str(len(added_word_ids)) + ", removed: " + str(len(removed_word_ids))
        """
        print "Updating model based on added words"
        
        if len(added_word_ids) > 0:
            documents_containing_added_words = self.loadDocumentsByWords(added_word_ids)
            for document_batch in documents_containing_added_words:
                added_word_id_set = set(word_id for word_id in added_word_ids)
                print "Updating words"
                self.inferProfiles(document_batch, update_document_profiles=False, 
                    num_iters=num_iters_partial_retrain, selected_word_ids=added_word_id_set, print_stats=True)
                print "Updating docs"
                self.inferProfiles(document_batch, update_word_profiles=True, initialize_document_profiles=True, 
                    num_iters=num_iters_partial_retrain, print_stats=True)
         
        print "Updating model based on removed words"
        
        if len(removed_word_ids) > 0:
            documents_containing_removed_words = self.loadDocumentsByWords(removed_word_ids)
            for document_batch in documents_containing_removed_words:
                self.inferProfiles(document_batch, update_word_profiles=False, initialize_document_profiles=True, num_iters=num_iters_partial_retrain, print_stats=True)
        
        """
        #sys.exit()
        
        print "Update time: " + str(time.time() - start_time)
        
        self.save(save_words=True)

        self.train(num_iters_full_retrain, print_stats=True)

        self.save(save_words=True)
    
    def loadDocumentsByWords(self, word_ids):
        document_ids = []
        for word_batch in numpy.array_split(word_ids, math.ceil(float(len(word_ids)) / 300)):
            word_list = ("'" + self.id_to_token[word_id] + "'" for word_id in word_batch)
            word_list = ', '.join(word_list)
            print word_list
            sql = "SELECT distinct document_id FROM pap_word_documents p WHERE p.word in (%s)" % word_list
            document_ids.extend(result["document_id"] for result in db.select(sql))

        document_ids = numpy.unique(document_ids)
        numpy.random.shuffle(document_ids)

        print "Document count: " + str(len(document_ids))

        for document_id_batch in numpy.array_split(document_ids, math.ceil(float(len(document_ids)) / 300)):
            where = "id in (%s)" % ', '.join(str(doc_id) for doc_id in document_id_batch)
            yield DocumentIterator(doc_filter=where).getAll()
 
    def predictValueOld(self, document, word_id):
        #print "Document id " + str(document_id)
        #print "Word id " + str(word_id)
        return numpy.dot(document.profile, self.word_profiles[word_id, :])

    def updateStatisticsForNewDocument(self, document):
        self.num_docs += 1

        processed_tokens = set()
        for token in document.tokenized_text:
            if len(token) > 100:
                continue
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
                self.id_to_token[self.token_to_id[token]] = token
                #self.active_tokens.add(self.token_to_id[token])

            if token not in processed_tokens:
                token_id = self.token_to_id[token]
                self.doc_freqs[token_id] += 1
                self.word_id_to_document_ids[token_id].append(document.id)
                processed_tokens.add(token)

        #print "The number of tokens: " + str(len(self.token_to_id))
        #print self.token_to_id.keys()

    def updateStatisticsForNewDocuments(self, documents):
        num_tokens_before_update = len(self.token_to_id)
        
        for document in documents:
            self.updateStatisticsForNewDocument(document)

        num_tokens_after_update = len(self.token_to_id)
        num_new_tokens = num_tokens_after_update - num_tokens_before_update

        #self.updated_word_ids.update(xrange(num_tokens_before_update, num_new_tokens))
        new_word_ids = set(xrange(num_tokens_before_update, num_new_tokens))
        return new_word_ids
        #self.compactify(added_ids=range(tokens_before_update, num_new_tokens), removed_ids=[])

    def getInitialDocumentProfile(self):
        return numpy.random.uniform(low=self.doc_prof_low, high=self.doc_prof_high, size=self.num_features)

    def initializeDocumentProfile(self, document):
        document.profile = self.getInitialDocumentProfile()

        return document

    def initializeDocumentProfiles(self, documents):
        for document in documents:
            self.initializeDocumentProfile(document)
        
    def initializeWordProfiles(self, word_ids, num_words):
        new_word_profiles = numpy.random.uniform(low=self.word_prof_low, high=self.word_prof_high, 
                size=(num_words, self.num_features))
        for new_profile_id, word_id in enumerate(word_ids):
            self.word_profiles[word_id, :] = new_word_profiles[new_profile_id, :]

    def setUpTf(self):
        if self.term_freq_weight == 'raw_frequency':
            return lambda bow, raw_freq: raw_freq
        elif self.term_freq_weight == 'log_normalization':
            return lambda bow, raw_freq: 1 + (math.log(raw_freq) if raw_freq > 0 else 0)
        elif self.term_freq_weight == 'augmented_frequency':
            return lambda bow, raw_freq: 0.5 + (0.5 * raw_freq / max(bow.values()))
        elif self.term_freq_weight == 'binary':
            return lambda bow, raw_freq: 1 if raw_freq > 0 else 0

    def setUpIdf(self):
        if not self.use_idf:
            return lambda word_id: 1

        return lambda num_docs, word_id, df: math.log(1 + (1.0 * num_docs / df[word_id])) if df[word_id] > 0 else 1

    #@profile
    def compactify(self, added_ids, removed_ids):
        """
        added_ids - added to the list of active tokens
        removed ids - removed from the list of active tokens
        """
        num_all_tokens = len(self.token_to_id)
        active_tokens_without_removed = (token_id for token_id in xrange(self.last_active_id + 1) if token_id not in removed_ids)
        active_token_ids = itertools.chain(active_tokens_without_removed, added_ids)
        
        inactive_tokens_without_added = (token_id for token_id in xrange(self.last_active_id + 1, num_all_tokens) if token_id not in added_ids)
        inactive_token_ids = itertools.chain(inactive_tokens_without_added, removed_ids)

        all_token_ids = itertools.chain(active_token_ids, inactive_token_ids)

        self.last_active_id = self.last_active_id + len(added_ids) - len(removed_ids)

        id_map = dict(itertools.izip(all_token_ids, xrange(num_all_tokens)))

        self.token_to_id = dict((token, id_map[token_id]) for token, token_id in self.token_to_id.iteritems())
        self.id_to_token = dict((id_map[token_id], token) for token_id, token in self.id_to_token.iteritems())
        self.doc_freqs = defaultdict(int, ((id_map[token_id], freq) for token_id, freq in self.doc_freqs.iteritems()))
        #self.updated_word_ids = set(id_map[token_id] for token_id in self.updated_word_ids)
        self.word_id_to_document_ids = defaultdict(list, ((id_map[token_id], doc_ids) for token_id, doc_ids in self.word_id_to_document_ids.iteritems()))

        for token_id in active_tokens_without_removed:
            temp = self.word_profiles[id_map[token_id], :]
            self.word_profiles[id_map[token_id], :] = self.word_profiles[token_id, :]
            self.word_profiles[token_id, :] = temp

        self.word_profiles = numpy.resize(self.word_profiles,
                (self.last_active_id + 1, self.num_features))

        self.initializeWordProfiles((id_map[word_id] for word_id in added_ids), len(added_ids))

        return id_map

    #@profile
    def limit_words(self, new_word_ids):
        min_document_count = (self.min_df
            if isinstance(self.min_df, numbers.Integral)
            else self.min_df * self.num_docs)
        max_document_count = (self.max_df 
            if isinstance(self.max_df, numbers.Integral) 
            else self.max_df * self.num_docs)

        kept_ids = set()
        removed_ids = set()
        added_ids = set()

        print "Min doc count: %s, min_df: %s, max doc count: %s, max_df: %s" % (min_document_count, self.min_df, max_document_count, self.max_df)

        for token_id, freq in self.doc_freqs.iteritems():
            if min_document_count <= freq and freq <= max_document_count:
                if token_id > self.last_active_id:
                #if token_id not in self.active_tokens:
                    #self.active_tokens.add(token_id)
                    added_ids.add(token_id)
                else:
                    kept_ids.add(token_id)
            else:
                if token_id <= self.last_active_id:
                #if token_id in self.active_tokens:
                    #self.active_tokens.remove(token_id)
                    removed_ids.add(token_id)
        other_new_word_ids = []
        for new_word_id in new_word_ids:
            if new_word_id not in removed_ids and new_word_id not in added_ids:
                other_new_word_ids.append(new_word_id)

        #self.updated_word_ids.update(added_ids)
        #self.updated_word_ids.update(removed_ids)

        id_map = self.compactify(added_ids, removed_ids)
        kept_ids = numpy.fromiter((id_map[kept_id] for kept_id in kept_ids), numpy.long)
        added_ids = numpy.fromiter((id_map[added_id] for added_id in added_ids), numpy.long)
        removed_ids = numpy.fromiter((id_map[removed_id] for removed_id in removed_ids), numpy.long)
        other_new_word_ids = numpy.fromiter((id_map[new_word_id] for new_word_id in other_new_word_ids), numpy.long)

        return kept_ids, added_ids, removed_ids, other_new_word_ids

    def calculateTfIdf(self, words, word_filter=lambda word_id: True):
        bag_of_words = defaultdict(int)
        for word in words:
            if word in self.token_to_id:
                word_id = self.token_to_id[word]

                if word_id <= self.last_active_id:
                #if token_id in self.active_tokens:
                    bag_of_words[word_id] += 1
         
        word_weights = dict((word_id, self.tf(bag_of_words, raw_freq) * self.idf(self.num_docs, word_id, self.doc_freqs)) 
                                  for word_id, raw_freq in bag_of_words.iteritems() if word_filter(word_id))

        return word_weights
    
    #@profile
    #@numba.jit(cache=True)
    def calculateRandomizedWordWeights(self, words, word_filter=lambda w_id: True, zero_weights=None):
        #validation_set_words = self.validation_set[document_id] if self.validation_set and document_id else set()
        #word_filter = (lambda w_id: w_id not in validation_set_words) if validation_set_words else (lambda w_id: True)
        word_weights = self.calculateTfIdf(words, word_filter) 

        return self.randomizeWordWeights(word_weights, word_filter, zero_weights)

    def calculateSelectedWordWeights(self, words, selected_word_ids):
        word_weights = self.calculateTfIdf(words, word_filter=lambda word_id: word_id in selected_word_ids)
        for word_id in selected_word_ids:
            if word_id not in word_weights:
                word_weights[word_id] = 0

        return list(word_weights.iteritems()) 

    def convertDocuments(self, documents, convert):
        """
        Tokenizes and converts each document from the raw text form to the bag-of-words format
        :param documents:
        :return: list of documents in the bag-of-words format with a local id assigned
        """
        for document in documents:
            document.word_weights = convert(document.id, document.tokenized_text)

        return documents

    #@numba.jit(cache=True)
    def randomizeWordWeights(self, word_weights, word_filter, zero_weights=None):
        """
        :param document: document represented as a map: token_id -> token_weight where token_id exists in the document
        :return: document represented as an iterator returning tuples: (token_id, token_weight)
                 token_weight != 0 if token exists in the document
                 token_weight  = 0 otherwise
        """
        positive_weight_count = len(word_weights)
        #print "Last active id: %s, positive weight count: %s" % (self.last_active_id, positive_weight_count)
        sample_size = min(self.last_active_id + 1, int(math.ceil(positive_weight_count * self.neg_weights)))
        if zero_weights:
            sample_size = min(self.last_active_id + 1, int(math.ceil(positive_weight_count * zero_weights)))

        random_word_ids = numpy.random.choice(self.last_active_id + 1, sample_size, replace=False) if self.last_active_id != -1 else numpy.empty(shape=(0,))

        
        zero_weights = ((word_id, 0) 
            for word_id in random_word_ids 
            if word_id not in word_weights and word_filter(word_id))

        weights = list(itertools.chain(zero_weights, word_weights.iteritems()))
        
        
        """
        weights = []
        for word_id in random_word_ids:
            if word_id in word_weights:
                weights.append((word_id, word_weights[word_id]))
            else:
                weights.append((word_id, 0))
        """

        numpy.random.shuffle(weights)
        return weights


    def resetDb(self):
        delete_words = "delete from pap_words"
        delete_word_document_mappings = "delete from pap_word_documents"
        reset_profiles_sql = "update pap_papers_view set profile = null"
        reset_learned_categories = "update pap_papers_view set learned_category = null where is_test = 0"
    
    #@profile
    def train(self, num_iter=None, print_stats=False):
        """
        Main loop that iterates over the db, periodically getting new document batches.
        Starts using empty profile matrices, incrementally resizing them in the process of model training.
        :return:
        """
        epoch = 0
        prev_rmse1 = sys.maxint
        prev_rmse2 = sys.maxint
        doc_count = 0
        docs_since_last_epoch = 0
        all_documents = []
        kept_indices, added_indices, removed_indices, other_indices = None, None, None, None

        if self.preanalyze_documents:
            all_documents = self.document_iterator.getAll(convert=None)
            doc_count = len(all_documents)
            new_word_ids = self.updateStatisticsForNewDocuments(all_documents)
            kept_indices, added_indices, removed_indices, other_indices = self.limit_words(new_word_ids)
            print "Kept: " + str(len(kept_indices)) + ", added: " + str(len(added_indices)) + ", removed: " + str(len(removed_indices))
            
        positive_validation_set, validation_set = self.createProportionateValidationSet()
        train_word_filter = (lambda doc_id, word_id: word_id not in validation_set[doc_id]) if validation_set else (lambda doc_id, word_id: True)
        train_doc_converter = lambda doc_id, words: self.calculateRandomizedWordWeights(words, word_filter=lambda word_id: train_word_filter(doc_id, word_id))

        if self.preanalyze_documents:
            all_documents = self.convertDocuments(all_documents, train_doc_converter)
            self.initializeDocumentProfiles(all_documents)
            self.current_document_batch = all_documents
            self.save(save_words=True, word_ids=itertools.chain(added_indices, removed_indices, other_indices))
        #while (epoch < self.min_iter or rmse_last - rmse >= self.min_improvement) and epoch < self.max_iter:
        #while num_iter is None or epoch < num_iter:
        #while True:
        for document_batch in self.document_iterator.batchIter(convert=train_doc_converter):
            self.current_document_batch = document_batch

            start_time = time.time()
            self.inferProfiles(document_batch, epoch, print_stats=print_stats)
            inference_time = time.time() - start_time
            
            start_time = time.time()
            self.save(save_words=(epoch % self.save_frequency == 0 and epoch != 0), word_ids=xrange(self.last_active_id + 1))

            saving_time = time.time() - start_time
            docs_since_last_epoch += len(document_batch)
            print "Iter: %s, Save: %s, Infer: %s" % (str(epoch), str(saving_time), str(inference_time))
               
            #if (doc_count and docs_since_last_epoch >= doc_count) or not doc_count:
            #    docs_since_last_epoch = 0

            if epoch % self.test_frequency == 0 and epoch != 0:
                                
                training_set = self.document_iterator.getAll(convert=train_doc_converter)
                training_set_rmse = self.calculateError(training_set)
                rmse_results = "Training set RMSE: " + str(training_set_rmse)
                
                validation_set_rmse = 0.0
                positive_validation_set_rmse = 0.0
                if validation_set:
                    validation_docs_ids = validation_set.keys()
                    validation_docs = self.document_iterator.getAllByIds(validation_docs_ids, 
                                           convert=lambda doc_id, words: self.calculateSelectedWordWeights(words, validation_set[doc_id]))
                    validation_set_rmse = self.calculateError(validation_docs)
                    rmse_results += ", validation set RMSE: " + str(validation_set_rmse)

                    positive_validation_docs_ids = positive_validation_set.keys()
                    positive_validation_docs = self.document_iterator.getAllByIds(positive_validation_docs_ids, 
                                           convert=lambda doc_id, words: self.calculateSelectedWordWeights(words, positive_validation_set[doc_id]))
                    positive_validation_set_rmse = self.calculateError(positive_validation_docs)
                    rmse_results += ", positive validation set RMSE: " + str(positive_validation_set_rmse)


                print rmse_results

                if self.tester:
                    start_time = time.time()
                    self.tester(epoch, training_set_rmse, validation_set_rmse, positive_validation_set_rmse)
                    print "Test: " + str(time.time() - start_time)

                
                #if total_rmse > prev_rmse1 and prev_rmse1 > prev_rmse2:
                #    break
                prev_rmse2 = prev_rmse1
                prev_rmse1 = training_set_rmse
            

            if num_iter is not None and epoch >= num_iter:
                break 

            epoch += 1

    def saveToFile(self, save_words):
        """
        Serializes the model to an external file.
        The document profiles are automatically saved to the db during updates,
        only remaining model part, i.e. word profile matrix, global statistics (e.g word counts) are serialized to an external file
        :return:
        """

        #if self.current_document_batch is not None:
        #    self.document_iterator.saveDocumentProfilesToDb(self.current_document_batch)

        if not save_words:
            return

        with open(self.file_name, 'w') as f:
            """
            TODO: add ALL model parameters to the first line
            """
            f.write(str(self.num_features) + '\t' + str(len(self.token_to_id)) + '\t' + str(self.last_active_id + 1) + '\t' + \
                str(self.num_docs) + '\t' + \
                str(self.min_df) + '\t' + str(self.max_df) + '\n')

            for token, local_token_id in self.token_to_id.iteritems():
                word_stats_line = token + '\t'
                word_stats_line += str(self.doc_freqs[local_token_id]) + '\t'
                #word_stats_line += str(int(local_token_id in self.active_tokens)) + '\t'
                word_stats_line += str(int(local_token_id <= self.last_active_id))

                if local_token_id <= self.last_active_id:
                    word_profile = self.word_profiles[local_token_id, :]
                    str_word_profile = ','.join(str(profile_element) for profile_element in word_profile)
                    word_stats_line += '\t' + str_word_profile

                word_stats_line += '\n'

                f.write(word_stats_line)

    def save(self, save_words, word_ids=None):
        if not self.save_model: return
        if self.save_to_db:
            self.saveToDb(save_words, word_ids)
        else:
            self.saveToFile(save_words)

    def saveToDb(self, save_words, word_ids=None):
        if self.current_document_batch is not None:
            self.document_iterator.saveDocumentProfilesToDb(self.current_document_batch)

        if not save_words:
            return

        with open(self.file_name, 'w') as f:
            """
            TODO: add ALL model parameters to the first line
            """
            f.write(str(self.num_features) + '\t' + str(len(self.token_to_id)) + '\t' + str(self.last_active_id + 1) + '\t' + \
                str(self.num_docs) + '\t' + \
                str(self.min_df) + '\t' + str(self.max_df) + '\n')

        current = 0

        sql_insert = "INSERT INTO pap_words (word, df, is_active, profile) VALUES "
        sql_update = " ON DUPLICATE KEY UPDATE df = VALUES(df), is_active = VALUES(is_active), profile = VALUES(profile)"

        sql_tuples = []

        if not word_ids: word_ids = xrange(self.last_active_id + 1)

        for local_token_id in word_ids:
            token = self.id_to_token[local_token_id]
            df = self.doc_freqs[local_token_id]
            #is_active = int(local_token_id in self.active_tokens)
            is_active = int(local_token_id <= self.last_active_id)

            if local_token_id <= self.last_active_id:
                profile = ','.join(str(profile_element) for profile_element in self.word_profiles[local_token_id, :])
                sql_tuples.append("('{0}', {1}, {2}, '{3}')".format(token, df, is_active, profile))
            else:
                sql_tuples.append("('{0}', {1}, {2}, NULL)".format(token, df, is_active))

            current += 1
            if current % 10000 == 0:
                print current
                db.query(sql_insert + ','.join(sql_tuples) + sql_update)
                db.commit()
                sql_tuples = []

        if len(sql_tuples) > 0:
            db.query(sql_insert + ','.join(sql_tuples) + sql_update)
            db.commit()

        return #saving word - document mappings disabled temporarily

        sql_insert = "INSERT INTO pap_word_documents (word, document_id) VALUES "
        sql_update = " ON DUPLICATE KEY UPDATE word = VALUES(word), document_id = VALUES(document_id)"
        sql_tuples = []
        current = 0        

        for word_id, document_ids in self.word_id_to_document_ids.iteritems():
            word = self.id_to_token[word_id]
            for document_id in document_ids:
                sql_tuples.append("('{0}', {1})".format(word, document_id))

                current += 1
                if current % 10000 == 0:
                    print current
                    db.query(sql_insert + ','.join(sql_tuples) + sql_update)
                    db.commit()
                    sql_tuples = []


        if len(sql_tuples) > 0:
            db.query(sql_insert + ','.join(sql_tuples) + sql_update)
            db.commit()

        self.word_id_to_document_ids.clear()

    @staticmethod
    def load(file_name, document_iterator, word_profiles_in_db=True):
        """
        :param file_name: serialized model file name
        :return: SemanticModel based on the serialized data
        """

        semantic_model = None

        with open(file_name, 'r') as f:
            snapshot_reader = SemanticModelSnapshotReader(f, db, word_profiles_in_db)

            num_features, num_words, num_active_words, num_docs, min_df, max_df = snapshot_reader.readGeneralStats()

            semantic_model = SemanticModel(document_iterator=document_iterator, num_features=num_features, file_name=file_name)
            semantic_model.num_docs = num_docs
            semantic_model.word_profiles = numpy.zeros((num_active_words, num_features))
            semantic_model.min_df = min_df
            semantic_model.max_df = max_df
            semantic_model.preanalyze_documents = False
            #semantic_model.active_tokens = SortedSet()
            semantic_model.last_active_id = num_active_words - 1

            inactive_words = []
            #read all words and process active ones
            for token, word_doc_freqs, is_active, word_profile in snapshot_reader.readWordProfiles():
                if is_active:
                    semantic_model.token_to_id[token] = len(semantic_model.token_to_id) 
                    semantic_model.id_to_token[semantic_model.token_to_id[token]] = token               
                    semantic_model.doc_freqs[semantic_model.token_to_id[token]] = word_doc_freqs
                    semantic_model.word_profiles[semantic_model.token_to_id[token], :] = word_profile
                else:
                    inactive_words.append((token, word_doc_freqs))
            
            #process inactive words
            for token, word_doc_freqs in inactive_words:
                semantic_model.token_to_id[token] = len(semantic_model.token_to_id)                
                semantic_model.id_to_token[semantic_model.token_to_id[token]] = token
                semantic_model.doc_freqs[semantic_model.token_to_id[token]] = word_doc_freqs    

        return semantic_model

    def loadJSON(self):
        model_data = None
        with open(file_name, 'r') as fp:
            model_data = json.load(fp)

        num_features = model_data['num_features']
        semantic_model = SemanticModel(num_features=num_features, file_name=file_name)

        word_profiles = model_data['word_profiles']        
        semantic_model.word_profiles = numpy.zeros((num_features, len(word_profiles)))
        for token, word_profile in word_profiles.iteritems():
            semantic_model.token_to_id[token] = len(semantic_model.token_to_id)
            semantic_model.word_profiles[:, semantic_model.token_to_id[token]] = numpy.asarray(word_profile)

        doc_freqs = model_data['doc_freqs']
        for token, doc_freqs in doc_freqs.iteritems():
            semantic_model.doc_freqs[semantic_model.token_to_id[token]] = doc_freqs

        return semantic_model


class SemanticModelSnapshotReader(object):
    def __init__(self, file_object, db, word_profiles_in_db=True):
        self.file_object = file_object
        self.db = db
        self.first_line_read = False
        self.word_profiles_in_db = word_profiles_in_db

    def isint(self, x):
        try:
            a = float(x)
            b = int(a)
        except ValueError:
            return False
        else:
            return a == b

    def readGeneralStats(self):
        if self.first_line_read:
            self.file_object.seek(0)
            self.first_line_read = False

        first_line_stats = self.file_object.readline()
        first_line_stats = first_line_stats.split('\t')

        num_features = int(first_line_stats[0])
        num_words = int(first_line_stats[1])
        num_active_words = int(first_line_stats[2])
        num_docs = int(first_line_stats[3])
        min_df_str = first_line_stats[4]
        max_df_str = first_line_stats[5]
        min_df = int(min_df_str) if self.isint(min_df_str) else float(min_df_str)
        max_df = int(max_df_str) if self.isint(max_df_str) else float(max_df_str)
        
        self.first_line_read  = True

        return num_features, num_words, num_active_words, num_docs, min_df, max_df

    def readWordProfiles(self):
        profile_generator = self.readWordProfilesFromDb if self.word_profiles_in_db else self.readWordProfilesFromFile
        
        for result in profile_generator():
            yield result

    def readWordProfilesFromFile(self):
        if not self.first_line_read:
            self.readGeneralStats()

        for word_stats in self.file_object:
            split_word_stats = word_stats.split('\t')
            token = split_word_stats[0]
            word_doc_freqs = int(split_word_stats[1])
            is_active = bool(int(split_word_stats[2]))
            if is_active:
                word_profile = split_word_stats[3]
                word_profile = numpy.asarray([float(value) for value in word_profile.split(',')])
            else:
                word_profile = None

            yield token, word_doc_freqs, is_active, word_profile

    def readWordProfilesFromDb(self):
        if not self.first_line_read:
            self.readGeneralStats()

        sql = "SELECT word, df, is_active, profile FROM pap_words"

        for word_stats in db.select(sql):
            token = word_stats['word']
            word_doc_freqs = int(word_stats['df'])
            is_active = ord(word_stats['is_active'])
            if is_active:
                word_profile = word_stats['profile']
                word_profile = numpy.asarray([float(value) for value in word_profile.split(',')])
            else:
                word_profile = None

            yield token, word_doc_freqs, is_active, word_profile

    def printOrderedWordProfiles(self, limit=1000, order='desc', min_freq=1, max_freq=sys.maxint, only_active=True):
        words = [(token, freq, is_active, profile)
            for token, freq, is_active, profile in self.readWordProfiles() 
            if freq >= min_freq and freq <= max_freq and is_active == only_active]

        if order == 'desc':
            words.sort(key=operator.itemgetter(1), reverse=True)
        else:
            words.sort(key=operator.itemgetter(1), reverse=False)

        print "Word count: " + str(len(words))

        i = limit
        for token, doc_freq, is_active, profile in words:
            print i, token, doc_freq, is_active, profile

            i -= 1
            if i == 0:
                break

    def showWordFrequencies(self):
        import matplotlib.pyplot as plt

        freqs = [freq for token, freq, is_active, profile in self.readWordProfiles()]
        print freqs
        plt.yscale('log', nonposy='clip')
        plt.hist(freqs, bins=range(500))
        plt.show()

if __name__ == "__main__":
    file_name = 'semantic_model.snapshot'
    
    try:
        """
        document_iterator = DocumentIterator()
        docs_with_given_text = [doc for doc in document_iterator.getAll() if 'computer' in doc.tokenized_text]
        rawtexts = [doc.rawtext for doc in docs_with_given_text]
        categories = [doc.category for doc in docs_with_given_text]
        print categories
        print str(len(docs_with_given_text)) + '\n'
        print rawtexts[0]
        """

        #semantic_model = SemanticModel.load(file_name, is_test=False)
        #semantic_model.min_df=0.003
        #semantic_model.max_df=0.5
        #semantic_model.preanalyze_documents=False
        

        #document_iterator = DocumentIterator(is_test=True)
        #document_iterator.saveDocumentProfilesToFile(file_name='document_profiles.train')
        
        
        #all_documents = document_iterator.getAll()

        #semantic_model.assignUntrainedDocumentProfiles(all_documents, num_iters=50, initialize_profiles=True)
        #document_iterator.saveDocumentProfilesToDb(all_documents)


        #semantic_model = SemanticModel(num_features=8, file_name=file_name, doc_filter="published = 1 and is_test = 0", min_df=0.003, max_df=0.5)
        #semantic_model.train()

        with open(file_name, 'r') as f:
            snapshot_reader = SemanticModelSnapshotReader(f, db)
            #print snapshot_reader.readGeneralStats()
            #snapshot_reader.printOrderedWordProfiles(order='desc', min_freq=0, max_freq=sys.maxint, only_active=True)
            snapshot_reader.showWordFrequencies()
    except (KeyboardInterrupt, SystemExit):
        #semantic_model.save()
        print "Saved"
        raise
    finally:
        db.disconnect()
    
