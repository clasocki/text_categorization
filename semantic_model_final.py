import numpy
from collections import defaultdict
import json
import operator
import math
import sys
import numbers
import time
import itertools
import numba
import random
from nltk.corpus import stopwords
import re

STOP = set(stopwords.words('english'))

def tokenize(text):
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
    
    def saveDocumentProfilesToFile(self, documents, file_name):
        with open(file_name, 'w') as f:
            for document in documents:
                str_doc_profile = str(document.id) + '\t'
                str_doc_profile += ','.join(str(profile_element) for profile_element in document.profile) + '\n'
                f.write(str_doc_profile)

#@profile
@numba.jit(nopython=True,cache=True)
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

    return squared_error, num_values

@numba.jit(nopython=True,cache=True)
def inferProfilesPerDocument2(word_weights, document_profiles, word_profiles, learning_rate, 
                             regul_factor, update_document_profiles, update_word_profiles):
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

    return squared_error, num_values

@numba.jit(nopython=True,cache=True)
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
                                 regularization_factor, update_document_profiles=True, update_word_profiles=False)

        return profile

    #@profile
    def inferProfiles(self, documents, epoch, update_document_profiles=True, update_word_profiles=True, initialize_document_profiles=False, num_iters=1):
        if initialize_document_profiles:
            documents = self.initializeDocumentProfiles(documents)  

        if num_iters > 1:
            documents = list(documents)

        for current_iter in xrange(num_iters):
            for document in documents:
                inferProfilesPerDocument(document.word_weights, document.profile, self.word_profiles, 
                    self.adaptiveLearningRate(self.decay, self.learning_rate, epoch), self.regularization_factor, 
                    update_document_profiles, update_word_profiles)
            
        return (document.profile for document in documents)
    
    def inferProfiles2(self, doc_profiles, word_weights, doc_ids, epoch, update_document_profiles=True, update_word_profiles=True, initialize_document_profiles=False, num_iters=1):

        for current_iter in xrange(num_iters):
            inferProfilesPerDocument2(word_weights, doc_profiles, self.word_profiles, 
                self.adaptiveLearningRate(self.decay, self.learning_rate, epoch), self.regularization_factor, 
                update_document_profiles, update_word_profiles)

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

        positive_sample_size = int(numpy.floor(0.2 * len(positive_weights)))
        zero_sample_size = int(numpy.floor(self.neg_weights * positive_sample_size))
       
        print len(positive_weights), positive_sample_size
        positive_sample = numpy.random.choice(len(positive_weights), positive_sample_size, replace=False)
        zero_sample = numpy.random.choice(total_training_set_size, zero_sample_size, replace=False)

        doc_id = lambda i: i / num_words
        word_id = lambda i: i % num_words

        validation_set = defaultdict(set)
        positive_validation_set = defaultdict(set)
       
        for i in positive_sample:
            validation_set[positive_weights[i][0]].add(positive_weights[i][1])
            positive_validation_set[positive_weights[i][0]].add(positive_weights[i][1])
         
        for i in zero_sample:
            validation_set[doc_id(i)].add(word_id(i))
       
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
    
    def updateStatisticsForNewDocument(self, document):
        self.num_docs += 1

        processed_tokens = set()
        for token in document.tokenized_text:
            if len(token) > 100:
                continue
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
                self.id_to_token[self.token_to_id[token]] = token

            if token not in processed_tokens:
                token_id = self.token_to_id[token]
                self.doc_freqs[token_id] += 1
                self.word_id_to_document_ids[token_id].append(document.id)
                processed_tokens.add(token)

        #print "The number of tokens: " + str(len(self.token_to_id))

    def updateStatisticsForNewDocuments(self, documents):
        num_tokens_before_update = len(self.token_to_id)
        
        for document in documents:
            self.updateStatisticsForNewDocument(document)

        num_tokens_after_update = len(self.token_to_id)
        num_new_tokens = num_tokens_after_update - num_tokens_before_update

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
            return lambda bow, raw_freq: 1 + math.log(raw_freq) if raw_freq > 0 else 0
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
    def limit_words(self):
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
                    added_ids.add(token_id)
                else:
                    kept_ids.add(token_id)
            else:
                if token_id <= self.last_active_id:
                    removed_ids.add(token_id)

        id_map = self.compactify(added_ids, removed_ids)
        kept_ids = numpy.fromiter((id_map[kept_id] for kept_id in kept_ids), numpy.long)
        added_ids = numpy.fromiter((id_map[added_id] for added_id in added_ids), numpy.long)
        removed_ids = numpy.fromiter((id_map[removed_id] for removed_id in removed_ids), numpy.long)

        return kept_ids, added_ids, removed_ids

    def calculateTfIdf(self, words, word_filter=lambda word_id: True):
        bag_of_words = defaultdict(int)
        for word in words:
            if word in self.token_to_id:
                word_id = self.token_to_id[word]

                if word_id <= self.last_active_id:
                    bag_of_words[word_id] += 1
         
        word_weights = dict((word_id, self.tf(bag_of_words, raw_freq) * self.idf(self.num_docs, word_id, self.doc_freqs)) 
                                  for word_id, raw_freq in bag_of_words.iteritems() if word_filter(word_id))

        return word_weights
    
    def calculateRandomizedWordWeights(self, words, word_filter=lambda w_id: True, zero_weights=None):
        word_weights = self.calculateTfIdf(words, word_filter) 

        return self.randomizeWordWeights(word_weights, word_filter, zero_weights)

    def calculateSelectedWordWeights(self, words, selected_word_ids):
        word_weights = self.calculateTfIdf(words, word_filter=lambda word_id: word_id in selected_word_ids)
        for word_id in selected_word_ids:
            if word_id not in word_weights:
                word_weights[word_id] = 0

        return list(word_weights.iteritems()) 

    def randomizeWordWeights(self, word_weights, word_filter, zero_weights=None):
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

        numpy.random.shuffle(weights)
        return weights

    #@profile
    def train(self, num_iter=None):
        """
        Main loop that iterates over the db, periodically getting new document batches.
        Starts using empty profile matrices, incrementally resizing them in the process of model training.
        :return:
        """
        epoch = 0
        prev_rmse1 = sys.maxint
        prev_rmse2 = sys.maxint
        doc_count = 0
        all_documents = []
        kept_indices, added_indices, removed_indices = None, None, None

        if self.preanalyze_documents:
            all_documents = self.document_iterator.getAll(convert=None)
            doc_count = len(all_documents)
            self.updateStatisticsForNewDocuments(all_documents)
            kept_indices, added_indices, removed_indices = self.limit_words()
            print "Kept: " + str(len(kept_indices)) + ", added: " + str(len(added_indices)) + ", removed: " + str(len(removed_indices))
            
        positive_validation_set, validation_set = self.createProportionateValidationSet()
        train_word_filter = (lambda doc_id, word_id: word_id not in validation_set[doc_id]) if validation_set else (lambda doc_id, word_id: True)
        train_doc_converter = lambda doc_id, words: self.calculateRandomizedWordWeights(words, word_filter=lambda word_id: train_word_filter(doc_id, word_id))

        if self.preanalyze_documents:
            for document in all_documents:
                document.word_weights = train_doc_converter(document.id, document.tokenized_text)

            self.initializeDocumentProfiles(all_documents)
            self.current_document_batch = all_documents
            self.save(save_words=True, word_ids=itertools.chain(added_indices, removed_indices))
        #while (epoch < self.min_iter or rmse_last - rmse >= self.min_improvement) and epoch < self.max_iter:
        #while num_iter is None or epoch < num_iter:
        #while True:
        for document_batch in self.document_iterator.batchIter(convert=train_doc_converter):
            self.current_document_batch = document_batch

            start_time = time.time()
            self.inferProfiles(document_batch, epoch)
            inference_time = time.time() - start_time
            
            start_time = time.time()
            self.save(save_words=(epoch % self.save_frequency == 0 and epoch != 0), word_ids=xrange(self.last_active_id + 1))

            saving_time = time.time() - start_time
            print "Iter: %s, Save: %s, Infer: %s" % (str(epoch), str(saving_time), str(inference_time))
               
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
        #if self.current_document_batch is not None:
        #    self.document_iterator.saveDocumentProfilesToFile(self.current_document_batch, self.file_name + '.doc_profiles.snapshot')

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
            TODO: save ALL model parameters
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

