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

#numpy.seterr(all='warn')
#warnings.filterwarnings('error')

class DocumentIterator(object):
    def __init__(self, where, document_batch_size=300, db_window_size=300):
        self.current_record_offset = 0
        self.DOCUMENT_BATCH_SIZE = document_batch_size
        self.DB_WINDOW_SIZE = db_window_size
        self.where = where

    def getRandomDocumentsFromDb(self):
        query = "SELECT * FROM " + \
            "((SELECT * " + \
            "FROM pap_papers_view WHERE " + \
            self.where + " " + \
            "LIMIT " + str(self.DB_WINDOW_SIZE) + " OFFSET " + str(self.current_record_offset) + ")"

        db_row_count = db.select("SELECT COUNT(1) FROM pap_papers_view WHERE " + self.where)[0]['COUNT(1)']

        remaining_batch_size = self.DB_WINDOW_SIZE - db_row_count + self.current_record_offset
        
        if db_row_count > 0:
            self.current_record_offset = (self.current_record_offset + self.DB_WINDOW_SIZE) % db_row_count

        if remaining_batch_size > 0:
            query += " UNION " + \
            "(SELECT * " + \
            "FROM pap_papers_view WHERE " + \
            self.where + " " + \
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
        db_row_count = db.select("SELECT COUNT(1) FROM pap_papers_view WHERE " + self.where)[0]['COUNT(1)']
        
        while self.current_record_offset < db_row_count:
            print self.current_record_offset
            
            query = "SELECT * " + \
                "FROM pap_papers_view WHERE " + self.where + " " + \
                "LIMIT " + str(self.DB_WINDOW_SIZE) + " OFFSET " + str(self.current_record_offset)

            document_batch = Paper.selectRaw(query)

            for document in document_batch:
                document.tokenized_text = tokenize(document.rawtext).split()
                document.profile = numpy.asarray(document.profile)

            yield document_batch

            self.current_record_offset += self.DB_WINDOW_SIZE

    def getAll(self):
        query = "SELECT * FROM pap_papers_view WHERE " + self.where

        all_documents = Paper.selectRaw(query)
        for document in all_documents:
            document.tokenized_text = tokenize(document.rawtext).split()
            document.profile = numpy.asarray(document.profile)

        return all_documents

    @staticmethod
    def getAll2():
        query = "SELECT * FROM pap_papers_view WHERE published = 1"

        all_documents = Paper.selectRaw(query)
        for document in all_documents:
            document.tokenized_text = tokenize(document.rawtext).split()
            document.profile = numpy.asarray(document.profile)

        return all_documents

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

            #sql_update = "UPDATE pap_papers_view SET profile = '" + \
            sql_update = "UPDATE pap_papers_3 SET profile = '" + \
                str_document_profile + \
                "' WHERE id = " + str(db_document_id)

            db.query(sql_update)
            #document.update()

        db.commit()

    def __iter__(self):
        while True:
            document_batch = self.getRandomDocumentsFromDb()
            if not document_batch:
                break

            for document in document_batch:
                document.tokenized_text = tokenize(document.rawtext).split()
                document.profile = numpy.asarray(document.profile)

            yield [doc for doc in document_batch if len(doc.tokenized_text) > 0]

class TermFrequencyWeight(Enum):
    RAW_FREQUENCY = 1
    LOG_NORMALIZATION = 2
    AUGMENTED_FREQUENCY = 3
    BINARY = 4

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

@numba.jit(nopython=True,cache=True)
def numbaInferProfile(word_id, weight, document_profile, word_profiles, learning_rate, regul_factor, update_document_profiles, update_word_profiles):
    #try:
    predicted_value = numpy.dot(document_profile, word_profiles[word_id, :])
    error = 1.0 * weight - predicted_value
    original_document_profile = numpy.copy(document_profile)

    if update_document_profiles:
        document_profile += \
            learning_rate * (error * word_profiles[word_id, :] - regul_factor * original_document_profile) 

    if update_word_profiles:
        word_profiles[word_id, :] += \
            learning_rate * (error * original_document_profile - regul_factor * word_profiles[word_id, :])
    
    return error
    #except:
    #    print predicted_value, document_profile, word_id, word_profiles[word_id, :]

class SemanticModel(object):
    MAX_UPDATE_ITER = 1
    REGULARIZATION_FACTOR = 0.01
    LEARNING_RATE = 0.005

    def __init__(self, num_features, file_name, where,
        term_freq_weight=TermFrequencyWeight.LOG_NORMALIZATION, use_idf = True,
        min_df=0.0, max_df=1.0, limit_features=True,
        preanalyze_documents=True, tester=None, save_frequency=40, test_frequency=40):
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
        self.use_idf = use_idf
        self.min_df = min_df
        self.max_df = max_df
        self.limit_features = limit_features
        self.preanalyze_documents = preanalyze_documents
        self.tester = tester
        self.save_frequency = save_frequency
        self.test_frequency = test_frequency

        self.word_profiles = numpy.random.uniform(low=-0.01, high=0.01, size=(1, self.num_features))
        self.document_iterator = DocumentIterator(where=where)
        self.token_to_id = dict() # token -> token id
        self.id_to_token = dict()
        self.doc_freqs = defaultdict(int)  # token id -> the number of documents this token appears in
        self.last_active_id = -1 #describes the last active id
        self.updated_word_ids = set()
        self.word_id_to_document_ids = defaultdict(list)
        #self.active_tokens = SortedSet() # set of tokens that aren't too rare or too common, 
                                   # i.e. they exist in (for float) 
                                   # [min_df * self.num_docs, max_df * self.num_docs] or (for int) [min_df, max_df] documents 
        self.current_document_batch = None  
   
    #@profile
    def inferProfile(self, document, update_word_profiles):
        addToUpdatedWordIds = self.updated_word_ids.add
        for word_id, value in document.full_converted_text:
            predicted_value = predictValue(document.profile, self.word_profiles, word_id)
            error = 1.0 * value - predicted_value

            #document.profile += \
            #    self.LEARNING_RATE * (error * self.word_profiles[word_id, :] - self.REGULARIZATION_FACTOR * document.profile)
            
            updateDocProfile(document.profile, self.LEARNING_RATE, error, self.word_profiles, word_id, self.REGULARIZATION_FACTOR) 
            if update_word_profiles:
                addToUpdatedWordIds(word_id)
                updateWordProfile(document.profile, self.LEARNING_RATE, error, self.word_profiles, word_id, self.REGULARIZATION_FACTOR)               
                #self.word_profiles[word_id, :] += \
                #    self.LEARNING_RATE * (error * document.profile - self.REGULARIZATION_FACTOR * self.word_profiles[word_id, :])
         
        #print "sub: " + str(document.profile)
 
    #@profile
    def inferProfiles(self, documents, update_document_profiles=True, update_word_profiles=True, initialize_document_profiles=False, 
                      initialize_word_profiles=False, num_iters=MAX_UPDATE_ITER, selected_word_ids=None, print_stats=False):
        """
        Calculates profiles for documents
        :param documents: a list of unconverted documents
        :return:
        """    

        start_time = time.time()

        if initialize_document_profiles:
            documents = self.initializeDocumentProfiles(documents)  
    
        if initialize_word_profiles and selected_word_ids is not None:
            self.initializeWordProfiles(selected_word_ids, len(selected_word_ids))

        documents = self.convertDocuments(documents)
        if num_iters > 1:
            documents = list(documents)

        current_iter = 0
        addToUpdatedWordIds = self.updated_word_ids.add

        for current_iter in xrange(num_iters):
            squared_error = 0.0
            num_values = 0

            #active_tokens_sample = numpy.random.choice(len(self.active_tokens), 400, replace=False)
            active_tokens_sample = None
            #print [d.id for d in documents]
            #for word_id, weight, document_profile in (id_weight_pair  
            #    for document in documents
            #        for id_weight_pair in document.full_converted_text):
            #profile_before = numpy.copy(documents[0].profile)

            for document in documents:
                for word_id, weight in document.full_converted_text:
                    if selected_word_ids is not None and word_id not in selected_word_ids:
                        continue
                    #print self.word_profiles[word_id, :], self.id_to_token[word_id], document.profile, document.id, document.pid
                    #doc_prof_before = numpy.copy(document.profile)
                    #word_prof_before = numpy.copy(self.word_profiles[word_id, :])
                    try: 
                        error = numbaInferProfile(word_id, weight, document.profile, self.word_profiles, self.LEARNING_RATE, self.REGULARIZATION_FACTOR, 
                                    update_document_profiles, update_word_profiles)
                        squared_error += error * error
                        num_values += 1
                        if update_word_profiles:
                            addToUpdatedWordIds(word_id)
                    except:
			print self.id_to_token[word_id]
                        #print document.id, document.profile, self.id_to_token[word_id], word_id, self.word_profiles[word_id, :]
            
            #profile_after = documents[0].profile
            rmse = numpy.sqrt(squared_error / num_values)
    
            if print_stats:
                print "Partial RMSE: " + str(rmse) + ", num words: " + str(len(documents[0].full_converted_text))
            #print "Partial RMSE: " + str(rmse) + ", num words: " + str(len(documents[0].full_converted_text)) + ", num values: " + str(num_values) + ", profile diff: " + str(numpy.linalg.norm(profile_after - profile_before))
            #print "Before: " + str(profile_before)
            #print "After: " + str(profile_after)

            #for document in documents:
            #    self.inferProfile(document, update_word_profiles)
            #map(lambda doc: self.inferProfile(doc, update_word_profiles), documents)
            """"
            for document in documents: 
                for word_id, value in self.fullDictionaryBasedDocument(document, active_tokens_sample):
                #for word_id, value in itertools.chain(weight_updates, document.converted_text.iteritems()):
                    num_values += 1

                    predicted_value = predictValue(document.profile, self.word_profiles, word_id)
                    error = 1.0 * value - predicted_value
                    squared_error += error * error

                    #document.profile += \
                    #    self.LEARNING_RATE * (error * self.word_profiles[word_id, :] - self.REGULARIZATION_FACTOR * document.profile)
                    
                    updateDocProfile(document.profile, self.LEARNING_RATE, error, self.word_profiles, word_id, self.REGULARIZATION_FACTOR)               
                    
                    if update_word_profiles:
                        self.updated_word_ids.add(word_id)
                        updateWordProfile(document.profile, self.LEARNING_RATE, error, self.word_profiles, word_id, self.REGULARIZATION_FACTOR)               
                    #self.word_profiles[word_id, :] += \
                    #    self.LEARNING_RATE * (error * document.profile - self.REGULARIZATION_FACTOR * self.word_profiles[word_id, :])
            """
            #print self.token_to_id.keys()
            #rmse = numpy.sqrt(squared_error / num_values)
            #print "Partial RMSE: " + str(rmse)
            #print "Num operations: " + str(num_operations)
        if print_stats:
            print "Infer profiles: " + str(time.time() - start_time)
        
        return (document.profile for document in documents)

    def calculateError(self, documents):
        squared_error = 0.0
        num_values = 0

        for document in documents:
            for word_id, value in document.full_converted_text:
                num_values += 1
                predicted_value = predictValue(document.profile, self.word_profiles, word_id)
                error = 1.0 * value - predicted_value
                squared_error += error * error

        rmse = numpy.sqrt(squared_error / num_values)

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
        self.inferProfiles(documents, update_document_profiles=False, initialize_word_profiles=True, num_iters=num_iters_partial_retrain, print_stats=True)
        self.updateStatisticsForNewDocuments(documents)       
        # czy w tym momencie trzeba zapisac statystyki mapowan miedzy slowami a dokumentami
        # ponizsze obliczenia bazuja teraz tylko na starych danych 
        print "Limiting the number of words..."
        kept_word_ids, added_word_ids, removed_word_ids = self.limit_words()
        print "Kept: " + str(len(kept_word_ids)) + ", added: " + str(len(added_word_ids)) + ", removed: " + str(len(removed_word_ids))

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
        
           
        sys.exit()
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
            yield DocumentIterator(where=where).getAll()
 
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

        self.updated_word_ids.update(xrange(num_tokens_before_update, num_new_tokens))
        #self.compactify(added_ids=range(tokens_before_update, num_new_tokens), removed_ids=[])

    def initializeDocumentProfile(self, document):
        document.profile = numpy.random.uniform(low=-0.01, high=0.01, size=self.num_features)

        return document

    def initializeDocumentProfiles(self, documents):
        for document in documents:
            yield self.initializeDocumentProfile(document)
        
    def initializeWordProfiles(self, word_ids, num_words):
        new_word_profiles = numpy.random.uniform(low=-0.01, high=0.01, 
                size=(num_words, self.num_features))
        for new_profile_id, word_id in enumerate(word_ids):
            self.word_profiles[word_id, :] = new_word_profiles[new_profile_id, :]

    def tf(self, bag_of_words, token_id):
        raw_frequency = bag_of_words[token_id]    
        if self.term_freq_weight == TermFrequencyWeight.RAW_FREQUENCY:
            return raw_frequency
        elif self.term_freq_weight == TermFrequencyWeight.LOG_NORMALIZATION:
            return 1 + (math.log(raw_frequency) if raw_frequency > 0 else 0)
        elif self.term_freq_weight == TermFrequencyWeight.AUGMENTED_FREQUENCY:
            return 0.5 + (0.5 * raw_frequency / max(bag_of_words.values()))
        elif self.term_freq_weight == TermFrequencyWeight.BINARY:
            return 1 if raw_frequency > 0 else 0 
        return 1.0 * token_freq / tf_sum

    def idf(self, token_id):
        total_docs = self.num_docs
        doc_freq = self.doc_freqs[token_id]

        return math.log(1 + (1.0 * total_docs / doc_freq)) if doc_freq > 0 else 1

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
        self.updated_word_ids = set(id_map[token_id] for token_id in self.updated_word_ids)
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

        self.updated_word_ids.update(added_ids)
        self.updated_word_ids.update(removed_ids)

        id_map = self.compactify(added_ids, removed_ids)
        kept_ids = numpy.fromiter((id_map[kept_id] for kept_id in kept_ids), numpy.long)
        added_ids = numpy.fromiter((id_map[added_id] for added_id in added_ids), numpy.long)
        removed_ids = numpy.fromiter((id_map[removed_id] for removed_id in removed_ids), numpy.long)

        return kept_ids, added_ids, removed_ids

    def convertDocuments(self, documents):
        """
        Tokenizes and converts each document from the raw text form to the bag-of-words format
        :param documents:
        :return: list of documents in the bag-of-words format with a local id assigned
        """
        for document in documents:
            tokenized_text = document.tokenized_text

            bag_of_words = defaultdict(int)
            for token in tokenized_text:
                if token not in self.token_to_id:
                    continue
                
                token_id = self.token_to_id[token]

                if token_id <= self.last_active_id:
                #if token_id in self.active_tokens:
                    bag_of_words[token_id] += 1

            converted_text = dict()
            for token_id in bag_of_words.keys():
                token_weight = self.tf(bag_of_words, token_id) * (self.idf(token_id) if self.use_idf else 1)
                converted_text[token_id] = token_weight

            document.converted_text = converted_text
            document.full_converted_text = self.fullConvertedText(converted_text)

            yield document

        #return documents

    def fullConvertedText(self, converted_text):
        """
        :param document: document represented as a map: token_id -> token_weight where token_id exists in the document
        :return: document represented as an iterator returning tuples: (token_id, token_weight)
                 token_weight != 0 if token exists in the document
                 token_weight  = 0 otherwise
        """

        """
        slowa sa dobrze dobierane, 
        bo active_tokens_sample zawiera identifikatory z zakresu [0, wielkosc active_tokens_sample],
        SortedSet (active_tokens) jest indeksowany w posortowanej kolejnosci!
        """
        
        positive_weight_count = len(converted_text)
        #all_active_tokens_count = len(self.active_tokens)
        #print "Last active id: %s, positive weight count: %s" % (self.last_active_id, positive_weight_count)
        sample_size = min(self.last_active_id + 1, int(math.ceil(positive_weight_count * 3.0)))
        active_tokens_sample_ids = numpy.random.choice(self.last_active_id + 1, sample_size, replace=False)

        #active_tokens_sample = numpy.random.choice(all_active_tokens_count, int(math.ceil(positive_weight_count * 3.0)), replace=False)
        #active_tokens_sample = numpy.random.choice(xrange(all_active_tokens_count), 400, replace=False)
        #active_tokens_sample = self.active_tokens.keys()
        #print "Positive weight count: " + str(positive_weight_count)
        
        """
        weights = []
        weights = [(self.active_tokens[sample_token_id], 0) 
            for sample_token_id in active_tokens_sample 
            if self.active_tokens[sample_token_id] not in document.converted_text]
        """
        weights = ((token_id, 0) 
            for token_id in active_tokens_sample_ids 
            if token_id not in converted_text)

        #weights.extend(document.converted_text.iteritems())

        #numpy.random.shuffle(weights)

        return list(itertools.chain(weights, converted_text.iteritems()))

        """

        for token_id in numpy.random.shuffle(document.converted_text.)

        for token_id in self.active_tokens:
            if token_id in document.converted_text:
                yield token_id, document.converted_text[token_id]
            else:
                yield token_id, 0

        """

    def splitDocuments(self, documents):
        """
        Checks whether each document has already been trained and 
        assigns it to a proper group (new document vs old document)
        :param documents:
        :return: list of documents that has not yet been trained, list of documents already known by the model
        """
        new_documents = []
        old_documents = []

        for document in documents: 
            if not document.profile.any():
                new_documents.append(document)
            else:
                old_documents.append(document)

        return new_documents, old_documents

    def resizeProfileMatrices(self, idmap):
        """
        idmap - map between old ids and new ids
        """
        (_, word_profile_count) = self.word_profiles.shape


        """
        while len(self.token_to_id) > word_profile_count:
            self.word_profiles = numpy.resize(self.word_profiles,
                (self.num_features, word_profile_count * 2))
            self.word_profiles[:, word_profile_count:] = numpy.random.uniform(low=-0.01, high=0.01, 
                size=(self.num_features, word_profile_count))

            (_, word_profile_count) = self.word_profiles.shape
        """ 
    
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

        if self.preanalyze_documents:
            all_documents = self.document_iterator.getAll()
            self.current_document_batch = all_documents
            all_documents = self.initializeDocumentProfiles(all_documents)
            self.updateStatisticsForNewDocuments(all_documents)
            kept_indices, added_indices, removed_indices = self.limit_words()
            print "Kept: " + str(len(kept_indices)) + ", added: " + str(len(added_indices)) + ", removed: " + str(len(removed_indices))
            self.save(save_words=True)

        #while (epoch < self.min_iter or rmse_last - rmse >= self.min_improvement) and epoch < self.max_iter:
        #while num_iter is None or epoch < num_iter:
        #while True:
        for document_batch in self.document_iterator:
            self.current_document_batch = document_batch

            start_time = time.time()
            self.inferProfiles(document_batch, print_stats=print_stats)
            inference_time = time.time() - start_time
            
            start_time = time.time()
            self.save(save_words=(epoch % self.save_frequency == 0 and epoch != 0))
            saving_time = time.time() - start_time

            print "Iter: %s, Save: %s, Infer: %s" % (str(epoch), str(saving_time), str(inference_time))

            if epoch % self.test_frequency == 0 and epoch != 0:
                if self.tester:
                    start_time = time.time()
                    self.tester(epoch)
                    print "Test: " + str(time.time() - start_time)     
            
            if epoch % self.save_frequency  == 0:
                all_documents = self.document_iterator.getAll()
                converted_all_documents = self.convertDocuments(all_documents)
                print "Total RMSE: " + str(self.calculateError(converted_all_documents))
            
                
            epoch += 1

            if num_iter is not None and epoch >= num_iter:
                break

    def save_old(self, save_words):
        """
        Serializes the model to an external file.
        The document profiles are automatically saved to the db during updates,
        only remaining model part, i.e. word profile matrix, global statistics (e.g word counts) are serialized to an external file
        :return:
        """

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

        self.updated_word_ids.clear()

    def save(self, save_words):
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

        for local_token_id in self.updated_word_ids:
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

        self.updated_word_ids.clear()

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

    def saveJson(self):
        model_data = dict()
        model_data['num_features'] = self.num_features

        word_profiles = dict()
        doc_freqs = dict()
        for token, local_token_id in self.token_to_id.iteritems():
            word_profiles[token] = self.word_profiles[local_token_id, :].tolist()
            doc_freqs[token] = self.doc_freqs[local_token_id]

        model_data['word_profiles'] = word_profiles
        model_data['doc_freqs'] = doc_freqs

        with open(self.file_name, 'w') as fp:
            json.dump(model_data, fp)

    @staticmethod
    def load(file_name, where):
        """
        :param file_name: serialized model file name
        :return: SemanticModel based on the serialized data
        """

        semantic_model = None

        with open(file_name, 'r') as f:
            snapshot_reader = SemanticModelSnapshotReader(f, db, word_profiles_in_db=True)

            num_features, num_words, num_active_words, num_docs, min_df, max_df = snapshot_reader.readGeneralStats()

            semantic_model = SemanticModel(num_features=num_features, file_name=file_name, where=where)
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


        #semantic_model = SemanticModel(num_features=8, file_name=file_name, where="published = 1 and is_test = 0", min_df=0.003, max_df=0.5)
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
    
