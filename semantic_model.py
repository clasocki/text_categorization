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
from sortedcontainers import SortedSet
import time
import itertools

class DocumentIterator(object):
    DOCUMENT_BATCH_SIZE = 300 # the number of documents returned in a single iteration
    DB_WINDOW_SIZE = 600 # the number of documents from which the current batch is randomly selected

    def __init__(self, where, document_batch_size=300, db_window_size=600):
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
        print query
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

    def saveDocumentProfilesToFile(self, file_name):
        all_documents = self.getAll()
        
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

            sql_update = "UPDATE pap_papers_view SET profile = '" + \
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

            yield document_batch

class TermFrequencyWeight(Enum):
    RAW_FREQUENCY = 1
    LOG_NORMALIZATION = 2
    AUGMENTED_FREQUENCY = 3
    BINARY = 4

class SemanticModel(object):
    MAX_UPDATE_ITER = 1
    REGULARIZATION_FACTOR = 0.00
    LEARNING_RATE = 0.005

    def __init__(self, num_features, file_name, where,
        term_freq_weight=TermFrequencyWeight.LOG_NORMALIZATION, use_idf = True,
        min_df=0.0, max_df=1.0, limit_features=True,
        preanalyze_documents=True, tester=None):
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

        self.word_profiles = numpy.random.uniform(low=-0.01, high=0.01, size=(self.num_features, 1))
        self.document_iterator = DocumentIterator(where=where)
        self.token_to_id = dict() # token -> token id
        self.doc_freqs = defaultdict(int)  # token id -> the number of documents this token appears in
        self.active_tokens = SortedSet() # set of tokens that aren't too rare or too common, 
                                   # i.e. they exist in (for float) 
                                   # [min_df * self.num_docs, max_df * self.num_docs] or (for int) [min_df, max_df] documents 
        self.current_document_batch = None  

    def inferProfiles(self, documents, update_word_profiles=True, initialize_document_profiles=False, num_iters=MAX_UPDATE_ITER):
        """
        Calculates profiles for documents
        :param documents: a list of unconverted documents
        :return:
        """    
        documents = self.convertDocuments(documents)

        if initialize_document_profiles:
            self.initializeDocumentProfiles(documents)  

        current_iter = 0

        if len(documents) == 0:
            return

        start_time = time.time()

        for current_iter in range(num_iters):
            squared_error = 0.0
            num_values = 0

            #active_tokens_sample = numpy.random.choice(len(self.active_tokens), 400, replace=False)
            active_tokens_sample = None
            
            for document in documents:
                for word_id, value in self.fullDictionaryBasedDocument(document, active_tokens_sample):
                #for word_id, value in itertools.chain(weight_updates, document.converted_text.iteritems()):
                    num_values += 1

                    predicted_value = self.predictValue(document, word_id)
                    error = 1.0 * value - predicted_value
                    squared_error += error * error

                    """
                    for feature_id in numpy.arange(self.num_features):
                        document_feature_value = document.profile[feature_id]
                        word_feature_value = self.word_profiles[feature_id, word_id]

                        document.profile[feature_id] += \
                            self.LEARNING_RATE * (error * word_feature_value - self.REGULARIZATION_FACTOR * document_feature_value)
                        
                        if update_word_profiles:
                            self.word_profiles[feature_id, word_id] += \
                                self.LEARNING_RATE * (error * document_feature_value - self.REGULARIZATION_FACTOR * word_feature_value)

                    """
                    document.profile += \
                        self.LEARNING_RATE * (error * self.word_profiles[:, word_id] - self.REGULARIZATION_FACTOR * document.profile)
                    
                    if update_word_profiles:
                        self.word_profiles[:, word_id] += \
                            self.LEARNING_RATE * (error * document.profile - self.REGULARIZATION_FACTOR * self.word_profiles[:, word_id])

 
            #print self.token_to_id.keys()
            #rmse = numpy.sqrt(squared_error / num_values)
            #print "Partial RMSE: " + str(rmse)
            #print "Num operations: " + str(num_operations)
        print "End - time: " + str(time.time() - start_time)

    def calculateError(self, documents):
        squared_error = 0.0
        num_values = 0

        for document in documents:
            for word_id, value in self.fullDictionaryBasedDocument(document):
                num_values += 1
                predicted_value = self.predictValue(document, word_id)
                error = 1.0 * value - predicted_value
                squared_error += error * error

        rmse = numpy.sqrt(squared_error / num_values)

        return rmse

    def update(self, documents, num_iters_model_update, num_iters_profile_inference=None):
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

        self.updateStatisticsForNewDocuments(documents)        
        if self.limit_features:
            print "Limiting the number of words..."
            kept_indices, added_indices, removed_indices = self.limit_words()
            print "Kept: " + str(len(kept_indices)) + ", added: " + str(len(added_indices)) + ", removed: " + str(len(removed_indices))
        self.resizeProfileMatrices()
        self.save()

        self.train(num_iters_model_update)
 

    def predictValue(self, document, word_id):
        #print "Document id " + str(document_id)
        #print "Word id " + str(word_id)
        return numpy.dot(document.profile, self.word_profiles[:, word_id])

    def updateStatisticsForNewDocument(self, document):
        self.num_docs += 1

        processed_tokens = set()
        for token in document.tokenized_text:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
                self.active_tokens.add(self.token_to_id[token])

            if token not in processed_tokens:
                self.doc_freqs[self.token_to_id[token]] += 1
                processed_tokens.add(token)

        #print "The number of tokens: " + str(len(self.token_to_id))
        #print self.token_to_id.keys()

    def updateStatisticsForNewDocuments(self, documents):
        for document in documents:
            self.updateStatisticsForNewDocument(document)

    def initializeDocumentProfiles(self, documents):
        for document in documents:
            document.profile = numpy.random.uniform(low=-0.01, high=0.01, size=self.num_features)

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

    def limit_words(self):
        min_document_count = (self.min_df
            if isinstance(self.min_df, numbers.Integral)
            else self.min_df * self.num_docs)
        max_document_count = (self.max_df 
            if isinstance(self.max_df, numbers.Integral) 
            else self.max_df * self.num_docs)

        kept_indices = []
        removed_indices = []
        added_indices = []

        print "Min doc count: %s, min_df: %s, max doc count: %s, max_df: %s" % (min_document_count, self.min_df, max_document_count, self.max_df)

        for token_id, freq in self.doc_freqs.iteritems():
            if min_document_count <= freq and freq <= max_document_count:
                if token_id not in self.active_tokens:
                    self.active_tokens.add(token_id)
                    added_indices.append(token_id)
                else:
                    kept_indices.append(token_id)
            else:
                if token_id in self.active_tokens:
                    self.active_tokens.remove(token_id)
                    removed_indices.append(token_id)

        return kept_indices, added_indices, removed_indices

    def convertDocuments(self, documents):
        """
        Tokenizes and converts each document from the raw text form to the bag-of-words format
        :param documents:
        :return: list of documents in the bag-of-words format with a local id assigned
        """
        for document in documents:
            document_id = document.id
            tokenized_text = document.tokenized_text

            bag_of_words = defaultdict(int)
            for token in tokenized_text:
                if token not in self.token_to_id:
                    continue
                
                token_id = self.token_to_id[token]

                if token_id in self.active_tokens:
                    bag_of_words[token_id] += 1

            converted_text = dict()
            for token_id in bag_of_words.keys():
                token_weight = self.tf(bag_of_words, token_id) * (self.idf(token_id) if self.use_idf else 1)
                converted_text[token_id] = token_weight

            document.converted_text = converted_text

        return documents

    def fullDictionaryBasedDocument(self, document, active_tokens_sample=None):
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
        
        if active_tokens_sample is None:
            positive_weight_count = len(document.converted_text)
            all_active_tokens_count = len(self.active_tokens)
            active_tokens_sample = numpy.random.choice(all_active_tokens_count, int(math.ceil(positive_weight_count * 3.0)), replace=False)
            #active_tokens_sample = numpy.random.choice(xrange(all_active_tokens_count), 400, replace=False)
            #active_tokens_sample = self.active_tokens.keys()
        #print "Positive weight count: " + str(positive_weight_count)
        
        weights = []
        weights = [(self.active_tokens[sample_token_id], 0) 
            for sample_token_id in active_tokens_sample 
            if self.active_tokens[sample_token_id] not in document.converted_text]
        weights.extend(document.converted_text.iteritems())

        numpy.random.shuffle(weights)

        return weights

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

    def resizeProfileMatrices(self):
        (_, word_profile_count) = self.word_profiles.shape

        while len(self.token_to_id) > word_profile_count:
            self.word_profiles = numpy.resize(self.word_profiles,
                (self.num_features, word_profile_count * 2))
            self.word_profiles[:, word_profile_count:] = numpy.random.uniform(low=-0.01, high=0.01, 
                size=(self.num_features, word_profile_count))

            (_, word_profile_count) = self.word_profiles.shape

    def train(self, num_iter=None):
        """
        Main loop that iterates over the db, periodically getting new document batches.
        Starts using empty profile matrices, incrementally resizing them in the process of model training.
        :return:
        """
        epoch = 0

        if self.preanalyze_documents:
            all_documents = self.document_iterator.getAll()
            self.current_document_batch = all_documents
            self.initializeDocumentProfiles(all_documents)
            self.updateStatisticsForNewDocuments(all_documents)
            kept_indices, added_indices, removed_indices = self.limit_words()
            print "Kept: " + str(len(kept_indices)) + ", added: " + str(len(added_indices)) + ", removed: " + str(len(removed_indices))
            self.resizeProfileMatrices()
            self.save()

        #while (epoch < self.min_iter or rmse_last - rmse >= self.min_improvement) and epoch < self.max_iter:
        #while num_iter is None or epoch < num_iter:
        #while True:
        for document_batch in self.document_iterator:
            self.current_document_batch = document_batch

            self.inferProfiles(document_batch)                 

            self.save()
            
            print "Current iteration: " + str(epoch)
            print "Total num iter: " + str(num_iter)

            if epoch % 35 == 0 and epoch != 0:
                if self.tester:
                    self.tester(epoch)
            
            if epoch % 35 == 0 and epoch != 0:
                all_documents = self.document_iterator.getAll()
                converted_all_documents = self.convertDocuments(all_documents)
                print "Total RMSE: " + str(self.calculateError(converted_all_documents))

                
            epoch += 1

            if num_iter is not None and epoch >= num_iter:
                break

    def save(self):
        """
        Serializes the model to an external file.
        The document profiles are automatically saved to the db during updates,
        only remaining model part, i.e. word profile matrix, global statistics (e.g word counts) are serialized to an external file
        :return:
        """

        if self.current_document_batch is not None:
            self.document_iterator.saveDocumentProfilesToDb(self.current_document_batch)

        with open(self.file_name, 'w') as f:
            """
            TODO: add ALL model parameters to the first line
            """
            f.write(str(self.num_features) + '\t' + str(len(self.token_to_id)) + '\t' + str(self.num_docs) + '\t' + \
                str(self.min_df) + '\t' + str(self.max_df) + '\n')

            for token, local_token_id in self.token_to_id.iteritems():
                word_stats_line = token + '\t'
                word_stats_line += str(self.doc_freqs[local_token_id]) + '\t'
                word_stats_line += str(int(local_token_id in self.active_tokens)) + '\t'

                word_profile = self.word_profiles[:, local_token_id]
                str_word_profile = ','.join(str(profile_element) for profile_element in word_profile)
                
                word_stats_line += str_word_profile + '\n'

                f.write(word_stats_line)


    def saveJson(self):
        model_data = dict()
        model_data['num_features'] = self.num_features

        word_profiles = dict()
        doc_freqs = dict()
        for token, local_token_id in self.token_to_id.iteritems():
            word_profiles[token] = self.word_profiles[:, local_token_id].tolist()
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
            snapshot_reader = SemanticModelSnapshotReader(f)

            num_features, num_words, num_docs, min_df, max_df = snapshot_reader.readGeneralStats()

            semantic_model = SemanticModel(num_features=num_features, file_name=file_name, where=where)
            semantic_model.num_docs = num_docs
            semantic_model.word_profiles = numpy.zeros((num_features, num_words))
            semantic_model.min_df = min_df
            semantic_model.max_df = max_df
            semantic_model.preanalyze_documents = False
            semantic_model.active_tokens = SortedSet()

            for token, word_doc_freqs, is_active, word_profile in snapshot_reader.readWordProfiles():
                semantic_model.token_to_id[token] = len(semantic_model.token_to_id)                
                semantic_model.doc_freqs[semantic_model.token_to_id[token]] = word_doc_freqs
                semantic_model.word_profiles[:, semantic_model.token_to_id[token]] = word_profile
                if is_active:
                    semantic_model.active_tokens.add(semantic_model.token_to_id[token])

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
    def __init__(self, file_object):
        self.file_object = file_object
        self.first_line_read = False

    def readGeneralStats(self):
        if self.first_line_read:
            self.file_object.seek(0)
            self.first_line_read = False

        first_line_stats = self.file_object.readline()
        first_line_stats = first_line_stats.split('\t')

        num_features = int(first_line_stats[0])
        num_words = int(first_line_stats[1])
        num_docs = int(first_line_stats[2])
        min_df = float(first_line_stats[3])
        max_df = float(first_line_stats[4])

        self.first_line_read  = True

        return num_features, num_words, num_docs, min_df, max_df

    def readWordProfiles(self):
        if not self.first_line_read:
            self.readGeneralStats()

        for word_stats in self.file_object:
            split_word_stats = word_stats.split('\t')
            token = split_word_stats[0]
            word_doc_freqs = int(split_word_stats[1])
            is_active = bool(int(split_word_stats[2]))
            word_profile = split_word_stats[3]
            word_profile = numpy.asarray([float(value) for value in word_profile.split(',')])

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
            snapshot_reader = SemanticModelSnapshotReader(f)
            #print snapshot_reader.readGeneralStats()
            snapshot_reader.printOrderedWordProfiles(order='asc', min_freq=0, max_freq=sys.maxint, only_active=True)
    except (KeyboardInterrupt, SystemExit):
        #semantic_model.save()
        print "Saved"
        raise
    finally:
        db.disconnect()
    
