from gensim import corpora, models
import itertools
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
import math

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def lognormalize(tf):
            return 1 + math.log(tf) if tf > 0 else 0

class SemanticModel(object):
	def __init__(self):
		self.lsi = None
		self.tfidf = None
		self.dict = None
                self.num_features = None
                self.stopwords = stopwords.words('english')
                self.stemmer = SnowballStemmer('english') 

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
		#rare_ids = [tokenid for tokenid, docfreq in new_dictionary.dfs.iteritems() 
		#			if docfreq < min_df or docfreq > max_df]
		#new_dictionary.filter_tokens(rare_ids)
		#new_dictionary.compactify()
                new_dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)

		return new_dictionary

        @staticmethod
        def getTfIdfCorpus(model, getTexts):
            corpus = (model.dict.doc2bow(tokenized_text) for tokenized_text in model.tokenizeTexts(getTexts()))
            
            model.tfidf = models.TfidfModel(corpus, dictionary=model.dict, wlocal=lognormalize, normalize=False)
            corpus_tfidf = model.tfidf[corpus]

            return corpus_tfidf

        @staticmethod
        def load(filepath, getTexts=None):
            model = SemanticModel()

            model.dict = corpora.Dictionary.load(filepath + '.dict')
	    model.lsi = models.LsiModel.load(filepath + '.lsi')
            model.tfidf = models.TfidfModel.load(filepath + '.tfidf')
            model.num_features = model.lsi.num_topics

            return model

	@staticmethod
	def build(getTexts, num_features, no_below=2, no_above=0.10, keep_n=300000, filepath='default'):
		model = SemanticModel()
                model.num_features = num_features

                #model.dict = corpora.Dictionary.load(filepath + '.dict')
		model.dict = model.createDictionary(model.tokenizeTexts(getTexts()), no_below, no_above, keep_n)
                if filepath:
                    model.dict.save(filepath + '.dict')
		
                corpus_tfidf = SemanticModel.getTfIdfCorpus(model, getTexts)

		model.lsi = models.LsiModel(corpus_tfidf, num_topics=num_features, id2word=model.dict)
                
                if filepath:
                    model.tfidf.save(filepath + '.tfidf')
                    model.lsi.save(filepath + '.lsi')

		return model

	def inferProfile(self, text):
                tokenized_text = self.tokenize(text)
		text_bow = self.dict.doc2bow(tokenized_text)
		text_tfidf = self.tfidf[text_bow]
		profile = self.lsi[text_tfidf]
	
		profile = [feature_val for feature_id, feature_val in profile]
                return profile if profile else [0] * self.num_features

	def inferProfiles(self, tokenized_texts):
		return [self.inferProfile(text) for text in tokenized_texts]
