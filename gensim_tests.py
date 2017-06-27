from gensim import corpora, models
import itertools

class SemanticModel(object):
	def __init__(self):
		self.lsi = None
		self.tfidf = None
		self.dic = None
                self.num_features = None

	def createDictionary(self, tokenized_texts, min_df, max_df):
		new_dictionary = corpora.Dictionary(tokenized_texts)
		rare_ids = [tokenid for tokenid, docfreq in new_dictionary.dfs.iteritems() 
					if docfreq < min_df or docfreq > max_df]
		new_dictionary.filter_tokens(rare_ids)
		new_dictionary.compactify()

		return new_dictionary

	@staticmethod
	def build(tokenized_texts, num_features, min_df, max_df):
		model = SemanticModel()
                model.num_features = num_features

		text_it1, text_it2 = itertools.tee(tokenized_texts, 2)

		model.dic = model.createDictionary(text_it1, min_df, max_df)
		corpus = [model.dic.doc2bow(tokenized_text) for tokenized_text in text_it2]

		model.tfidf = models.TfidfModel(corpus)
		corpus_tfidf = model.tfidf[corpus]

		model.lsi = models.LsiModel(corpus_tfidf, num_topics=num_features, id2word=model.dic)
		#lsiBOW = models.LsiModel(corpus, num_topics=NUM_TOP, id2word=dic)
		#corpus_lsi = lsi[corpus_tfidf]

		return model

	def inferProfile(self, tokenized_text):
		text_bow = self.dic.doc2bow(tokenized_text)
		text_tfidf = self.tfidf[text_bow]
		profile = self.lsi[text_tfidf]
	
		profile = [feature_val for feature_id, feature_val in profile]
                return profile if profile else [0] * self.num_features

	def inferProfiles(self, tokenized_texts):
		return [self.inferProfile(text) for text in tokenized_texts]
