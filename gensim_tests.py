from gensim import corpora, models

def createDictionary(tokenized_texts):
	document_count = len(tokenized_texts)
	new_dictionary = corpora.Dictionary(tokenized_text for tokenized_text in tokenized_texts)
	rare_ids = [tokenid for tokenid, docfreq in new_dictionary.dfs.iteritems() 
				if docfreq < 0.003 * document_count or docfreq > 0.7 * document_count]
	new_dictionary.filter_tokens(rare_ids)
	new_dictionary.compactify()

	return new_dictionary

def inferProfiles(train_tokenized_texts, test_tokenized_texts):
	dic = createDictionary(train_tokenized_texts)

	corpus = [dic.doc2bow(tokenized_text) for tokenized_text in train_tokenized_texts]

	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]

	lsi = models.LsiModel(corpus_tfidf, num_topics=20, id2word=dic)
	#lsiBOW = models.LsiModel(corpus, num_topics=NUM_TOP, id2word=dic)
	corpus_lsi = lsi[corpus_tfidf]

	train_profiles = []
	for train_document, profile in zip(train_tokenized_texts, corpus_lsi):
		train_profiles.append([feature[1] for feature in profile])

	test_profiles = []
	for test_tokenized_text in test_tokenized_texts:
		test_document_bow = dic.doc2bow(test_tokenized_text)
		test_document_tfif = tfidf[test_document_bow]
		test_document_profile = lsi[test_document_tfif]
		test_profiles.append([feature[1] for feature in test_document_profile])

	return train_profiles, test_profiles


