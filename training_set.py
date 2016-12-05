from paperity.environ import db
from paperity.content.paper import Paper, Journal
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import pandas as pd
from collections import defaultdict
from nifty.text import tokenize
import numbers
from semantic_model import DocumentIterator
from gensim import corpora, models
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import logging
import numpy as np
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from collections import Counter
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def tokenize_and_stem(text):
	stemmer = SnowballStemmer("english")
	# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
	tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
	filtered_tokens = []
	# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)

	stems = [stemmer.stem(t) for t in filtered_tokens]

	return stems

def limit_words(papers, min_df, max_df):
    min_document_count = (min_df
        if isinstance(min_df, numbers.Integral)
        else min_df * len(papers))
    max_document_count = (max_df 
        if isinstance(max_df, numbers.Integral) 
        else max_df * len(papers))

    kept_indices = []
    removed_indices = []
    added_indices = []

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


def preprocess_texts(tokenized_texts, min_df, max_df):
	updated_texts = []
	token_frequency = defaultdict(int)
	doc_frequency = defaultdict(int)

	min_document_count = (min_df
		if isinstance(min_df, numbers.Integral)
		else min_df * len(tokenized_texts))
	max_document_count = (max_df 
		if isinstance(max_df, numbers.Integral) 
		else max_df * len(tokenized_texts))


	for tokens in tokenized_texts:
		processed_tokens = set()

		for token in tokens:
			token_frequency[token] += 1

			if token not in processed_tokens:
				doc_frequency[token] += 1
				processed_tokens.add(token)

	for tokens in tokenized_texts:
		filtered_text = [token for token in tokens if 
			token_frequency[token] > 1 and 
			min_document_count <= doc_frequency[token] and 
			doc_frequency[token] <= max_document_count]
		
		updated_texts.append(filtered_text)

	return updated_texts

class MyCorpus(object):
    def __init__(self, where="published = 1 and rawtext <> '' and journal_id in (select journal_id from pap_papers_view p where p.published = 1 group by journal_id having count(1) > 5)"):
    	self.dictionary = corpora.Dictionary.load('tmp/simpleDict.dict')
    	self.where = where
        #self.dictionary = self.create_new_dictionary()

    def __iter__(self):
        for document in Documents(self.where):
            yield self.dictionary.doc2bow(document.tokenized_text)

    @staticmethod
    def create_new_dictionary():
        new_dictionary = corpora.Dictionary(document.tokenized_text for document in Documents())
        new_dictionary.filter_extremes(no_below=2, no_above=0.9, keep_n=None)
        new_dictionary.compactify()
        new_dictionary.save('tmp/simpleDict.dict')

        return new_dictionary

class Documents(object):
	def __init__(self, where):
		self.document_iterator = DocumentIterator()
		self.where =  where

	def __iter__(self):
		for document_batch in self.document_iterator.getAllInBatches(cond=self.where):
			for document in document_batch:
				if len(document.tokenized_text) > 0:
					yield document

def visualize_clusters(dist, clusters, titles):
	print "Calculating MDS..."
	mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

	pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
	print "Finished MDS"
	
	xs, ys = pos[:, 0], pos[:, 1]
	#create data frame that has the result of the MDS plus the cluster numbers and titles
	df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 
	groups = df.groupby('label')
	fig, ax = plt.subplots(figsize=(17, 9))
	for name, group in groups:
		ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, #label=cluster_names[name], color=cluster_colors[name], 
			mec='none')
		ax.set_aspect('auto')
		ax.tick_params(\
			axis= 'x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			labelbottom='off')
		ax.tick_params(\
			axis= 'y',         # changes apply to the y-axis
			which='both',      # both major and minor ticks are affected
			left='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			labelleft='off')
    
	ax.legend(numpoints=1)  #show legend with only 1 point

	#add label in x,y position with the label as the film title
	#for i in range(len(df)):
	#	ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

	plt.show()

def determine_num_of_clusters(X):
	MAX_CLUST = 10
	cs = range(1, MAX_CLUST + 1)

	inertias = np.zeros(MAX_CLUST)
	diff = np.zeros(MAX_CLUST)
	diff2 = np.zeros(MAX_CLUST)
	diff3 = np.zeros(MAX_CLUST)

	for c in cs:
		print "Num of clusters: %d" % c 
		kmeans = KMeans(c).fit(X)
		inertias[c - 1] = kmeans.inertia_
		if c > 1:
			diff[c - 1] = inertias[c - 1] - inertias[c - 2]
		if c > 2:
			diff2[c - 1] = diff[c - 1] - diff[c - 2]
		if c > 3:
			diff3[c - 1] = diff2[c - 1] - diff2[c - 2]

	elbow = np.argmin(diff[3:]) + 3
	
	plt.plot(cs, inertias, "b*-")
	plt.plot(cs[elbow], inertias[elbow], marker='o', markersize=12,
		markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
	plt.ylabel("Inertia")
	plt.xlabel("K")
	print "Elbow: %d" % cs[elbow]
	plt.show()

def hierarchical_clustering(dist, titles):
	#linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
	#joblib.dump(linkage_matrix, 'linkage_matrix.mtrx')
	linkage_matrix = joblib.load('linkage_matrix.mtrx')
	"""
	print "Finished clustering"

	fig, ax = plt.subplots(figsize=(15, 20)) # set size
	ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

	plt.tick_params(\
		axis= 'x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		labelbottom='off')

	plt.tight_layout() #show plot with tight layout

	#uncomment below to save figure
	plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
	"""
	return fcluster(linkage_matrix, 30, criterion='maxclust')

if __name__ == "__main__":
	try:
		"""
		journals = dict()
		for journal in Journal.selectRaw("select * from pap_journals"):
			journals[journal.id] = journal.title

		paper_titles = []
		paper_ids = []
		paper_contents = []
		journal_ids = []
		journal_titles = []


		for idx, paper in enumerate(Documents()):
			paper_titles.append(paper.title)
			paper_ids.append(paper.id)

			journal_ids.append(paper.journal.id)
			journal_titles.append(journals[paper.journal.id])
		"""
		#stopwords = nltk.corpus.stopwords.words('english')
		"""
		print "Building corpus..."

		corpus = MyCorpus()
		
		print "Building tfidf model..."

		#tfidf = models.TfidfModel(corpus)
		#tfidf.save('model.tfidf')
		tfidf = models.TfidfModel.load('model.tfidf')

		corpus_tfidf = tfidf[corpus]

		print "Building lsi model..."

		#lsi = models.LsiModel(corpus_tfidf, id2word=corpus.dictionary, num_topics=100)
		#lsi.print_topics(10)
		#lsi.save('model.lsi')
		"""
		"""
		lsi = models.LsiModel.load('model.lsi')
		topics = lsi.show_topics(num_topics=40, num_words=10, log=False, formatted=False)
		for topic_no, topic in topics:
			print [word for (word, weight) in topic]
		"""
		"""
		print "Calculating weights..."

		corpus_lsi = lsi[corpus_tfidf]
		doc_weights = []
		for lsi_doc in corpus_lsi:
			doc_weights.append([weight for (feature, weight) in lsi_doc])

		joblib.dump(doc_weights, 'doc_weights.mtrx')
		"""
		"""
		doc_weights = joblib.load('doc_weights.mtrx')
		
		print "Calculating distance matrix..."
		dist = 1 - cosine_similarity(doc_weights)
		"""
		"""
		#tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
		#								   min_df=0.2, stop_words='english',
		#								   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

		#tfidf_matrix = tfidf_vectorizer.fit_transform(paper_contents)
		#determine_num_of_clusters(doc_weights)

		"""
		"""
		print "Clustering..."
		num_clusters = 40
		km = KMeans(n_clusters=num_clusters)
		km.fit(doc_weights)

		print "Finished clustering"

		joblib.dump(km,  'doc_cluster.pkl')
		
	
		km = joblib.load('doc_cluster.pkl')
		clusters = km.labels_.tolist()
		
		distances = []
		for idx, doc in enumerate(doc_weights):
			cluster = clusters[idx]
			cluster_center = km.cluster_centers_[cluster]
			distance_to_center = euclidean_distances(np.asarray([doc]), np.asarray([cluster_center]))
			distances.append(distance_to_center)
		
		print "Finished distance calculation"
		"""
		"""
		results = { 'pap_title': paper_titles, 'pap_id': paper_ids, 'cluster': clusters, 
			        'journal_id': journal_ids, 'journal_title': journal_titles }

		result_frame = pd.DataFrame(results, index=[clusters])
		result_frame.to_csv('clustered_docs.csv', encoding='utf-8')
		"""
		
		result_frame = pd.DataFrame.from_csv('clustered_docs.csv')
		#clusters = hierarchical_clustering(dist=None, titles=result_frame['journal_title'].tolist())
		#result_frame['cluster'] = pd.Series(clusters, index=result_frame.index)
		#result_frame['distance'] = pd.Series(distances, index=result_frame.index)
		#result_frame.to_csv('clustered_docs.csv', encoding='utf-8')
		#result_frame.group_by(result_frame['cluster'])
		
		with open('cluster_descriptions.descr', 'w') as f:
			for cluster in range(0, 40):
				paper_ids = result_frame[result_frame.cluster == cluster]['pap_id'].tolist()
				word_count = Counter()
				doc_freqs = Counter()

				for document in Documents(where="id in (" + ', '.join([str(id) for id in paper_ids]) + ")"):
					processed_words = set()
					for word in document.tokenized_text:
						word_count[word] += 1

	            		if word not in processed_words:
	                		doc_freqs[word] += 1
	                		processed_words.add(word)

				f.write("Cluster " + str(cluster) + ", " + \
					', '.join([word for word, count in word_count.most_common(60)]) + "\n")
		
	finally:
		db.disconnect()