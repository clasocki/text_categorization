from semantic_model import SemanticModel, DocumentIterator
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import numpy as np
from paperity.environ import db
from sklearn.metrics.pairwise import pairwise_distances
import numpy
from random import randint
import gensim_tests
from nifty.text import tokenize
import itertools

from svd_tester import test_accuracy
import MySQLdb

LABELED_DOCUMENTS_CONDITION = "published = 1 AND learned_category IS NOT NULL"
UNLABELED_DOCUMENTS_CONDITION = "published = 1 AND learned_category IS NULL"
PROFILE_INFERENCE_NUM_ITERS = 10
N_NEIGHBORS = 15
NEW_LABELS_BATCH = 1000
FINAL_DOCUMENT_COUNT = 50000
acceptable_distance = 1.0
MODEL_SNAPSHOT_FILENAME = 'semantic_model.snapshot'
NUM_ITERS_MODEL_UPDATE = 30
DOCUMENT_BATCH_SIZE = 1000
DB_WINDOW_SIZE = 800

def find_most_common_label(labels):
	count = Counter(labels)
	freq_list = count.values()
	total_most_common = freq_list.count(max(freq_list))
	most_common = count.most_common(total_most_common)
	most_common = [elem[0] for elem in most_common]

	idx = randint(0, len(most_common) - 1)
	return most_common[idx]

def get_labeled_set():
	labeled_documents = DocumentIterator(where=LABELED_DOCUMENTS_CONDITION).getAll()
	
	labeled_profiles = []
	labels = []
	
	for doc in labeled_documents:
		labeled_profiles.append(doc.profile)
		labels.append(doc.learned_category)

	return labeled_profiles, labels

def find_closest_category(categories, distances, indices):
	closest_categories = [categories[i] for i in indices]
	closest_category = find_most_common_label(closest_categories)
	closest_category_indices = [i for i, category in enumerate(closest_categories) if category == closest_category]
	average_distance = numpy.mean(distances[closest_category_indices])
	return closest_category, average_distance

def assign_category(document, category, assigned_category_counter, newly_labeled_documents):
	assigned_category_counter[category] += 1
	newly_labeled_documents.append(document)
	document.learned_category = category

	sql_update = "update pap_papers_2 set learned_category = '" + str(category) + \
		"' WHERE id = " + str(document.id)
	#print document.title + " " + category

	db.query(sql_update)
	db.commit()


def propagate_labels(labeled_profiles, labels, acceptable_distance):
	print "Label propagation..."

	semantic_model = SemanticModel.load(file_name=MODEL_SNAPSHOT_FILENAME, where=LABELED_DOCUMENTS_CONDITION)

	
	#db = MySQLdb.connect(host='localhost', user='root', passwd='1qaz@WSX', db='paperity')
	
	#semantic_model.tester = lambda epoch: test_accuracy(semantic_model, db, epoch, 'accuracy_result.csv')


	assigned_category_counter = Counter()
	newly_labeled_documents = []

	unlabeled_document_iterator = DocumentIterator(document_batch_size=DOCUMENT_BATCH_SIZE, db_window_size=DB_WINDOW_SIZE, where=UNLABELED_DOCUMENTS_CONDITION)
	nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles)

	stop_propagation = semantic_model.num_docs >= FINAL_DOCUMENT_COUNT
	for unlabeled_document_batch in unlabeled_document_iterator:
		if stop_propagation:
			break

		for i, unlabeled_document in enumerate(unlabeled_document_batch):
			print i
			semantic_model.inferProfiles([unlabeled_document], 
				num_iters=PROFILE_INFERENCE_NUM_ITERS, update_word_profiles=False, initialize_document_profiles=True)
			distances, indices = nbrs.kneighbors([unlabeled_document.profile])

			closest_category, average_distance = find_closest_category(labels, distances[0], indices[0])

			if average_distance <= acceptable_distance:
				assign_category(unlabeled_document, closest_category, assigned_category_counter, newly_labeled_documents)
				
				if len(newly_labeled_documents) == NEW_LABELS_BATCH:
					print "Updating model..."
					semantic_model.document_iterator.saveDocumentProfilesToDb(newly_labeled_documents)
					semantic_model.update(newly_labeled_documents, num_iters=NUM_ITERS_MODEL_UPDATE)
					
					labeled_profiles, labels = get_labeled_set()
					
					#mean, standard_deviation = calculate_statistics(labeled_profiles)
					#acceptable_distance = mean - standard_deviation
					#print mean, standard_deviation, acceptable_distance

					nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles)

					newly_labeled_documents = []


			if semantic_model.num_docs >= FINAL_DOCUMENT_COUNT:
				stop_propagation = True
				break

def propagate_labels_gensim(labeled_profiles, labels, acceptable_distance, num_features, semantic_model, calc_distances):
	assigned_category_counter = Counter()
	newly_labeled_documents = []

	unlabeled_document_iterator = DocumentIterator(document_batch_size=DOCUMENT_BATCH_SIZE, db_window_size=DB_WINDOW_SIZE, where=UNLABELED_DOCUMENTS_CONDITION)
	nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles)

	#stop_propagation = semantic_model.num_docs >= FINAL_DOCUMENT_COUNT
	for unlabeled_document_batch in unlabeled_document_iterator:
		#if stop_propagation:
		#	break

		for i, unlabeled_document in enumerate(unlabeled_document_batch):
			print i
			
			profile = semantic_model.inferProfile(unlabeled_document.tokenized_text)
			if len(profile) == 0:
				continue
			distances, indices = nbrs.kneighbors([profile])

			closest_category, average_distance = find_closest_category(labels, distances[0], indices[0])

			if average_distance <= acceptable_distance:
				assign_category(unlabeled_document, closest_category, assigned_category_counter, newly_labeled_documents)
				
				if len(newly_labeled_documents) == NEW_LABELS_BATCH:
					print "Updating model..."
					
					labeled_profiles, labels, semantic_model = getLabeledSetGensim(num_features)
					
					#mean, standard_deviation = calculate_statistics(labeled_profiles, semantic_model, calc_distances)
					#acceptable_distance = mean - standard_deviation
					#print mean, standard_deviation, acceptable_distance

					nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles)

					newly_labeled_documents = []


			#if semantic_model.num_docs >= FINAL_DOCUMENT_COUNT:
			#	stop_propagation = True
			#	break


def calculate_statistics(labeled_profiles, semantic_model, calc_distances):
	print "Calculating statistics..."

	distance_sum, counter = 0.0, 0
	last_record_offset = -1	

	#### online standard deviation
	mean, M2 = 0.0, 0.0
	####

	unlabeled_document_iterator = DocumentIterator(document_batch_size=DOCUMENT_BATCH_SIZE, db_window_size=3500, where=UNLABELED_DOCUMENTS_CONDITION)

	all_distances = []

	for unlabeled_document_batch in unlabeled_document_iterator:
		distances = calc_distances(labeled_profiles, semantic_model, unlabeled_document_batch)
		
		#### online standard deviation

		all_distances.extend(distances.flatten().tolist())
		
		for dist in distances.flatten():
			counter += 1
			delta = dist - mean
			mean += delta / counter
			delta2 = dist - mean
			M2 += delta * delta2
		####

		distance_sum += numpy.sum(distances)

		if unlabeled_document_iterator.current_record_offset < last_record_offset:
			break

		last_record_offset = unlabeled_document_iterator.current_record_offset

	mean = distance_sum / counter
	standard_deviation = numpy.sqrt(M2 / (counter - 1))

	return mean, standard_deviation

def calcDistancesIter(labeled_profiles, semantic_model, unlabeled_document_batch):
	semantic_model.inferProfiles(unlabeled_document_batch, 
			num_iters=PROFILE_INFERENCE_NUM_ITERS, update_word_profiles=False, initialize_document_profiles=True)

	return pairwise_distances(labeled_profiles, [doc.profile for doc in unlabeled_document_batch], metric='cosine')

def calcDistancesGensim(labeled_profiles, semantic_model, unlabeled_document_batch):
	profiles = []

	for doc in unlabeled_document_batch:
		profile = semantic_model.inferProfile(doc.tokenized_text)
		if len(profile) > 0:
			profiles.append(profile)
	
	print sum(len(p) for p in labeled_profiles) / float(len(labeled_profiles)), sum(len(p) for p in profiles) / float(len(profiles))
	return pairwise_distances(labeled_profiles, profiles, metric='cosine')

class LocalDocumentGenerator(object):
	def __init__(self, cursor):
		self.cursor = cursor

	def __iter__(self):
		for row in self.cursor:
			tokenized_text = tokenize(row[0]).split()
			label = row[1]
			if len(tokenized_text) > 0:
				yield tokenized_text, label

def getLabeledSetGensim(num_features):
	db = MySQLdb.connect(host='localhost', user='root', passwd='1qaz@WSX', db='paperity')
	cursor = db.cursor()
	cursor.execute("SELECT rawtext, learned_category FROM pap_papers_view where %s" % (LABELED_DOCUMENTS_CONDITION,))
	labeled_docs = LocalDocumentGenerator(cursor)

	semantic_model = gensim_tests.SemanticModel.build((text for text, _ in labeled_docs), num_features, 0.002 * cursor.rowcount, 0.33 * cursor.rowcount)
	
	labeled_profiles, labels = [], []

	for text, label in labeled_docs:
		profile = semantic_model.inferProfile(text)
		if len(profile) > 0:
			labeled_profiles.append(profile)
			labels.append(label)

	cursor.close()
	db.close()

	return labeled_profiles, labels, semantic_model

if __name__ == "__main__":
	iterative = False

	if iterative:
		semantic_model = SemanticModel.load(file_name=MODEL_SNAPSHOT_FILENAME, where=LABELED_DOCUMENTS_CONDITION)
		labeled_profiles, labels = get_labeled_set()

		mean, standard_deviation = calculate_statistics(labeled_profiles, semantic_model, calcDistancesIter)
		print mean, standard_deviation, mean - standard_deviation
		mean, standard_deviation = 0.563052939071, 0.153272002339
		acceptable_distance = mean - standard_deviation
		propagate_labels(labeled_profiles, labels, acceptable_distance)
	
		#propagate_labels(labeled_profiles, labels, 0.905576610851 - 0.258739991239)
	else:
		num_features = 50
		labeled_profiles, labels, semantic_model = getLabeledSetGensim(num_features)
		print "Len labeled: ", len(labeled_profiles) 		
		#mean, standard_deviation = calculate_statistics(labeled_profiles, semantic_model, calcDistancesGensim)
		#print mean, standard_deviation
		mean, standard_deviation = 0.853455089498, 0.166213023676
		acceptable_distance = mean - standard_deviation

		propagate_labels_gensim(labeled_profiles, labels, acceptable_distance, num_features, semantic_model, calcDistancesGensim)