from semantic_model import SemanticModel, DocumentIterator
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import numpy as np
from paperity.environ import db
from sklearn.metrics.pairwise import pairwise_distances
import numpy
from random import randint

from svd_tester import test_accuracy
import MySQLdb

LABELED_DOCUMENTS_CONDITION = "published = 1 AND learned_category IS NOT NULL"
UNLABELED_DOCUMENTS_CONDITION = "published = 1 AND learned_category IS NULL"
PROFILE_INFERENCE_NUM_ITERS = 10
N_NEIGHBORS = 15
NEW_LABELS_BATCH = 250
FINAL_DOCUMENT_COUNT = 4500
acceptable_distance = 1.0
MODEL_SNAPSHOT_FILENAME = 'semantic_model.snapshot'
NUM_ITERS_MODEL_UPDATE = 40
DOCUMENT_BATCH_SIZE = 500

def find_most_common_label(labels):
	count = Counter(labels)
	freq_list = count.values()
	total_most_common = freq_list.count(max(freq_list))
	most_common = count.most_common(total_most_common)
	most_common = [elem[0] for elem in most_common]

	idx = randint(0, len(most_common) - 1)
	return most_common[idx]

def find_initial_label_set():
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
	print document.title + " " + category

	db.query(sql_update)


def propagate_labels(labeled_profiles, labels, acceptable_distance):
	semantic_model = SemanticModel.load(file_name=MODEL_SNAPSHOT_FILENAME, where=LABELED_DOCUMENTS_CONDITION)

	"""
	db = MySQLdb.connect(host='localhost', user='root',
                         passwd='1qaz@WSX', db='test')
	"""
	semantic_model.tester = lambda epoch: test_accuracy(semantic_model, db, epoch, 'accuracy_result.csv')


	assigned_category_counter = Counter()
	newly_labeled_documents = []

	unlabeled_document_iterator = DocumentIterator(document_batch_size=DOCUMENT_BATCH_SIZE, where=UNLABELED_DOCUMENTS_CONDITION)
	nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles)

	stop_propagation = semantic_model.num_docs >= FINAL_DOCUMENT_COUNT
	for unlabeled_document_batch in unlabeled_document_iterator:
		if stop_propagation:
			break

		for unlabeled_document in unlabeled_document_batch:
			semantic_model.inferProfiles([unlabeled_document], 
				num_iters=PROFILE_INFERENCE_NUM_ITERS, update_word_profiles=False, initialize_document_profiles=True)
			distances, indices = nbrs.kneighbors([unlabeled_document.profile])

			closest_category, average_distance = find_closest_category(labels, distances[0], indices[0])

			if average_distance <= acceptable_distance:
				assign_category(unlabeled_document, closest_category, assigned_category_counter, newly_labeled_documents)
				
				if len(newly_labeled_documents) == NEW_LABELS_BATCH:
					semantic_model.document_iterator.saveDocumentProfilesToDb(newly_labeled_documents)
					semantic_model.update(newly_labeled_documents, num_iters_model_update=NUM_ITERS_MODEL_UPDATE,
						num_iters_profile_inference=PROFILE_INFERENCE_NUM_ITERS)
					
					for doc in newly_labeled_documents:
						labeled_profiles.append(doc.profile)
						labels.append(doc.learned_category)

					nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles)

					newly_labeled_documents = []


			if semantic_model.num_docs >= FINAL_DOCUMENT_COUNT:
				stop_propagation = True
				break


def calculate_statistics(labeled_profiles):
	distance_sum, counter = 0.0, 0
	last_record_offset = -1
	semantic_model = SemanticModel.load(file_name=MODEL_SNAPSHOT_FILENAME, where=LABELED_DOCUMENTS_CONDITION)

	#### online standard deviation
	mean, M2 = 0.0, 0.0
	####

	unlabeled_document_iterator = DocumentIterator(document_batch_size=DOCUMENT_BATCH_SIZE, db_window_size=2000, where=UNLABELED_DOCUMENTS_CONDITION)

	all_distances = []

	for unlabeled_document_batch in unlabeled_document_iterator:
		semantic_model.inferProfiles(unlabeled_document_batch, 
			num_iters=PROFILE_INFERENCE_NUM_ITERS, update_word_profiles=False, initialize_document_profiles=True)

		distances = pairwise_distances(labeled_profiles, [doc.profile for doc in unlabeled_document_batch], metric='cosine')
		
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

if __name__ == "__main__":
	labeled_profiles, labels = find_initial_label_set()

	mean, standard_deviation = calculate_statistics(labeled_profiles)
	#print mean, standard_deviation
	propagate_labels(labeled_profiles, labels, mean - standard_deviation)
	#propagate_labels(labeled_profiles, labels, 0.905576610851 - 0.258739991239)
