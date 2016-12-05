from semantic_model import SemanticModel, DocumentIterator
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import numpy as np
from paperity.environ import db

LABELED_DOCUMENTS_CONDITION = "published = 1 AND learned_category IS NOT NULL"
UNLABELED_DOCUMENTS_CONDITION = "published = 1 AND learned_category IS NULL"
PROFILE_INFERENCE_NUM_ITERS = 60
N_NEIGHBORS = 15
NEW_LABELS_BATCH = 50
FINAL_DOCUMENT_COUNTER_MULTIPLIER = 100
acceptable_distance = 1.0
MODEL_SNAPSHOT_FILENAME = 'semantic_model.snapshot'

def find_most_common_label(labels, indices):
	predicted = [labels[i] for i in indices]
	count = Counter(predicted)
	freq_list = count.values()
	total_most_common = freq_list.count(max(freq_list))
	most_common = count.most_common(total_most_common)
	most_common = [elem[0] for elem in most_common]

	return most_common[0]

def find_initial_label_set():
	labeled_documents = DocumentIterator(where=LABELED_DOCUMENTS_CONDITION).getAll()
	
	labeled_profiles = []
	labels = []
	
	for doc in labeled_documents:
		labeled_profiles.append(doc.profile)
		labels.append(doc.learned_category)

	return labeled_profiles, labels

def find_closest_category(labels, distances):
	closest_category = find_most_common(labels, indices)
	closest_category_indices = [i for i, label in enumerate(labels) if label == closest_category]
	average_distance = numpy.mean(distances[closest_category_indices])

	return closest_category, average_distance

def assign_category(document, category, assigned_category_counter, newly_labeled_documents):
	assigned_category_counter[category] += 1
	newly_labeled_documents.append(document)
	document.learned_category = category

	sql_update = "update pap_papers_view set learned_category = '" + str(category) + \
		"' WHERE pid = " + str(document.pid)

	db.query(sql_update)


def propagate_labels():
	model_snapshot_filename = MODEL_SNAPSHOT_FILENAME
	semantic_model = SemanticModel.load(file_name=model_snapshot_filename, is_test=False)

	final_num_docs = semantic_model.num_docs * FINAL_DOCUMENT_COUNTER_MULTIPLIER

	labeled_profiles, labels = find_initial_label_set()

	assigned_category_counter = Counter()
	newly_labeled_documents = []

	unlabeled_document_iterator = DocumentIterator(where=UNLABELED_DOCUMENTS_CONDITION)
	nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS).fit(labeled_profiles)

	for unlabeled_document in unlabeled_document_iterator:
		semantic_model.assignUntrainedDocumentProfiles([unlabeled_document]), 
			num_iters=PROFILE_INFERENCE_NUM_ITERS, initialize_profiles=True)
		distances, indices = nbrs.kneighbors(unlabeled_document.profile)

		closest_category, average_distance = find_closest_category(labels, indices)

		if average_distance <= ACCEPTABLE_DISTANCE:
			assign_category(unlabeled_document, closest_category, assigned_category_counter, newly_labeled_documents)

			if len(newly_labeled_document_counter) == NEW_LABELS_BATCH:
				semantic_model.update(newly_labeled_documents)
				semantic_model.save()
				
				for doc in newly_labeled_documents:
					labeled_profiles.append(doc.profile)
					labels.append(doc.learned_category)

				nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS).fit(labeled_profiles)

				newly_labeled_documents = []


		if semantic_model.num_docs >= final_num_docs:
			break

def propagate_labels2():
	pass

