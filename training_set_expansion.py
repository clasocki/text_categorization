from semantic_model import SemanticModel, DocumentIterator
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from collections import Counter
import numpy as np
from paperity.environ import db
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import numpy
from random import randint
import gensim_tests
from nifty.text import tokenize
import itertools
import sys
#from svd_tester import test_accuracy
import MySQLdb
import time
import pandas as pd
from document_classification_20newsgroups import testClassifiers
from collections import defaultdict
import random
import math
import csv
import numbers
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
import pickle
import os
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skmultilearn.problem_transform.br import BinaryRelevance
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from skmultilearn.ensemble import RakelD
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

LABELED_DOCUMENTS_CONDITION = "published = 1 AND learned_category IS NOT NULL"
UNLABELED_DOCUMENTS_CONDITION = "published = 1 AND learned_category IS NULL"
VALIDATION_DOCUMENTS_CONDITION = "published = 0 and is_test = 1"
PROFILE_INFERENCE_NUM_ITERS = 30
N_NEIGHBORS = 15
NEW_LABELS_BATCH = 1000
FINAL_DOCUMENT_COUNT = 80000
acceptable_distance = 1.0
MODEL_SNAPSHOT_FILENAME = 'semantic_model.snapshot'
NUM_ITERS_MODEL_UPDATE = 90
DOCUMENT_BATCH_SIZE = 10000
DB_WINDOW_SIZE = 80000
N_OUTPUT_LABELS = 1

"""
def find_most_common_label(labels, n):
	counter = Counter(labels)
        freq_set = set(sorted(list(set(count.values())), reverse=True)[:n])
        
	total_most_common = freq_list.count(max(freq_list))
	most_common = count.most_common(total_most_common)
	most_common = [elem[0] for elem in most_common]

	idx = randint(0, len(most_common) - 1)
	return most_common[idx]
"""

def find_most_common_labels(labels, n):
	flatten = lambda l: [item for sublist in l for item in sublist]
	labels = flatten(labels)
	counter = Counter(labels)
	most_common_labels = [elem[0] for elem in counter.most_common(n) if elem[1] > 0.33 * len(labels)]
	return most_common_labels

def get_labeled_set():
	labeled_documents = DocumentIterator(doc_filter=LABELED_DOCUMENTS_CONDITION).getAll()
	
	labeled_profiles = []
	labels = []
	
	for doc in labeled_documents:
                if len(doc.profile) < 50:
			print doc.id, len(doc.profile)
		labeled_profiles.append(doc.profile)
		labels.append(doc.learned_category)

	return labeled_profiles, labels

def find_closest_categories(all_categories, distances, n):
        print all_categories
	closest_categories = find_most_common_labels(all_categories, n)
        print "Closest cats: " + str(closest_categories)
        if len(closest_categories) == 0:
		return [], sys.maxint 

	closest_category_indices = [i for i, categories in enumerate(all_categories) if any(category in categories for category in closest_categories)]
        if len(closest_category_indices) == 0:
		print all_categories, closest_categories
        print "distances: " + str(distances[closest_category_indices])
	average_distance = numpy.mean(distances[closest_category_indices])
	return closest_categories, average_distance

def assign_category(document_pid, categories):
        #sql_update = "update pap_papers_2 p2 join pap_papers_1 p1 on p1.id = p2.id set p2.learned_category = '" + categories + \
	#        	"' WHERE p1.pid = " + str(document_pid)
        sql_update = "update pap_papers_2 set learned_category = '" + categories + \
	        	"' WHERE id = " + str(document_pid)
        
	db.query(sql_update)
	db.commit()


def propagate_labels(labeled_profiles, labels, acceptable_distance):
	print "Label propagation..."

	semantic_model = SemanticModel.load(file_name=MODEL_SNAPSHOT_FILENAME, doc_filter=LABELED_DOCUMENTS_CONDITION)

	
	#db = MySQLdb.connect(host='localhost', user='root', passwd='1qaz@WSX', db='paperity')
	
	#semantic_model.tester = lambda epoch: test_accuracy(semantic_model, db, epoch, 'accuracy_result.csv')

	newly_labeled_documents = []

	unlabeled_document_iterator = DocumentIterator(document_batch_size=DOCUMENT_BATCH_SIZE, db_window_size=DB_WINDOW_SIZE, doc_filter=UNLABELED_DOCUMENTS_CONDITION,
                                                       convertText=semantic_model.convertText)
	#nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles)
        clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles, labels)

	stop_propagation = semantic_model.num_docs >= FINAL_DOCUMENT_COUNT
	for unlabeled_document_batch in unlabeled_document_iterator:
		if stop_propagation:
			break

		for i, unlabeled_document in enumerate(unlabeled_document_batch):
			semantic_model.inferProfiles([unlabeled_document], 
				num_iters=PROFILE_INFERENCE_NUM_ITERS, update_word_profiles=False, initialize_document_profiles=True)
			distances, indices = nbrs.kneighbors([unlabeled_document.profile])

			closest_categories, average_distance = find_closest_categories([labels[i] for i in indices[0]], distances[0], N_OUTPUT_LABELS)

			if average_distance <= acceptable_distance:
				assign_category(unlabeled_document, closest_categories, newly_labeled_documents)
				
				if len(newly_labeled_documents) == round(0.40 * semantic_model.num_docs):#NEW_LABELS_BATCH:
					print "Updating model..."
					semantic_model.document_iterator.saveDocumentProfilesToDb(newly_labeled_documents)
					semantic_model.update(newly_labeled_documents, num_iters_full_retrain=NUM_ITERS_MODEL_UPDATE, 
						num_iters_partial_retrain=PROFILE_INFERENCE_NUM_ITERS)
					
					labeled_profiles, labels = get_labeled_set()
					
					#mean, standard_deviation = calculate_statistics(labeled_profiles)
					#acceptable_distance = mean - standard_deviation
					#print mean, standard_deviation, acceptable_distance

					#nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles)
                                        clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles, labels)
					newly_labeled_documents = []


			if semantic_model.num_docs >= FINAL_DOCUMENT_COUNT:
				stop_propagation = True
				break

def prepareDocSet(doc_filter, semantic_model):
    doc_iter = getDocumentIterator1(doc_filter)
    doc_profiles, profiles_dict, doc_labels = [], defaultdict(list), []
    bows = []
    
    for doc in doc_iter:
        cats = doc.learned_category
        profile = semantic_model.inferProfile(doc.rawtext)

        for cat in cats:
            profiles_dict[cat].append(profile)
        doc_profiles.append(profile)
        doc_labels.append(cats)
        bows.append(semantic_model.dict.doc2bow(semantic_model.tokenize(doc.rawtext)))

    return numpy.asarray(doc_profiles), numpy.asarray(doc_labels), bows

def save_labels():
    initial_train_set_iter = getDocumentIterator1("published = 0 and is_test = 1")
    initial_test_set_iter = getDocumentIterator1("published = 1 and is_test = 1")
    extended_set_iter = getDocumentIterator1("is_test is null and learned_category is not null")

    iters = [(initial_train_set_iter, 'initial_train_set'), (initial_test_set_iter, 'initial_test_set'), (extended_set_iter, 'extended_set')]

    for it, setname in iters:
        doc_set = []
        for doc in it:
            doc_set.append((doc.id, doc.learned_category))
            
        pickle.dump(doc_set, open( 'dataset/backup_' + setname, "wb" ) )

def inspect_docs_as_bows():
    docs = getDocumentIterator1("learned_category = 'lit'")
    #semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_400_300000')
    semantic_model = gensim_tests.SemanticModel.load('gensim/initial_set_400_dict_initial_set')

    for doc in docs:
        words = []
        semantic_model.tokenize(doc.rawtext)
        tokenized_text = semantic_model.tokenize(doc.rawtext)
        text_bow = semantic_model.dict.doc2bow(tokenized_text)
        for word_id in text_bow:
            w = semantic_model.dict[word_id[0]]
            words.append(w)

        print doc.id, words

def prepare_doc_split():
    labeled_docs = getDocumentIterator1("is_test = 1")

    ids, labels = [], []
    for doc in labeled_docs:
        ids.append(doc.id)
        labels.append(doc.learned_category[0])

    X_train, X_test, y_train, y_test = train_test_split(ids, labels, test_size=0.3, stratify=labels) 
    
    for x in X_train:
        sql_update = "update pap_papers_1 set published = 0 WHERE id = " + str(x)
        
	db.query(sql_update)
	db.commit()
    
    for x in X_test:
        sql_update = "update pap_papers_1 set published = 1 WHERE id = " + str(x)
        
	db.query(sql_update)
	db.commit()

def test_training_and_validation_set():
    multilabel = True
    cross_validate = True
    use_extended_predictions = True
    
    #semantic_model = gensim_tests.SemanticModel.build(lambda: validation_texts, 150)
    #semantic_model = gensim_tests.SemanticModel.load('gensim/small_corpus_400_300000')
    #print semantic_model.lsi.num_topics
    #semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_250_300000')
    #semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_400_300000')
    semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_400_300000_noabove010')
    #semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_400_300000_noabove033')
    #semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_400_100000')
    #semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_400_300000_stem')
    #semantic_model = gensim_tests.SemanticModel.load('gensim/initial_set_400_dict_initial_set')
    #semantic_model = gensim_tests.SemanticModel.load('gensim/initial_set_400_dict_full_set')
   
    splits = []
    if cross_validate:
        #semantic_model = gensim_tests.SemanticModel.build(lambda: (doc.rawtext for doc in getDocumentIterator1("is_test = 1")), no_below=2, no_above=0.33, num_features=400, filepath='gensim/default')
        profiles, labels, _ = prepareDocSet("is_test = 1", semantic_model)

        for train_ids, test_ids in StratifiedKFold(n_splits=5).split(X=numpy.zeros(len(labels)), y=[x[0] for x in labels]):
            splits.append(
                    (profiles[train_ids], numpy.array([l for l in labels[train_ids]]), profiles[test_ids], numpy.array([l for l in labels[test_ids]])))
    else:
        train_pids = []
        train_filter = ""
        #train_pids = expansionStats().keys()
        #print len(train_pids)
        initial_train_set = pickle.load( open('dataset/backup_initial_train_set', "rb" ) )
        #train_pids += [pid for pid, _ in initial_train_set]
        #train_filter += 'pid in (' + ','.join([str(x) for x in train_pids]) + ')'
        train_filter = 'id in (' + ','.join([str(pid) for pid, _ in initial_train_set]) + ')'
        initial_test_set = pickle.load( open('dataset/backup_initial_test_set', "rb" ) )
        test_filter = 'id in (' + ','.join([str(pid) for pid, _ in initial_test_set]) + ')'
        #train_filter += " or "
        #train_filter += " (learned_category is not null and is_test is null)"
        #train_filter += "is_test = 1"
        #train_filter += "(published = 0 and is_test = 1)"
        #test_filter = train_filter
        #test_filter = "published = 1 and is_test = 1"
        #semantic_model = gensim_tests.SemanticModel.build(lambda: (doc.rawtext for doc in getDocumentIterator1(train_filter)), no_below=2, no_above=0.33, num_features=150, filepath='gensim/default')

        training_profiles, training_labels, _ = prepareDocSet(train_filter, semantic_model)#, multilabel)
        #training_profiles, training_labels = prepareDocSet("learned_category is not null and is_test is null", semantic_model)#, multilabel)
        validation_profiles, validation_labels, validation_bows = prepareDocSet(test_filter, semantic_model)#, multilabel) #"published = 1 and is_test = 1"
        #profiles, labels, _ = prepareDocSet("is_test = 1", semantic_model)
        print len(training_profiles)
        print len(validation_profiles)
        #training_profiles, validation_profiles, training_labels, validation_labels = train_test_split(profiles, labels, test_size=0.5, stratify=[x[0] for x in labels]) 
        """
        pickle.dump(training_profiles, open( 'dataset/X_train', "wb" ) )
        pickle.dump(training_labels, open( 'dataset/y_train', "wb" ) )
        pickle.dump(validation_profiles, open( 'dataset/X_test', "wb" ) )
        pickle.dump(validation_labels, open( 'dataset/y_test', "wb" ) )
        return
        """
        splits = [(training_profiles, training_labels, validation_profiles, validation_labels)]
    
    precisions, recalls, f1s = [], [], []
    
    cnf_matrix_overall = None
    y_predicted_overall = None
    y_test_overall = None
    
    max_labels = 2

    if not multilabel:
        max_labels = 1
    for training_profiles, training_labels, validation_profiles, validation_labels in splits:
        if not multilabel:
            training_labels = [p[0] for p in training_labels]
            validation_labels = [p[0] for p in validation_labels]
        print training_labels
        #validation_profiles = training_profiles
        #validation_labels = training_labels
        #validation_profiles = numpy.asarray([semantic_model.inferProfile(x) for x in validation_texts])
        #training_profiles = numpy.asarray([semantic_model.inferProfile(x) for x in training_texts])

        #clf_names, score_results, pred_results = testClassifiers(X_test=validation_profiles, y_test=validation_labels, 
        #                                                          X_train=training_profiles, y_train=training_labels, multilabel=multilabel)
        
        #knn.fit(training_profiles, training_labels)

        mlb = None
        if multilabel:
            mlb = MultiLabelBinarizer()
            mlb = mlb.fit(training_labels)
            training_labels = mlb.transform(training_labels)
            #non_zeros = numpy.asarray([numpy.count_nonzero(x) for x in training_labels])
            #print "Train: " + str(non_zeros)
            #print len(non_zeros[non_zeros == 0])

        #clf = LinearSVC(class_weight='balanced')
        #clf = RandomForestClassifier(n_estimators=100)
       
        #clf = KNeighborsClassifier(n_neighbors=10, metric='cosine')
        clf = LogisticRegression(C=0.1, solver='lbfgs', max_iter=100) #, class_weight='balanced')
        #clf = GaussianNB() 
        #clf = make_pipeline(StandardScaler(), clf)
        #clf = CalibratedClassifierCV(clf, method='sigmoid')
        #clf = LogisticRegression(solver='sag', max_iter=100, random_state=42, multi_class='multinomial')
        #clf = SGDClassifier(alpha=0.001, n_iter=100, loss="log")#, class_weight='balanced')
        #clf = BaggingClassifier(LogisticRegression(C=1.0, solver='lbfgs'), max_samples=0.5, max_features=0.5)
        #clf = SVC(C=0.01,probability=False)

        if multilabel:
            clf = OneVsRestClassifier(clf)
        #clf = ClassifierChain(clf)
        #clf = RakelD(classifier=clf, labelset_size=3, require_dense=[False, False])
        #clf = MLPClassifier()
        #clf = BinaryRelevance(LogisticRegression(C=1.0, solver='lbfgs'))
        clf.fit(training_profiles, training_labels)
        #print clf.classes_
        
        #probas = mlb.inverse_transform(probas)
        #probas /= probas.sum(axis=1).reshape((probas.shape[0], -1))

        #decision_values =  clf.decision_function(validation_profiles).tolist()

        allocs, max_probas = [], []
        if use_extended_predictions:
            probas = clf.predict_proba(validation_profiles)
            for proba in probas:
                max_proba_ids = numpy.argsort(proba)[-max_labels:]
                print proba[max_proba_ids]
                print sum(proba)
                #converted_probas = 100 * proba / proba.sum()
                #print converted_probas[max_proba_ids]

                local_allocs, local_max_probas = [], []
                for max_proba_id in max_proba_ids:
                    if proba[max_proba_id] < 0.6: continue
                    local_allocs.append(max_proba_id)
                    local_max_probas.append(proba[max_proba_id])
                
                if not local_allocs:
                    local_allocs.append(max_proba_ids[-1])
                    local_max_probas.append(proba[max_proba_ids[-1]])
                
                if multilabel:
                    local_allocs = mlb.classes_[local_allocs]
                else:
                    local_allocs = clf.classes_[local_allocs]
                                    
                max_probas.append(local_max_probas)
                allocs.append(local_allocs)
            pred = allocs
        else:
            pred = clf.predict(validation_profiles)
            pred = mlb.inverse_transform(pred) if multilabel else pred
        #print pred
        #print max_probas
        """
        cnt = 0
        for pl, dv, vl, vb, pr, vp in zip(max_probas, decision_values, validation_labels, validation_bows, pred, validation_profiles):
            for p in pl:
                if p < 0.2:
                    print knn.predict_proba([vp]), knn.predict([vp])
                    print p, pr, vl
                    cnt += 1
        print "Low max probas: " + str(cnt)
        """
        zero_ids = numpy.where(numpy.array([numpy.count_nonzero(p) for p in pred]) == 0)
        non_zero_ids = numpy.where(numpy.array([numpy.count_nonzero(p) for p in pred]) != 0)

        pickle.dump(numpy.array(validation_profiles)[zero_ids], open( 'dataset/zero_profiles', "wb" ) )
        pickle.dump(numpy.array(validation_labels)[zero_ids], open( 'dataset/_zero_labels', "wb" ) )
        pickle.dump(numpy.array(validation_profiles)[non_zero_ids], open( 'dataset/non_zero_profiles', "wb" ) )
        pickle.dump(numpy.array(validation_labels)[non_zero_ids], open( 'dataset/non_zero_labels', "wb" ) )

        #print mlb.inverse_transform(clf.label_binarizer_.transform(numpy.argsort(probas[3])))
        #print probas
        #print [p[p > 0.2] for p in probas]
        non_zeros = numpy.asarray([numpy.count_nonzero(x) for x in pred])
        print "Preds: " + str(non_zeros)
        #print len(non_zeros[non_zeros == 0])
        
        if not multilabel:
            y_test = validation_labels
            cnf_matrix = confusion_matrix(validation_labels, pred)
            if cnf_matrix_overall is None:
                cnf_matrix_overall = cnf_matrix
            else:
                cnf_matrix_overall += cnf_matrix

            #cnf_matrix = confusion_matrix([lbls[0] for lbls in validation_labels], [lbls[0] for lbls in allocs])
            #print clf.classes_.tolist()
            #print cnf_matrix.tolist()
            #mlb = MultiLabelBinarizer()
            #mlb = mlb.fit([[x] for x in validation_labels])
            #y_test = mlb.transform([[x] for x in validation_labels])
            #pred = mlb.transform(pred)
            pass
        else:
            #print validation_labels, allocs
            #print [x for x in validation_labels], [x for x in allocs]
            y_test = mlb.transform(validation_labels)
            pred = mlb.transform(pred)
        precision = metrics.precision_score(y_test, pred, average='micro')
        recall = metrics.recall_score(y_test, pred, average='micro')
        f1 = metrics.f1_score(y_test, pred, average='micro')
        print pred 
        print precision, recall, f1
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        if y_predicted_overall is None:
            y_predicted_overall = pred
            y_test_overall = y_test
        else: 
            y_predicted_overall = np.concatenate([y_predicted_overall, pred])
            y_test_overall = np.concatenate([y_test_overall, y_test])

        report = classification_report(y_test, pred, target_names=mlb.classes_ if multilabel else clf.classes_)
        print report

        """
        for clf_name, preds in zip(clf_names, pred_results):
            with open('training_set_lsi_allocation.csv','w') as f:
                csv_writer = csv.writer(f, delimiter=';')
                for test_label, pred_label, text_id in zip(labels, preds, labeled_docs_ids):
                    csv_writer.writerow([','.join(test_label) if multilabel else test_label, ','.join(pred_label) if multilabel else pred_label, text_id])
        """
    print precisions
    print recalls
    avg_precision = numpy.sum(precisions) / float(len(splits))
    avg_recall = numpy.sum(recalls) / float(len(splits))
    avg_f1 = numpy.sum(f1s) / float(len(splits))
    print avg_precision, avg_recall, avg_f1
    if cnf_matrix_overall is not None:
        print cnf_matrix_overall.tolist()

    print classification_report(y_test_overall, y_predicted_overall, target_names=mlb.classes_ if multilabel else clf.classes_)

def propagate_labels_gensim(labeled_profiles, labels, acceptable_distance, num_features, semantic_model, calc_distances, min_df, max_df, scores):
	newly_labeled_documents = []

	unlabeled_document_iterator = DocumentIterator(document_batch_size=DOCUMENT_BATCH_SIZE, db_window_size=DB_WINDOW_SIZE, doc_filter=UNLABELED_DOCUMENTS_CONDITION)
        validation_documents = DocumentIterator(doc_filter=VALIDATION_DOCUMENTS_CONDITION).getAll()
        validation_texts, validation_labels = [], []
        for doc in validation_documents:
            validation_texts.append(doc.tokenized_text)
            validation_labels.append(doc.learned_category[0])
        
        #nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles)
        clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles, labels)

        current_iter = 0
	stop_propagation = semantic_model.num_docs >= FINAL_DOCUMENT_COUNT
	for unlabeled_document_batch in unlabeled_document_iterator.batchIter():
                print "Retrieved document batch"
		if stop_propagation:
			break
                
                predictions = defaultdict(list)
		for i, unlabeled_document in enumerate(unlabeled_document_batch):
			profile = semantic_model.inferProfile(unlabeled_document.tokenized_text)
			if len(profile) == 0:
				#print "no elements", unlabeled_document.id
				continue
			#distances, indices = nbrs.kneighbors([profile])
                        predict_result = clf.predict_proba([profile])
                        max_idx = numpy.argmax(predict_result)
                        proba = predict_result[0][max_idx]
                        closest_category = clf.classes_[max_idx]
			#closest_categories, average_distance = find_closest_categories([labels[i] for i in indices[0]], distances[0], N_OUTPUT_LABELS)
                        if proba >= 0.7:
	                        predictions[closest_category].append(unlabeled_document)

                average_num_docs = numpy.mean([len(docs) for docs in predictions.values()])
                print [(cat, len(docs)) for cat, docs in predictions.iteritems()]
                for category, docs in predictions.iteritems():
                    newly_labeled_documents.append((category, random.sample(docs, int(math.ceil(average_num_docs)))))

                print len(newly_labeled_documents)	
                if len(newly_labeled_documents) >= round(0.40 * semantic_model.num_docs):#NEW_LABELS_BATCH:
                        print "Updating model..."
                        
                        for category, docs in newly_labeled_documents:
                            for doc in docs:
                                assign_category(doc, category)

                        labeled_profiles, labels, semantic_model = getLabeledSetGensim(num_features, min_df, max_df)
                        
                        #mean, standard_deviation = calculate_statistics(labeled_profiles, semantic_model, calc_distances)
                        #acceptable_distance = mean - standard_deviation
                        #print mean, standard_deviation, acceptable_distance
                        validation_profiles = numpy.asarray([semantic_model.inferProfile(x) for x in validation_texts])
                        clf_names, score, train_time, test_time = testClassifiers(X_train=labeled_profiles, y_train=labels, 
                                X_test=validation_profiles, y_test=validation_labels, multilabel=False)

                        print score

                        for i, clf_name in enumerate(clf_names):
                            scores[clf_name][current_iter] = score[i]

                        #nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles)
                        clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, algorithm='brute', metric='cosine').fit(labeled_profiles, labels)

                        newly_labeled_documents = []
                        current_iter += 1

			if semantic_model.num_docs >= FINAL_DOCUMENT_COUNT:
				stop_propagation = True
				break


def calculate_statistics(labeled_profiles, semantic_model, calc_distances):
	print "Calculating statistics..."

	distance_sum, counter = 0.0, 0
	last_record_offset = -1	

	#### online standard deviation
	mean, M2 = 0.0, 0.0
	####

	unlabeled_document_iterator = DocumentIterator(document_batch_size=DOCUMENT_BATCH_SIZE, db_window_size=50000, doc_filter=UNLABELED_DOCUMENTS_CONDITION)

	all_distances = []

	for unlabeled_document_batch in unlabeled_document_iterator.batchIter():
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

                print mean, standard_deviation

	
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
	def __init__(self, doc_filter, rowmapper):
                self.doc_filter = doc_filter
		self.rowmapper = rowmapper
                self.db = MySQLdb.connect(host='127.0.0.1', user='root', passwd='1qaz@WSX', db='paperity_full')
                #self.rowcount = db.select(self.query)[0]['count(1)']

        def __iter__(self):
                cursor = self.db.cursor(MySQLdb.cursors.DictCursor)#SSDictCursor
		#cursor.execute(self.query)
                current_row = 0
                batch_size = 10000
                while True:
                    query = "select p.rawtext " + \
                    "from (select id from pap_papers_view where " + self.doc_filter + " " \
                    " order by id limit " + str(batch_size) + " OFFSET " + str(current_row) + \
                    ") q join pap_papers_view p on p.id = q.id"; 
                    cursor.execute(query) 
                    rows = cursor.fetchall()
                    print "Fetched rows"
                    if not rows: break
                    for i, row in enumerate(rows):
                        yield self.rowmapper(row)
                    current_row += batch_size

                    if current_row % 10000 == 0:
                        print "Iterator: " + str(current_row)

                #for row in self.cursor:
		
	def __enter__(self):
		return self	

	def __exit__(self, type, value, tb):
                pass    
                #self.db.close()

def docRowMapper(row, multilabel):
	#tokenized_text = tokenize(row['rawtext']).split()
        #if not tokenized_text: print row['file_name'], row['learned_category'], row['rawtext']
	labels = row['learned_category'].split(',') if multilabel else row['learned_category']
	
	return row['rawtext'], labels, row['pid']


def getLabeledSetGensim(num_features, min_df, max_df, multilabel):
	sql_query = "SELECT rawtext, learned_category, pid FROM pap_papers_view where %s" % (LABELED_DOCUMENTS_CONDITION,)
        
	labeled_profiles, labels, semantic_model, labeled_docs_ids = [], [], None, []
        with LocalDocumentGenerator(sql_query, lambda row: docRowMapper(row, multilabel)) as labeled_docs:
		training_start_time = time.time()

                min_document_count = (min_df
                    if isinstance(min_df, numbers.Integral)
                    else min_df * len(labeled_docs))
                print len(labeled_docs)
                max_document_count = (max_df 
                    if isinstance(max_df, numbers.Integral) 
                    else max_df * len(labeled_docs))

		semantic_model = gensim_tests.SemanticModel.build((text for text, _, _ in labeled_docs if text), num_features)

		print "Model training time: " + str(time.time() - training_start_time)

		for text, label, doc_id in labeled_docs:
			if text:
				profile = semantic_model.inferProfile(text)
				if profile:
					labeled_profiles.append(profile)
					labels.append(label)
                                        labeled_docs_ids.append(doc_id)

	return labeled_profiles, labels, semantic_model, labeled_docs_ids

def getDocumentIterator1(doc_filter):
    doc_iterator = DocumentIterator(db_window_size=10000, doc_filter=doc_filter)
    return (doc for doc in doc_iterator.getAllInBatches() if doc.rawtext)

    #sql_query = "SELECT rawtext FROM pap_papers_view where %s" % (doc_filter,)
    #return LocalDocumentGenerator(doc_filter, lambda row: row['rawtext'])


def getDocumentIterator2():
    sql_query = "select rawtext from pap_papers_view where (published = 1) or (published = 0 and is_test = 1)"

    doc_iterator = LocalDocumentGenerator(sql_query, lambda row: row['rawtext'])
    
    return doc_iterator

def testModelBuilding():
    start_time = time.time()
    #doc_filter = "is_test = 1"
    doc_filter = "(published = 1) or (published = 0 and is_test = 1)"

    semantic_model = gensim_tests.SemanticModel.build(lambda: (doc.rawtext for doc in getDocumentIterator1(doc_filter)), no_below=2, no_above=0.1, keep_n=300000, num_features=400, filepath='gensim/full_corpus_400_300000_noabove010')

    print "Model building finished: " + str(time.time() - start_time)

def propagateLabelsRowMapper(row):
    return row['rawtext'], row['pid']

#@profile
def propagateLabels():
    multilabel = True
    train_iter = getDocumentIterator1("(is_test = 1)")
    #train_iter = getDocumentIterator1("(published = 0 and is_test = 1)")
    #train_iter = getDocumentIterator1("(learned_category is not null and is_test is null) or (published = 0 and is_test = 1)") #"is_test = 1") # published = 0 and is_test = 1
    #validation_iter = getDocumentIterator1("published = 1 and is_test = 1")
    #semantic_model = gensim_tests.SemanticModel.load('gensim/initial_set_400_dict_initial_set')
    #semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_250_300000')
    semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_400_300000')
    #semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_400_300000_noabove010')
    #semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_400_300000_noabove033')
    #semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_400_100000')
    #semantic_model = gensim_tests.SemanticModel.load('gensim/initial_set_400_dict_full_set')
    #semantic_model = gensim_tests.SemanticModel.build(lambda: (doc.rawtext for doc in getDocumentIterator1("is_test = 1")), no_below=0.001, no_above=0.33, num_features=400, filepath='gensim/default')

    train_profiles, profiles_dict, train_labels = [], defaultdict(list), []
    for doc in train_iter:
        cats = doc.learned_category
        profile = semantic_model.inferProfile(doc.rawtext)

        for cat in cats:
            profiles_dict[cat].append(profile)
        train_profiles.append(profile)
        train_labels.append(cats)

    max_labels = 2
    #training_profiles, validation_profiles, training_labels, validation_labels = train_test_split(profiles, labels, test_size=0.5, stratify=[x[0] for x in labels]) 

    #clf = KNeighborsClassifier(n_neighbors=10, algorithm='brute', metric='cosine')
    #clf = SGDClassifier(alpha=0.001, n_iter=1000, loss="log", class_weight='balanced')
    clf = LogisticRegression(C=0.001, solver='lbfgs', max_iter=100)
    #clf = RandomForestClassifier(n_estimators=100)

    clf = OneVsRestClassifier(clf)
    mlb = MultiLabelBinarizer()
    mlb = mlb.fit(train_labels)
    train_labels = mlb.transform(train_labels)
    print mlb.classes_

    clf.fit(train_profiles, train_labels)
    new_lbl_allocs = defaultdict(list)
    #allocs_file = ""
    #confidencs_scores_file = ""
    allocs_file =  "allocs_fullcorpus_250_0_001_first_iter.pkl"
    confidence_scores_file = 'confidence_scores_250_0_001_first_iter.pkl'
    #if os.path.isfile(allocs_file):
    #    new_lbl_allocs = pickle.load( open(allocs_file, "rb" ) )
    processed_docs = set()
    for cat, allocs in new_lbl_allocs.iteritems():
        for proba, pid in allocs:
            processed_docs.add(pid)
    print "Classifier learned"
    current_doc = 1
    confidence_scores = defaultdict(list)
    
    start_time = time.time()
    print "Propagation started"
    current_iter_time = time.time()
    for doc in getDocumentIterator1("published = 1 and is_test is null"): #learned_category is null and published = 1"): #"published = 1 and is_test is null"):
        text = doc.rawtext
        pid = doc.pid
        current_doc += 1
        #if pid in processed_docs: continue
 #       try:
        if not text: continue
        profile = semantic_model.inferProfile(text)
        if not profile.any() or len(profile) < semantic_model.lsi.num_topics: continue
        #pred = clf.predict([profile])[0]
        #mean_dists = pairwise_distances([profile], profiles_dict[pred], metric='cosine').sum() / float(len(profiles_dict[pred]) + 1)
        #new_lbl_allocs[pred].append((pred, pid, mean_dists))
        
        probas = clf.predict_proba([profile])[0]
        max_proba_ids = numpy.argpartition(probas, -max_labels)[-max_labels:]
        current_doc_allocs = []

        for max_proba_id in max_proba_ids:
            class_ = mlb.inverse_transform(clf.label_binarizer_.transform([max_proba_id]))[0][0]
            confidence_scores[class_].append(probas[max_proba_id])
            if probas[max_proba_id] >= 0.3:
                #class_ = clf.classes_[max_proba_id]
                mean_dists = pairwise_distances([profile], profiles_dict[class_], metric='cosine').sum() / float(len(profiles_dict[class_]))
                #class_ = mlb.classes_[max_proba_id]
                new_lbl_allocs[class_].append((mean_dists, probas[max_proba_id], pid))
        
        if current_doc % 1000 == 0:
            pickle.dump(new_lbl_allocs, open( allocs_file, "wb" ) )
            pickle.dump(confidence_scores, open( confidence_scores_file, "wb" ) )

            print current_doc, time.strftime("%H:%M:%S"),  time.time() - current_iter_time
            current_iter_time = time.time()
#        except Exception as ex:
#            print ex
#            raise ex
#            pass

    print "Overall time: ", time.time() - start_time

def testLabelPropagation():
    from sklearn.semi_supervised import LabelSpreading
    from sklearn import preprocessing
    label_enc = preprocessing.LabelEncoder()

    label_prop_model = LabelSpreading(kernel='knn')
    train_iter = getDocumentIterator1("published = 0 and is_test = 1")
    validation_iter = getDocumentIterator1("published = 1 and is_test = 1")
    semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_300000')
    all_profiles, labels = [], []
    propagation_labels = []
    for doc in train_iter:
        all_profiles.append(semantic_model.inferProfile(doc.rawtext))
        labels.append(doc.learned_category[0])
        propagation_labels.append(doc.learned_category[0])

    label_enc.fit(propagation_labels)
    propagation_labels = label_enc.transform(propagation_labels).tolist()

    for doc in validation_iter:
        all_profiles.append(semantic_model.inferProfile(doc.rawtext))
        labels.append(doc.learned_category[0])
        propagation_labels.append(-1)
    print propagation_labels
    print "Fitting"
    label_prop_model.fit(all_profiles, propagation_labels)
    output_labels = label_prop_model.transduction_
    for propagated, orig in zip(label_enc.inverse_transform(output_labels), labels):
        print propagated, orig


def expansionStats(save=False):
    paperity_df = pd.read_csv('2016-04-23_fulldump.csv', sep=';')
    #print paperity_df.columns
    lbls_file =  "allocs_fullcorpus_knn_first_iter.pkl"
    lbls = pickle.load(open(lbls_file, "rb"))

    allocs_per_doc = defaultdict(set)
    filtered_allocs_dict = dict()
    for lbl, allocs in lbls.iteritems():
        filtered_allocs = [a for a in allocs if a[1] >= 0.6]
        #filtered_allocs = sorted(filtered_allocs, key=operator.itemgetter(0))[: int(0.1 * len(filtered_allocs))]
#        print lbl, len(filtered_allocs)
        filtered_allocs_dict[lbl] = filtered_allocs
   
    
    least_represented_allocs = sorted([len(l) for l in filtered_allocs_dict.values()])[:3]
    print least_represented_allocs
    avg_num_allocs = numpy.mean(least_represented_allocs)
    s = 0
    for lbl in filtered_allocs_dict.keys():
        #filtered_allocs_dict[lbl] = filtered_allocs_dict[lbl][:int(3 * avg_num_allocs)]
        print lbl, len(filtered_allocs_dict[lbl])
        s += len(filtered_allocs_dict[lbl])

    #print s
    for lbl, allocs in filtered_allocs_dict.iteritems():
        for alloc in allocs:
            doc_pid = alloc[2]
            allocs_per_doc[doc_pid].add(lbl)
   
    #return allocs_per_doc
    pid_id_map = dict()
    for row in db.select("select pid, id from pap_papers_view where is_test is null"):
        pid_id_map[row['pid']] = row['id']

    multi_lbls = defaultdict(int)
    for doc_pid, cats_set in allocs_per_doc.iteritems():
        cats = ','.join(cats_set)
        #assign_category(pid_id_map[doc_pid], cats)
        if len(cats_set) > 1:
            multi_lbls[cats] += 1

    for pair in multi_lbls.iteritems():
        print pair

    allocs_df = pd.DataFrame([(str(doc_id) + '.pdf', ','.join(cats)) for (doc_id, cats) in allocs_per_doc.iteritems()])
    allocs_df.columns = ['pdf_name', 'categories']
    allocs_df.merge(paperity_df, on='pdf_name', how='left')[['pid', 'url_paperity', 'title', 'categories']].to_csv('allocs_fullcorpus_knn_first_iter.csv', sep=';')

def labeledSetStats():
    train_iter = getDocumentIterator1("learned_category is not null")
    
    categories_dict = defaultdict(int)
    for doc in train_iter:
        categories = doc.learned_category
        for cat in categories:
            categories_dict[cat] += 1

    for cat, count in categories_dict.iteritems():
        print cat, ", ", count

def calcDistances():
    train_iter = getDocumentIterator1("published = 1 and is_test = 1")
    semantic_model = gensim_tests.SemanticModel.load('gensim/full_corpus_400_300000')
    profiles = defaultdict(list)
    
    for doc in train_iter:
        category = doc.learned_category[0]
        profile = semantic_model.inferProfile(doc.rawtext)
        profiles[category].append(profile)

    for cat, profs in profiles.iteritems():
        other_profs = []
        for other_c, other_p in profiles.iteritems():
            if cat != other_c:
                other_profs.extend(other_p)

        this_cat_dists_sum = pairwise_distances(profs, metric='cosine').sum() / float(2)
        this_cats_mean = this_cat_dists_sum / float(len(profs))
        other_cat_dists_sum = pairwise_distances(profs, other_profs, metric='cosine').sum()
        other_cats_mean = other_cat_dists_sum / float(len(profs) * len(other_profs))

        print cat, this_cats_mean, other_cats_mean

if __name__ == "__main__":
	iterative = False
        scores = defaultdict(dict)
        #labeledSetStats()
        #save_labels()
        #prepare_doc_split()
        #propagateLabels()
        #testModelBuilding()
        test_training_and_validation_set()
        #testLabelPropagation()
        #calcDistances()
        #expansionStats()
        #inspect_docs_as_bows()
"""
        try:
            if iterative:
                    semantic_model = SemanticModel.load(file_name=MODEL_SNAPSHOT_FILENAME, doc_filter=LABELED_DOCUMENTS_CONDITION,
                                                        learning_rate=0.005, regularization_factor=0.01)
                    labeled_profiles, labels = get_labeled_set()

                    #mean, standard_deviation = calculate_statistics(labeled_profiles, semantic_model, calcDistancesIter)
                    #print mean, standard_deviation, mean - standard_deviation
                    mean, standard_deviation = 0.563052939071, 0.153272002339
                    acceptable_distance = mean - standard_deviation
                    propagate_labels(labeled_profiles, labels, acceptable_distance)
            
                    #propagate_labels(labeled_profiles, labels, 0.905576610851 - 0.258739991239)
            else:
                    num_features = 150
                    min_df = 0.001
                    max_df = 0.33
                    labeled_profiles, labels, semantic_model = getLabeledSetGensim(num_features, min_df, max_df)
                    print "Len labeled: ", len(labeled_profiles) 		
                    #mean, standard_deviation = calculate_statistics(labeled_profiles, semantic_model, calcDistancesGensim)
                    #print mean, standard_deviation
                    mean, standard_deviation = 0.923287365338, 0.0989641845609

                    acceptable_distance = mean - standard_deviation

                    propagate_labels_gensim(labeled_profiles, labels, acceptable_distance, num_features, semantic_model, calcDistancesGensim, min_df, max_df, scores)

        finally:
            db.disconnect()
            pd.DataFrame(scores).to_pickle('set_expansion_`scores.pkl')
"""
