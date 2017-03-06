import os
import MySQLdb
import sys
from sklearn import cluster
import numpy
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, ward, dendrogram, fcluster
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import gensim_tests
from nifty.text import tokenize
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
import matplotlib.pyplot as plt
from semantic_model import SemanticModel, DocumentIterator
import datetime
from sklearn.ensemble import RandomForestClassifier
from training_set_expansion import getLabeledSetGensim, LocalDocumentGenerator
import time
from sklearn.multiclass import OneVsRestClassifier

rootdir = '/home/clasocki/20news-bydate/'
rootdir_test = rootdir + '20news-bydate-test'
rootdir_train = rootdir + '20news-bydate-train'

def insert_all(db, rootdir, is_test):
	cursor = db.cursor()

	all_documents = 0
	for subdir, dirs, files in os.walk(rootdir):
		for f in files:
			all_documents += 1

	current_document = 1
	errors = []
	try:
		for subdir, dirs, files in os.walk(rootdir):
			for f in files:
				path = os.path.join(subdir, f)
				category = path.split('/')[-2]

				print str(current_document) + " / " + str(all_documents)
				current_document += 1

				try:    		
					with open(path, 'r') as f_obj:
						rawtext = f_obj.readlines()

						num_lines = 0
						for l in rawtext:
							if l.startswith('Lines:'):
								num_lines = int(l[len('Lines: '):])
								break

						rawtext = rawtext[-num_lines:]
						rawtext = ''.join(rawtext)
						rawtext = rawtext.replace("'", "''")
						print rawtext[:20]
						query = "INSERT INTO pap_papers_view(rawtext, category, is_test, file_name) VALUES('" + \
							rawtext + "', '" + category + "', " + str(is_test) + ", " + str(f) + ")"

						cursor.execute(query)
						db.commit()
				except:
					print sys.exc_info()
					errors.append(path)

	finally:
		with open('doc_insert' + str(is_test) + '.err', 'w') as fp:
			for error in errors:
				fp.write(error + "\n") 

def get_documents(db, where, num_features):
	cursor = db.cursor()

	query = "SELECT COUNT(1) FROM pap_papers_view WHERE " + where

	cursor.execute(query)
	profile_count = int(cursor.fetchone()[0])

	query = "SELECT profile, category, rawtext, id FROM pap_papers_view WHERE " + where
	
	cursor.execute(query)

	profiles = numpy.zeros((profile_count, num_features))

	labels = []
	rawtexts = []
	current_row = 0
	for row in cursor:
		profile = row[0]
		profile = numpy.asarray([float(value) for value in profile.split(',')])
		profiles[current_row, :] = profile

		category = row[1]
		labels.append(category)

		rawtext = row[2]
		rawtexts.append(rawtext)

		current_row += 1

	return profiles, labels, rawtexts

def print_clustering_results(original_labels, derived_labels):
	clustering_classes = dict()
	with open('clustering_results.out', 'w') as f:
		for original_label, derived_label in zip(original_labels, derived_labels):
			if original_label in clustering_classes:
				clustering_classes[original_label][derived_label] += 1
			else:
				clustering_classes[original_label] = [0, 0, 0]

			#f.write(str(derived_label) + '\t' + str(original_label) + '\n')

		for clustering_class, decision_distr in clustering_classes.iteritems():
			f.write(str(clustering_class) + '\t' + str(decision_distr) + '\n')

def perform_clustering(profiles, original_labels):
	k_means = cluster.KMeans(n_clusters=50)
	k_means.fit(profiles) 
	
	print_clustering_results(original_labels, k_means.labels_)

#@profile
def calculate_classification_accuracy(train_set, train_set_target, test_set, semantic_model, num_labels=3):
	neigh = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='cosine')
	neigh.fit(train_set)
	rfClf = RandomForestClassifier(n_estimators=80)
	rfClf.fit(train_set, train_set_target)

        predicted = []
	test_set_target = []
	correctly_classified = 0
	all_classified = 0
	for test_elem, test_target in test_set:
		#test_profiles = semantic_model.inferProfiles([test_elem])
                test_profiles = semantic_model.inferProfiles([test_elem], initialize_document_profiles=True, num_iters=10, update_word_profiles=False)
		test_profiles = list(test_profiles)
		#dists, indices = neigh.kneighbors([test_profile])
		#prediction, _ = find_closest_category(train_set_target, dists[0], indices[0])
		#print next(test_profiles)
		#print test_target
		#print len(list(test_profiles)[0]) > 0, file_name, test_target
                test_profile = test_profiles[0]
                #print test_profile
		if len(test_profile) > 0:
			prediction = rfClf.predict(test_profiles)[0]
                        #predictions = predictions.argpartition(num_labels)[:num_labels]

                        predicted.append(prediction)
			test_set_target.append(test_target)
		
			if test_target == prediction:
				correctly_classified += 1
			
			all_classified += 1

	accuracy = str(float(correctly_classified) / all_classified)
	print "Classification accuracy: " + accuracy
	#print "Accuracy score: " + str(accuracy_score(test_set_target, predicted))

	return test_set_target, predicted

def shuffle_set(data_set, target_set):
	new_data_set_ids = range(len(data_set))
	shuffle(new_data_set_ids)

	new_data_set = numpy.zeros(data_set.shape)
	new_target_set = numpy.zeros(len(target_set))
	for id, sh_id in enumerate(new_data_set_ids):
		new_data_set[id, :] = data_set[sh_id, :]
		new_target_set[id] = target_set[sh_id]

	return new_data_set, new_target_set

def normalize_confusion_matrix(cm):
	return cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def classify(train_set, train_labels, test_set):
	rfClf = RandomForestClassifier(n_estimators=100)
	nghClf = KNeighborsClassifier(n_neighbors=10)

	print "Fitting random forest..."
	rfClf.fit(train_set, train_labels)

	rfScore = rfClf.score(test_samples, true_labels)

	print "Random forest score: " + str(rfScore)

	print "Fitting k nearest neighbors...."
	nghClf.fit(train_set, train_labels)

	nghScore = nghClf.score(test_samples, true_labels)

	print "Nearest neighbors score: " + str(nghScore)

#@profile
def test_accuracy(semantic_model, db, current_epoch, result_filename):
	#unlabeled_document_iterator = DocumentIterator(where="published = 1 and learned_category is null")
	#unlabeled_document_iterator = DocumentIterator(where="published = 1 and year is null")
	#document_iterator.saveDocumentProfilesToFile(file_name='document_profiles.train')   

	
	#unlabeled_documents = unlabeled_document_iterator.getAll()
	
	#semantic_model.inferProfiles(unlabeled_documents, update_word_profiles=False, num_iters=10, initialize_document_profiles=True)
	#unlabeled_document_iterator.saveDocumentProfilesToDb(unlabeled_documents)
	

	#train_profiles, train_original_labels, train_rawtexts = get_documents(db, where="published = 1 and profile is not null and learned_category is not null", num_features=semantic_model.num_features)
	#test_profiles, test_original_labels, test_rawtexts = get_documents(db, where="published = 1 and profile is not null and learned_category is null", num_features=semantic_model.num_features)
	
	#train_profiles, train_original_labels, train_rawtexts = get_documents(db, where="published = 1 and learned_category is not null and year is not null", num_features=semantic_model.num_features)
	#test_profiles, test_original_labels, test_rawtexts = get_documents(db, where="published = 1 and learned_category is not null and year is null", num_features=semantic_model.num_features)

	"""
	train_profiles, train_original_labels, train_rawtexts = get_documents(db, where="published = 1 and is_test = 0", num_features=50)
	test_profiles, test_original_labels, test_rawtexts = get_documents(db, where="published = 1 and is_test = 1", num_features=50)
	
	db.commit()

	train_set_target = labels_text_to_id(train_original_labels)
	test_set_target = labels_text_to_id(test_original_labels)
	#print test_profiles[10:]
	
	train_tokenized_texts = [tokenize(rawtext).split() for rawtext in train_rawtexts]
	test_tokenized_texts = [tokenize(rawtext).split() for rawtext in test_rawtexts]
	semantic_model = gensim_tests.SemanticModel.build(train_tokenized_texts, 50, 0.002 * len(train_tokenized_texts), 0.5 * len(train_tokenized_texts))
	train_profiles = semantic_model.inferProfiles(train_tokenized_texts)
	test_profiles = semantic_model.inferProfiles(test_tokenized_texts)

	train_set = train_profiles
	test_set = test_profiles
	"""
	"""

	profiles = train_profiles
	distArray = 1 - pairwise_distances(profiles, metric='cosine')
	print "Mean: " + str(numpy.mean(distArray))

	linkage_matrix = ward(distArray)
	derived_labels = fcluster(linkage_matrix, 3, criterion='maxclust')
	text_clf = SVC(kernel="linear", C=0.025)
	text_clf = text_clf.fit(train_set, train_set_target)

	print "Score: " + str(text_clf.score(test_set, test_set_target))
	"""

	#iterative
	
	query = "SELECT profile, learned_category FROM pap_papers_view WHERE published = 1 and profile is not null and learned_category is not null"
	labeled_profiles, labels = [], []

	rowmapper = lambda row: (numpy.asarray([float(value) for value in row['profile'].split(',')]), row['learned_category'])
	with LocalDocumentGenerator(query, rowmapper) as labeled_documents:
		for profile, label in labeled_documents:
			labeled_profiles.append(profile)
			labels.append(label)
	
	#gensim

	#labeled_profiles, labels, semantic_model = getLabeledSetGensim(num_features=50)
	#print labeled_profiles, labels
	class Doc:
		def __init__(self, tokenized_text):
			self.tokenized_text = tokenized_text

	query = "SELECT rawtext, category FROM pap_papers_view where published = 1 and learned_category is null"
	rowmapper = lambda row: (Doc(tokenize(row['rawtext']).split()), row['category'])
	#rowmapper = lambda row: (tokenize(row['rawtext']).split(), row['category'])
	y_test, y_pred = [], []
	with LocalDocumentGenerator(query, rowmapper) as unlabeled_documents:
		y_test, y_pred = calculate_classification_accuracy(labeled_profiles, labels, unlabeled_documents, semantic_model)
		#classify(labeled_profiles, labels, unlabeled_documents)
 
	# Compute confusion matrix
	cnf_matrix = confusion_matrix(y_test, y_pred)
	print("epoch: " + str(current_epoch))
	cnf_matrix_diag = normalize_confusion_matrix(cnf_matrix).diagonal()
	accuracy = accuracy_score(y_test, y_pred)
	with open(result_filename, "a") as results:
		results.write(",".join(map(str, cnf_matrix_diag)) + "," + str(accuracy) + "\n")

	numpy.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	#plt.figure()
	#plot_confusion_matrix(cnf_matrix, classes=class_names,
	#                      title='Confusion matrix, without normalization')

	# Plot normalized confusion matrix
	#plt.figure()
	#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
	#                      title='Normalized confusion matrix')

	#plt.show()

def testAccuracyGensim():
	pass

if __name__ == "__main__":
	db = MySQLdb.connect(host='127.0.0.1', user='root',
                         passwd='1qaz@WSX', db='test')

	#insert_all(db, rootdir_test, 1)
	#insert_all(db, rootdir_train, 0)

		
	accuracy_result_filename = 'accuracy_result.csv'
	model_snapshot_filename = 'semantic_model.snapshot'
	num_features = 50
	
	
	#semantic_model = SemanticModel.load(model_snapshot_filename, where="published = 1 and learned_category is not null")
	#semantic_model.tester = lambda epoch: test_accuracy(semantic_model, db, epoch, accuracy_result_filename)
	#semantic_model.tester(10)
	
	start_time = time.time()
	
	#import cProfile
	#cProfile.run('test_accuracy(None, db, 10, accuracy_result_filename)', 'teststats')
	#test_accuracy(None, db, 10, accuracy_result_filename)

	
        	
	semantic_model = SemanticModel(num_features=num_features, file_name=model_snapshot_filename, 
								   #where="published = 1 and learned_category is not null", min_df=20, max_df=0.33)
								   where="published = 1 and learned_category is not null", min_df=0.002, max_df=0.33)
								   #tester = lambda epoch: test_accuracy(semantic_model, db, epoch, accuracy_result_filename))
	
	#import cProfile
	#pr = cProfile.Profile()
	#pr.enable()
        
	try:
		semantic_model.train()
		#cProfile.run('semantic_model.train()', 'teststats')
		pass
	except (KeyboardInterrupt, SystemExit):
		semantic_model.save(save_words=True)
		print "Saved"
		raise
	finally:
                """
		pr.disable()
		pr.dump_stats('teststats1')
                """
		print "Training total time: " + str(time.time() - start_time)
		db.close()
	
