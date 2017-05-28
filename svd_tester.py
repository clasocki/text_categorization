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
from semantic_model import SemanticModel, InMemoryDocumentIterator, DocumentIterator
import datetime
from sklearn.ensemble import RandomForestClassifier
from training_set_expansion import getLabeledSetGensim, LocalDocumentGenerator
import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_20newsgroups
from document_classification_20newsgroups import testClassifiers
from collections import defaultdict
import pandas as pd
from optparse import OptionParser

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
	#neigh = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='cosine')
	#neigh.fit(train_set)
	clf = RandomForestClassifier(n_estimators=80)
        clf = Perceptron(n_iter=50)
	clf.fit(train_set, train_set_target)
        

        predicted = []
	test_set_target = []
	correctly_classified = 0
	all_classified = 0
	for test_elem, test_target in test_set:
                #test_profiles = semantic_model.inferProfiles([test_elem])
	        test_profile = semantic_model.inferProfile(test_elem, num_iters=10, learning_rate=0.001, regularization_factor=0.01)
		#dists, indices = neigh.kneighbors([test_profile])
		#prediction, _ = find_closest_category(train_set_target, dists[0], indices[0])
		if len(test_profile) > 0:
			prediction = clf.predict([test_profile])[0]

                        predicted.append(prediction)
			test_set_target.append(test_target)
		
			if test_target == prediction:
				correctly_classified += 1
			
			all_classified += 1

	accuracy = str(float(correctly_classified) / all_classified)
	print "Classification accuracy: %s, positive: %s, all: %s" % (accuracy, correctly_classified, all_classified)
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

	#iterative
	
	query = "SELECT profile, learned_category FROM pap_papers_view WHERE published = 1 and profile is not null and learned_category is not null"
	labeled_profiles, labels = [], []

	rowmapper = lambda row: (numpy.asarray([float(value) for value in row['profile'].split(',')]), row['learned_category'])
	with LocalDocumentGenerator(query, rowmapper) as labeled_documents:
		for profile, label in labeled_documents:
			labeled_profiles.append(profile)
			labels.append(label)
        class Doc:
		def __init__(self, tokenized_text):
			self.tokenized_text = tokenized_text
                        self.word_weights = semantic_model.calculateWordWeights(self.tokenized_text)

        rowmapper = lambda row: (row['rawtext'], row['category'])
	
	
	#gensim

	#labeled_profiles, labels, semantic_model = getLabeledSetGensim(num_features=50)
        #rowmapper = lambda row: (tokenize(row['rawtext']).split(), row['category'])
        
        query = "SELECT rawtext, category FROM pap_papers_view where published = 1 and learned_category is null"
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

def testAccuracyIter(word_profiles_in_db, train_rmse, val_rmse, current_iter, rmses, scores, 
        train_times, test_times, snapshot_file):
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
        'rec.motorcycles',
        'sci.electronics',
        'sci.med',
        'talk.politics.guns',
        'rec.autos'
    ]

    remove = ()
    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

    data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
    y_train, y_test = data_train.target, data_test.target
    iter_semantic_model = SemanticModel.load(snapshot_file, document_iterator=None, word_profiles_in_db=word_profiles_in_db)
    X_train = numpy.asarray([iter_semantic_model.inferProfile(x, num_iters=10, learning_rate=0.002, regularization_factor=0.01) for x in data_train.data])
    X_test = numpy.asarray([iter_semantic_model.inferProfile(x, num_iters=10, learning_rate=0.002, regularization_factor=0.01) for x in data_test.data])

    y_train, y_test = data_train.target, data_test.target
    clf_names, score, train_time, test_time = testClassifiers(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print score

    for i, clf_name in enumerate(clf_names):
        scores[clf_name][current_iter] = score[i]
        train_times[clf_name][current_iter] = train_time[i]
        test_times[clf_name][current_iter] = test_time[i]

    rmses['rmse - zbior treningowy'][current_iter] = train_rmse
    rmses['rmse - zbior walidacyjny'][current_iter] = val_rmse

def testAccuracyGensim(num_features, min_df, max_df):
        labeled_profiles, labels, semantic_model = getLabeledSetGensim(num_features=num_features, min_df=min_df, max_df=max_df)
        rowmapper = lambda row: (tokenize(row['rawtext']).split(), row['category'])
        
        query = "SELECT rawtext, category FROM pap_papers_view where published = 1 and learned_category is null"
	y_test, y_pred = [], []
	with LocalDocumentGenerator(query, rowmapper) as unlabeled_documents:
		y_test, y_pred = calculate_classification_accuracy(labeled_profiles, labels, unlabeled_documents, semantic_model)
		#classify(labeled_profiles, labels, unlabeled_documents)
def test():
	db = MySQLdb.connect(host='127.0.0.1', user='root',
                         passwd='1qaz@WSX', db='test')
        iterative = True
	#insert_all(db, rootdir_test, 1)
	#insert_all(db, rootdir_train, 0)

        op = OptionParser()
        op.add_option("--experiment_dir",
                      action="store", type=str, dest="experiment_dir")
        op.add_option("--num_features",
                      action="store", type=int, dest="num_features")
        op.add_option("--learning_rate",
                      action="store", type=float, dest="learning_rate")
        op.add_option("--regul_factor",
                      action="store", type=float, dest="regul_factor")
        op.add_option("--zero_weights",
                      action="store", type=float, dest="zero_weights")
        op.add_option("--doc_prof_low",
                      action="store", type=float, dest="doc_prof_low")
        op.add_option("--doc_prof_high",
                      action="store", type=float, dest="doc_prof_high")
        op.add_option("--word_prof_low",
                      action="store", type=float, dest="word_prof_low")
        op.add_option("--word_prof_high",
                      action="store", type=float, dest="word_prof_high")
        op.add_option("--decay",
                      action="store", type=float, dest="decay")
        op.add_option("--min_df",
                      action="store", type=float, dest="min_df")
        op.add_option("--max_df",
                      action="store", type=float, dest="max_df")

        argv = sys.argv[1:]
        (opts, args) = op.parse_args(argv)
        print opts.experiment_dir
        print opts.num_features
        print opts.learning_rate
        print opts.regul_factor
        print opts.zero_weights
        print opts.doc_prof_low
	print opts.doc_prof_high
        print opts.word_prof_low
	print opts.word_prof_high
        print opts.decay
        print opts.min_df
        print opts.max_df
	
	if iterative:
		accuracy_result_filename = 'accuracy_result.csv'

		start_time = time.time()

                categories = [
                    'alt.atheism',
                    'talk.religion.misc',
                    'comp.graphics',
                    'sci.space',
                    'rec.motorcycles',
                    'sci.electronics',
                    'sci.med',
                    'talk.politics.guns',
                    'rec.autos'
                ]
                data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42, remove = ())# 'footers', 'quotes'))
                data_test = fetch_20newsgroups(subset='test', categories=categories,
                                shuffle=True, random_state=42, remove = ())# 'footers', 'quotes'))

	        training_set_iterator = InMemoryDocumentIterator(data_set=data_train.data)
                #training_set_iterator = DocumentIterator(doc_filter="published = 1 and learned_category is not null", 
                #                                     document_batch_size=5000, db_window_size=5000)
                
                save_to_db = False
                rmses, scores, train_times, test_times = [defaultdict(dict) for x in xrange(4)]
                
                if not os.path.exists(opts.experiment_dir):
                    os.makedirs(opts.experiment_dir)
                model_snapshot_filename = opts.experiment_dir + 'semantic_model.snapshot'
                
                tester = lambda current_iter, train_rmse, val_rmse: testAccuracyIter(save_to_db, train_rmse, val_rmse, current_iter, 
                        rmses, scores, train_times, test_times, model_snapshot_filename)
		semantic_model = SemanticModel(document_iterator=training_set_iterator, num_features=opts.num_features, file_name=model_snapshot_filename, 
                                               learning_rate=opts.learning_rate, regularization_factor=opts.regul_factor,
                                               neg_weights=opts.zero_weights, doc_prof_low=opts.doc_prof_low, doc_prof_high=opts.doc_prof_high, 
                                               word_prof_low=opts.word_prof_low, word_prof_high=opts.word_prof_high, decay=opts.decay,
					       min_df=opts.min_df, max_df=opts.max_df, save_frequency=5, test_frequency=5, 
                                               save_model=True,  with_validation_set=True, save_to_db=save_to_db,
                                               tester=tester)
					       #tester = lambda epoch: test_accuracy(semantic_model, db, epoch, accuracy_result_filename))	
        
		try:
			semantic_model.train()
			pass
		except (KeyboardInterrupt, SystemExit):
			raise
		finally:
                        pd.DataFrame(rmses).to_pickle(opts.experiment_dir + 'rmses.pkl')
                        pd.DataFrame(scores).to_pickle(opts.experiment_dir + 'scores.pkl')
                        pd.DataFrame(train_times).to_pickle(opts.experiment_dir + 'train_times.pkl')
                        pd.DataFrame(test_times).to_pickle(opts.experiment_dir + 'test_times.pkl')
                        
			semantic_model.save(save_words=True)

			print "Training total time: " + str(time.time() - start_time)
			db.close()
	else:
		testAccuracyGensim(opts.num_features, 0.002, 0.33)


if __name__ == "__main__":
	test()
