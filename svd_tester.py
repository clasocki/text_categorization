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
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
import matplotlib.pyplot as plt
from semantic_model import SemanticModel, InMemoryDocumentIterator, DocumentIterator, tokenize
import datetime
from sklearn.ensemble import RandomForestClassifier
from training_set_expansion import getLabeledSetGensim, LocalDocumentGenerator
import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_20newsgroups
from document_classification_20newsgroups import testClassifiers, newsgroupsSet, reutersSet
from collections import defaultdict
import pandas as pd
from optparse import OptionParser
from multiprocessing import Process

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

def testProfileInference(experiment_dir):
    num_iters = [5, 10, 15, 25, 30]
    learning_rates = [0.001, 0.005, 0.01]
    regularization_factors = [0.0, 0.01, 0.1, 1.0]
    decay_rates = [0.0, 2.0, 4.0]

    params = [num_iters, learning_rates, regularization_factors, decay_rates]
    combinations = list(itertools.product(*params))

    scores = defaultdict(dict)
    try:
        for i, args in enumerate(combinations):
            start_time = time.time()
            scores['rate combination'][i] = 'iter=%s,learn_r=%s,regul_f=%s,decay=%s' % args
            testAccuracyIter(snapshot_file=experiment_dir + '/semantic_model.snapshot', current_iter=i, scores=scores, 
                    num_iters=args[0], learning_rate=args[1], regularization_factor=args[2], decay=args[3])
            print args, time.time() - start_time
    finally:
        pd.DataFrame(scores).to_pickle(experiment_dir + '/scores_profile_infer.pkl')

def testAccuracyIter(snapshot_file='semantic_model.snapshot', word_profiles_in_db=False, train_rmse=0.0, val_rmse=0.0, current_iter=0,
        rmses=defaultdict(dict), train_times=defaultdict(dict), test_times=defaultdict(dict),
        num_iters=15, learning_rate=0.002, regularization_factor=0.01, decay=0.0, zero_weights=None, positive_val_rmse=0.0,
        train_docs=None, test_docs=None, y_train=None, y_test=None, precision=defaultdict(dict), recall=defaultdict(dict), scores=defaultdict(dict),
        multilabel=False, semantic_model=None):
    
    start_time = time.time()
    X_train = None
    if not semantic_model:
        print "Hej"
        semantic_model = SemanticModel.load(snapshot_file, document_iterator=None, word_profiles_in_db=word_profiles_in_db)
        X_train = numpy.asarray([semantic_model.inferProfile(x, num_iters=num_iters, learning_rate=learning_rate, 
            regularization_factor=regularization_factor, decay=decay) for x in train_docs])
    else:
        X_train = numpy.asarray([x.profile for x in semantic_model.document_iterator.docs.values()])
    X_test = numpy.asarray([semantic_model.inferProfile(x, num_iters=num_iters, learning_rate=learning_rate, 
        regularization_factor=regularization_factor, decay=decay, zero_weights=zero_weights) for x in test_docs])
    print "Extracting profiles, ",  time.time() - start_time
    clf_names, score, train_time, test_time = testClassifiers(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, multilabel=multilabel)
    print "Testing classifiers, ",  time.time() - start_time

    print score

    for i, clf_name in enumerate(clf_names):
        if not multilabel:
            scores[clf_name][current_iter] = score[i]
        else:
            precision[clf_name][current_iter] = score[i]['precision']
            recall[clf_name][current_iter] = score[i]['recall']

        train_times[clf_name][current_iter] = train_time[i]
        test_times[clf_name][current_iter] = test_time[i]

    rmses['rmse - zbior treningowy'][current_iter] = train_rmse
    rmses['rmse - zbior walidacyjny'][current_iter] = val_rmse
    rmses['rmse - zbior pozytywny walidacyjny'][current_iter] = positive_val_rmse


def runTraining(experiment_dir, num_features, learning_rate, regularization_factor, zero_weights, doc_prof_low, doc_prof_high, 
          word_prof_low, word_prof_high, decay, min_df, max_df, term_freq_weight, num_iter, test_set_num_iters, test_zero_weights):
    start_time = time.time()

    #train_docs, test_docs, y_train, y_test, multilabel = newsgroupsSet()
    train_docs, test_docs, y_train, y_test, multilabel = reutersSet()

    training_set_iterator = InMemoryDocumentIterator(data_set=train_docs)
    #db = MySQLdb.connect(host='127.0.0.1', user='root',
    #                     passwd='1qaz@WSX', db='test')
    #training_set_iterator = DocumentIterator(doc_filter="published = 1 and learned_category is not null", 
    #                                     document_batch_size=5000, db_window_size=5000)
    
    save_to_db = False
    rmses, precision, recall, train_times, test_times = [defaultdict(dict) for x in xrange(5)]
    scores = defaultdict(dict)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    model_snapshot_filename = experiment_dir + '/semantic_model.snapshot'
    
    semantic_model = None
    tester = lambda current_iter, train_rmse, val_rmse, positive_val_rmse: testAccuracyIter(model_snapshot_filename, save_to_db, train_rmse, val_rmse, current_iter, 
            rmses, train_times, test_times, precision=precision, recall=recall, scores=scores, positive_val_rmse=positive_val_rmse, train_docs=train_docs, test_docs=test_docs,
            y_train=y_train, y_test=y_test, multilabel=multilabel, semantic_model=semantic_model, learning_rate=learning_rate, regularization_factor=regularization_factor, 
            decay=decay, num_iters=test_set_num_iters, zero_weights=test_zero_weights)
    semantic_model = SemanticModel(document_iterator=training_set_iterator, num_features=num_features, file_name=model_snapshot_filename, 
                                   learning_rate=learning_rate, regularization_factor=regularization_factor,
                                   neg_weights=zero_weights, doc_prof_low=doc_prof_low, doc_prof_high=doc_prof_high, 
                                   word_prof_low=word_prof_low, word_prof_high=word_prof_high, decay=decay,
                                   min_df=min_df, max_df=max_df, save_frequency=5, test_frequency=5, 
                                   save_model=True,  with_validation_set=True, save_to_db=save_to_db,
                                   term_freq_weight=term_freq_weight, tester=tester)
                                   #tester = lambda epoch: test_accuracy(semantic_model, db, epoch, accuracy_result_filename))	

    try:
            semantic_model.train(num_iter=num_iter)
            pass
    except (KeyboardInterrupt, SystemExit):
            raise
    finally:
            pd.DataFrame(rmses).to_pickle(experiment_dir + '/rmses.pkl')
            if not multilabel:
                pd.DataFrame(scores).to_pickle(experiment_dir + '/scores.pkl')
            else:
                pd.DataFrame(precision).to_pickle(experiment_dir + '/precision.pkl')
                pd.DataFrame(recall).to_pickle(experiment_dir + '/recall.pkl')

            pd.DataFrame(train_times).to_pickle(experiment_dir + '/train_times.pkl')
            pd.DataFrame(test_times).to_pickle(experiment_dir + '/test_times.pkl')
            
            semantic_model.save(save_words=True)

            print "Training total time: " + str(time.time() - start_time)
            #db.close()

def runMultipleTests():
    """
    args = (experiment_dir=opts.experiment_dir, num_features=opts.num_features, learning_rate=opts.learning_rate, regularization_factor=opts.regul_factor,
                                   zero_weights=opts.zero_weights, doc_prof_low=opts.doc_prof_low, doc_prof_high=opts.doc_prof_high, 
                                   word_prof_low=opts.word_prof_low, word_prof_high=opts.word_prof_high, decay=opts.decay,
          
                                   min_df=opts.min_df, max_df=opts.max_df, term_freq_weight=opts.term_freq_weight, num_iter=opts.num_iter)
    """
    num_features = [300]
    learning_rates = [0.001]
    regularization_factors = [0.01, 0.2]
    zero_weights = [3.0,]
    doc_prof_lows = [-0.01,]
    doc_prof_highs = [0.01,]
    word_prof_lows = [-0.01,]
    word_prof_highs = [-0.01,]
    decays = [2.0]
    min_dfs = [0.001, 5]
    max_dfs = [0.33,]
    term_freq_weights = ['log_normalization', ]
    num_iters = [80,]
    test_set_num_iters = [40]
    test_set_zero_weights = [0.0, 3.0,]
    params = [num_features, learning_rates, regularization_factors, zero_weights, doc_prof_lows, doc_prof_highs, 
              word_prof_lows, word_prof_highs, decays, min_dfs, max_dfs, term_freq_weights, num_iters, test_set_num_iters, test_set_zero_weights]
    combinations = list(itertools.product(*params))
    
    i_start = 282
    ps = []
    for i, args in enumerate(combinations):
        descr = str(i + i_start) + "__reuters_propValSet_test_zero_w_randEveryIter_"
        descr += "num_f=%s-learn_r=%s-regul_f=%s-zero_w=%s-doc_low=%s-doc_high=%s-word_low=%s-word_high=%s-decay=%s-min_df=%s-max_df=%s,term_w=%s-iter=%s-test_iter=%s-test_zero_w=%s" % args
        experiment_dir = 'experiments/' + descr
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        
        with open(experiment_dir + '/cmd', 'w+') as f:
            f.write(descr)
        
        #runTraining(*((experiment_dir, ) + args))
        ps.append(Process(target=runTraining, args=(experiment_dir, ) + args))
        
    
    for p in ps:
        p.start()
    
    for p in ps:
        p.join()
    

def testUsingOptions():
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
        op.add_option("--num_iter",
                      action="store", type=float, dest="num_iter")
        op.add_option("--term_freq_weight",
                      action="store", type=str, dest="term_freq_weight")

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
	
        runTraining(experiment_dir=opts.experiment_dir, num_features=opts.num_features, learning_rate=opts.learning_rate, regularization_factor=opts.regul_factor,
                                   zero_weights=opts.zero_weights, doc_prof_low=opts.doc_prof_low, doc_prof_high=opts.doc_prof_high, 
                                   word_prof_low=opts.word_prof_low, word_prof_high=opts.word_prof_high, decay=opts.decay,
                                   min_df=opts.min_df, max_df=opts.max_df, term_freq_weight=opts.term_freq_weight, num_iter=opts.num_iter) 

if __name__ == "__main__":
    #testUsingOptions()
    runMultipleTests()
    #testProfileInference(sys.argv[1])
