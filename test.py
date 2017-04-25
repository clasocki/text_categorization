import numpy
from semantic_model import DocumentIterator
import time

LABELED_DOCUMENTS_CONDITION = "published = 1 AND learned_category IS NOT NULL"

class A(object):
    def __init__(self):
        self.a = numpy.asarray(xrange(10))

def f(arr):
   arr += 10

a = A()
print a.a
map(lambda d: f(d.a), [a])
print a.a

def f1(doc, labeled_profiles, labels):
    if len(doc.profile) < 50:
        print doc.id, len(doc.profile)
    labeled_profiles.append(doc.profile)
    labels.append(doc.learned_category)


def get_labeled_set():
    labeled_documents = DocumentIterator(where=LABELED_DOCUMENTS_CONDITION).getAll()
	
    labeled_profiles = []
    labels = []
    
    start = time.time()
    #map(lambda d: f1(d, labeled_profiles, labels), labeled_documents)
    for doc in labeled_documents:
        f1(doc, labeled_profiles, labels)
    print time.time() - start

    return labeled_profiles, labels

get_labeled_set()
