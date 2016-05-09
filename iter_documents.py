import re
import os
from helpers import Document

DOCS_HCI_GRAPHS = ["Human machine interface for lab abc computer applications",
                   "A survey of user opinion of computer system response time",
                   "The EPS user interface management system",
                   "System and human system engineering testing of EPS",
                   "Relation of user perceived response time to error measurement",
                   "The generation of random binary unordered trees",
                   "The intersection graph of paths in trees",
                   "Graph minors IV Widths of trees and well quasi ordering",
                   "Graph minors A survey"]


# "Trees are special cases of graphs so whatever works for a general graph works for trees",
# "In mathematics, and more specifically in graph theory, a tree"
# " is an undirected graph in which any two vertices are connected by exactly one path, paths"]

DOCS_TWO_GROUPS = [
    "A B C",
    "A B C C C F F B B",
    "A A A A B B B B C C C C",
    "C C C C C A A A A A B B B B B",
    "D E D E D E D E",
    "E D E D E D E D E D",
    "D D E E"
]


def tokenize(document, lower=False):
    tokens = re.split('\W+', document)

    return [token.lower() if lower else token for token in tokens]


def iter_documents(top_dir):
    document_id = 0
    for root, dirs, files in os.walk(top_dir):
        for file_name in filter(lambda x: x.endswith('.txt'), files):
            document = open(os.path.join(root, file_name)).read()
            yield Document(id=document_id, tokens=tokenize(document, lower=True))
            document_id += 1


def iter_documents_hci_graphs():
    document_id = 0
    for doc in DOCS_HCI_GRAPHS:
        yield Document(id=document_id, tokens=tokenize(doc, lower=True))
        document_id += 1


def iter_documents_two_explicit_groups():
    document_id = 0
    for doc in DOCS_TWO_GROUPS:
        yield Document(id=document_id, tokens=tokenize(doc, lower=True))
        document_id += 1


def iter_documents_db_table(db, sql):
    cursor = db.cursor()
    cursor.execute(sql)

    for paper_row in cursor:
        document_id = paper_row[0]
        txt = paper_row[1]
        yield Document(id=document_id, tokens=tokenize(txt, lower=True))