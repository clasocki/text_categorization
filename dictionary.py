from collections import Mapping, defaultdict
from itertools import izip


class Dictionary(Mapping):
    def __len__(self):
        return len(self.token_to_id)

    def __getitem__(self, token_id):
        if len(self.id_to_token) != len(self.token_to_id):
            self.id_to_token = dict((v, k) for k, v in self.token_to_id.iteritems())
        return self.id_to_token[token_id]

    def keys(self):
        return list(self.token_to_id.values())

    def __iter__(self):
        return iter(self.keys())

    def __init__(self, documents=None):
        self.num_docs = 0
        self.token_to_id = dict()  # token -> token id
        self.id_to_token = dict()  # token id -> token
        self.doc_freqs = defaultdict(int)  # token id -> the number of documents this token appears in
        self.doc_db_id_to_local_id = dict()  # document id in the db -> local document id

        if documents is not None:
            self.add_documents(documents)

    def add_documents(self, documents):
        for document in documents:
            self.doc_to_bag_of_words(document, allow_update=True)

    def doc_to_bag_of_words(self, document, allow_update=False):
        token_freq_map = defaultdict(int)
        for token in document.tokens:
            token_freq_map[token] += 1

        if allow_update:
            self.num_docs += 1
            self.doc_db_id_to_local_id[document.id] = len(self.doc_db_id_to_local_id)
            for token, freq in token_freq_map.iteritems():
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)
                self.doc_freqs[self.token_to_id[token]] += 1

        bag_of_words = dict((self.token_to_id[token], freq)
                            for token, freq in token_freq_map.iteritems()
                            if token in self.token_to_id)

        return bag_of_words

    def filter_tokens(self, bad_ids):
        bad_ids = set(bad_ids)
        self.token_to_id = dict((token, token_id)
                                for token, token_id in self.token_to_id.iteritems()
                                if token_id not in bad_ids)
        self.doc_freqs = dict((token_id, freq)
                              for token_id, freq in self.doc_freqs.iteritems()
                              if token_id not in bad_ids)

        self.compactify()

    def compactify(self):
        idmap = dict(izip(self.token_to_id.itervalues(), xrange(len(self.token_to_id))))

        self.token_to_id = dict((token, idmap[token_id]) for token, token_id in self.token_to_id.iteritems())
        self.doc_freqs = dict((idmap[token_id], freq) for token_id, freq in self.doc_freqs.iteritems())
        self.id_to_token = dict()