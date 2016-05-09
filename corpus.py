from dictionary import Dictionary
from tfidf import TfidfModel


class Corpus(object):
    def __init__(self, document_iterator, stop_words):
        self.document_iterator = document_iterator
        self.stop_list = stop_words
        self.dictionary = Dictionary(document_iterator())
        self.tfidf_model = TfidfModel(self.dictionary)
        stop_ids = [self.dictionary.token_to_id[stop_word] for stop_word in self.stop_list
                    if stop_word in self.dictionary.token_to_id]
        once_ids = [token_id for token_id, doc_freq in self.dictionary.doc_freqs.iteritems() if doc_freq == 1]
        self.dictionary.filter_tokens(stop_ids + once_ids)

    def __iter__(self):
        for document in self.document_iterator():
            # yield self.dictionary.doc_to_bag_of_words(tokens)
            #yield doc_to_vec(len(self.dictionary.items()), self.dictionary.doc_to_bag_of_words(document))
            #yield doc_to_vec(len(self.dictionary.items()),
            #                 self.tfidf_model[self.dictionary.doc_to_bag_of_words(document)])
            converted_document = self.dictionary.doc_to_bag_of_words(document)
            converted_document = self.tfidf_model[converted_document]

            word_count = len(self.dictionary.items())
            for word_id in xrange(word_count):
                if word_id in converted_document:
                    yield document.id, word_id, converted_document[word_id]
                else:
                    yield document.id, word_id, 0


            #for (word_id, value) in converted_document.iteritems():
            #    yield document.id, word_id, value


def doc_to_vec(term_count, term_freqs):
    doc_vector = [0] * term_count
    for (word_id, freq) in term_freqs.iteritems():
        doc_vector[word_id] = freq

    return doc_vector