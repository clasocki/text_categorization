import math


class TfidfModel(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __getitem__(self, bag_of_words):
        tf_sum = sum([token_freq for (token_id, token_freq) in bag_of_words.iteritems()])
        return dict((token_id,
                     self.tf(token_freq, tf_sum) * self.idf(self.dictionary.doc_freqs[token_id],
                                                            self.dictionary.num_docs))
                    for (token_id, token_freq) in bag_of_words.iteritems())

    @staticmethod
    def tf(token_freq, tf_sum):
        return 1.0 * token_freq / tf_sum

    @staticmethod
    def idf(doc_freq, total_docs, log_base=2.0):
        return math.log(1.0 * total_docs / doc_freq, log_base)