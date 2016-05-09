import numpy
import datetime


class SVDEngine(object):
    def __init__(self, num_docs, num_words, num_features=2, profile_db=None):
        self.min_iter = 1000
        self.max_iter = 3000
        self.num_docs = num_docs
        self.num_words = num_words
        self.num_features = num_features
        self.feature_init_low = -0.01
        self.feature_init_high = 0.01
        self.svd_u = None
        self.svd_v = None
        self.min_improvement = 0.0001
        self.learning_rate = 0.002
        self.regul_factor = 0.02
        self.profile_db = profile_db

    def predict_value(self, document_id, word_id):
        return numpy.dot(self.svd_u[document_id, :], self.svd_v[:, word_id])

    def feature_training(self, documents_iterator):
        rmse = 2.0
        rmse_last = 2.0

        self.svd_u = numpy.random.uniform(low=-0.01, high=0.01, size=(self.num_docs, self.num_features))
        self.svd_v = numpy.random.uniform(low=-0.01, high=0.01, size=(self.num_words, self.num_features))
        self.svd_v = self.svd_v.T

        epoch = 0

        #while (epoch < self.min_iter or rmse_last - rmse >= self.min_improvement) and epoch < self.max_iter:
        while epoch < self.max_iter:
            squared_error = 0.0
            rmse_last = rmse
            num_values = 0

            for document_id, word_id, value in documents_iterator:
                num_values += 1

                predicted_value = self.predict_value(document_id, word_id)
                error = 1.0 * value - predicted_value
                squared_error += error * error

                for feature_id in numpy.arange(self.num_features):
                    document_feature = self.svd_u[document_id, feature_id]
                    word_feature = self.svd_v[feature_id, word_id]

                    self.svd_u[document_id, feature_id] += \
                        self.learning_rate * (error * word_feature - self.regul_factor * document_feature)
                    self.svd_v[feature_id, word_id] += \
                        self.learning_rate * (error * document_feature - self.regul_factor * word_feature)

            rmse = numpy.sqrt(squared_error / num_values)

            if epoch % 100 == 0:
                print epoch

            epoch += 1

        print "Last epoch: " + str(epoch)
        print "Last improvement: " + str(rmse_last - rmse)

    def save_complete_model_compressed(self, file_name):
        numpy.savez_compressed(file_name + '_model.npz', svd_u=self.svd_u, svd_v=self.svd_v)

    def load_complete_model_compressed(self, file_name):
        model = numpy.load(file_name + '_model.npz')
        self.svd_u = model['svd_u']
        self.svd_v = model['svd_v']

    def save_document_profile_batch(self, document_ids):
        if self.profile_db is None:
            return False

        document_profile_batch = []
        for document_id in document_ids:
            document_profile = {
                'profile_id': document_id,
                'features': self.svd_u[document_id, :],
                'snapshot_time': datetime.datetime.utcnow()
            }

            document_profile_batch.append(document_profile)

        success = self.profile_db.save_or_udate_all('document_profiles', document_profile_batch)
        return success

    def save_word_profile_batch(self, word_ids):
        if self.profile_db is None:
            return False

        word_profile_batch = []
        for word_id in word_ids:
            word_profile = {
                'profile_id': word_id,
                'features': self.svd_v[:, word_id],
                'snapshot_time': datetime.datetime.utcnow()
            }

            word_profile_batch.append(word_profile)

        success = self.profile_db.save_or_udate_all('word_profiles', word_profile_batch)
        return success

    def load_document_profile_batch(self, document_ids):
        pass

    def load_word_profiles_batch(self, word_ids):
        pass