import numpy
import datetime


class SVDEngine(object):

    MIN_ITER = 1000

    def __init__(self, num_docs, num_words, num_features=2, profile_db=None):
        self.min_iter = 1000
        self.max_iter = 3000
        self.num_docs = num_docs
        self.num_words = num_words
        self.num_features = num_features
        self.feature_init_low = -0.01
        self.feature_init_high = 0.01
        self.document_profiles = None
        self.word_profiles = None
        self.min_improvement = 0.0001
        self.learning_rate = 0.002
        self.regul_factor = 0.02
        self.profile_db = profile_db

    def predict_value(self, document_id, word_id):
        return numpy.dot(self.document_profiles[document_id, :], self.word_profiles[:, word_id])

    def feature_training(self, documents_iterator):
        rmse = 2.0
        rmse_last = 2.0

        self.document_profiles = numpy.random.uniform(low=-0.01, high=0.01, size=(self.num_docs, self.num_features))
        self.word_profiles = numpy.random.uniform(low=-0.01, high=0.01, size=(self.num_words, self.num_features))
        self.word_profiles = self.word_profiles.T

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
                    document_feature = self.document_profiles[document_id, feature_id]
                    word_feature = self.word_profiles[feature_id, word_id]

                    self.document_profiles[document_id, feature_id] += \
                        self.learning_rate * (error * word_feature - self.regul_factor * document_feature)
                    self.word_profiles[feature_id, word_id] += \
                        self.learning_rate * (error * document_feature - self.regul_factor * word_feature)

            rmse = numpy.sqrt(squared_error / num_values)

            if epoch % 100 == 0:
                print epoch

            epoch += 1

        print "Last epoch: " + str(epoch)
        print "Last improvement: " + str(rmse_last - rmse)

    def save_model_compressed(self, file_name):
        numpy.savez_compressed(file_name + '_model.npz', svd_u=self.document_profiles, svd_v=self.word_profiles)

    def load_model_compressed(self, file_name):
        model = numpy.load(file_name + '_model.npz')
        self.document_profiles = model['svd_u']
        self.word_profiles = model['svd_v']

    def save_document_profile_batch(self, document_ids):
        if self.profile_db is None:
            return False

        document_profile_batch = []
        for document_id in document_ids:
            document_profile = {
                'profile_id': document_id,
                'features': self.document_profiles[document_id, :].tolist(),
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
                'features': self.word_profiles[:, word_id].tolist(),
                'snapshot_time': datetime.datetime.utcnow()
            }

            word_profile_batch.append(word_profile)

        success = self.profile_db.save_or_udate_all('word_profiles', word_profile_batch)
        return success

    def load_document_profile_batch(self, document_ids):
        document_profiles = self.profile_db.find_all('document_profiles', document_ids)

        for document_profile in document_profiles:
            document_id = document_profile['profile_id']
            features = numpy.asarray(document_profile['features'])
            self.document_profiles[document_id, :] = features

    def load_word_profiles_batch(self, word_ids):
        word_profiles= self.profile_db.find_all('word_profiles', word_ids)

        for word_profile in word_profiles:
            word_id = word_profile['profile_id']
            features = numpy.asarray(word_profile['features'])
            self.word_profiles[:, word_id] = features