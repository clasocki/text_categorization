import datetime
import pymongo
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
import logging


class SnapshotDbAdapter(object):
    AVAILABLE_COLLECTION_NAMES = ['document_profiles', 'word_profiles']

    def __init__(self, db_name):
        self.client = MongoClient('localhost', 27017)
        self.db_name = db_name
        self.db = self.client[db_name]
        logging.basicConfig(filename='snapshot_db_adapter.log', level=logging.DEBUG)

    def initialize_collections(self):
        for collection_name in self.AVAILABLE_COLLECTION_NAMES:
            self.db[collection_name].create_index([('profile_id', pymongo.ASCENDING)], unique=True)

    def save_or_update_all(self, collection_name, items):
        collection = self.db[collection_name]
        bulk = collection.initialize_ordered_bulk_op()

        for item in items:
            bulk.find({'profile_id': item['profile_id']})\
                .upsert().update_one({'$set': item})

        try:
            bulk.execute()
        except BulkWriteError as bwe:
            logging.debug("Time: " + str(datetime.datetime.now()) + "Db: " + self.db_name +
                          " Collection: " + collection_name + " Detail: " + str(bwe.details))
            return False

    def find_all(self, collection_name, item_ids):
        collection = self.db[collection_name]
        profiles = collection.find({'profile_id': {'$in': item_ids}})

        return profiles
