from pymongo import MongoClient, UpdateOne 
from typing import *
import numpy, sys

class SentimentAnalysisDatabase(MongoClient):

    SENTIMENT_COLLECTION_NAMES = [
        "neutral", "positive", "negative"
    ]

    DATA_TAG_NAMES = [
        "training", "testing "
    ]

    SENTIMENT_NEU = 0

    SENTIMENT_POS = 1

    SENTIMENT_NEG = 2

    DATA_TRAINING = 0

    DATA_TESTING  = 1

    def __init__(self, database_uri : str) -> None:
        super(SentimentAnalysisDatabase, self).__init__(database_uri)
        self.sentiment_analysis_database = self["sentiment-analysis"]
        self.database_collections = [
            self.sentiment_analysis_database[self.SENTIMENT_COLLECTION_NAMES[collection_idx]] for collection_idx 
                in range(self.SENTIMENT_NEU, self.SENTIMENT_NEG + 1)
        ]

    def __remove_data_tags(self, verbose : Optional[ bool ] = True) -> None:
        if (verbose):
            sys.stdout.write("\r[ Removing Tags ]")
            sys.stdout.flush()
        for collection_idx in range(self.SENTIMENT_NEU, self.SENTIMENT_NEG + 1):
            current_collection = self.database_collections[collection_idx]
            current_collection.update_many(
                filter = {  "tag"    : { "$exists" : True } }, 
                update = {  "$unset" : { "tag"     : None } }
            )
        if (verbose):
            sys.stdout.write("\r< Removal Success >")
            sys.stdout.flush()

    def train_test_split(self, test_ratio : float, verbose : Optional[ bool ] = True) -> None:
        assert (0 <= test_ratio <= 1)
        self.__remove_data_tags(verbose = verbose)
        for collection_idx in range(self.SENTIMENT_NEU, self.SENTIMENT_NEG + 1):
            current_collection = self.database_collections[collection_idx]
            num_documents = current_collection.count_documents({})
            random_tag = numpy.random.choice([ self.DATA_TESTING, self.DATA_TRAINING ], size = num_documents, p = [ test_ratio, 1 - test_ratio ])
            bulk_updates = []
            for document_idx, document in enumerate(current_collection.find({})):
                if (verbose):
                    sys.stdout.write("\r[ Splitting ] [ {0:.1f}% ] [ {1:.1f}% ]".format((collection_idx + 1) / (self.SENTIMENT_NEG + 1) * 100, (document_idx + 1) / num_documents * 100) + "  ")
                    sys.stdout.flush()
                bulk_updates.append(UpdateOne(
                    filter = {  "_id"  : document["_id"]                            },
                    update = {  "$set" : { "tag" : int(random_tag[document_idx]) }  }
                ))
            current_collection.bulk_write(bulk_updates)
            current_collection.create_index("tag")
        if (verbose):
            print("\n\n< Splitting Complete >")

    def generate_data(self, sentiment : int, tag : int) -> Iterator[ List[ str ] ]:
        for document in self.database_collections[sentiment].find(filter = {  "tag" : tag  }):
            yield document["content"] 

    def num_documents(self, sentiment : int, tag : int) -> int:
        return self.database_collections[sentiment].count_documents({  "tag" : tag  })

if (__name__ == "__main__"):

    # connecting to MongoDB instance 
    database = SentimentAnalysisDatabase("mongodb://localhost:27017")

    # spliting data into training and testing set 
    database.train_test_split(0.10)

    print("")

    # observing data quantity after splitting 
    for collection_idx in range(database.SENTIMENT_NEU, database.SENTIMENT_NEG + 1):
        print("Collection: \"{}\"".format(database.SENTIMENT_COLLECTION_NAMES[collection_idx]))
        for data_tag in range(database.DATA_TRAINING, database.DATA_TESTING + 1):
            print("    {0} # {1:,}".format(database.DATA_TAG_NAMES[data_tag], database.num_documents(collection_idx, data_tag)))
        print("")

    # testing data generator 
    for document in database.generate_data(database.SENTIMENT_POS, database.DATA_TRAINING):
        print(document)
        break 