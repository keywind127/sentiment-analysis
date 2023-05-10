from pymongo import MongoClient
import pandas, ast, sys, re, os 
from typing import *

def csvs_in_folder(folder_name : str, sort : Optional[ bool ] = True) -> List[ str ]:
    files_in_folder = [
        os.path.join(folder_name, filename) for filename in os.listdir(folder_name) 
            if ((os.path.splitext(filename)[1].lower() == ".csv") and (os.path.dirname(filename) == ""))
    ]
    return ((sorted(files_in_folder)) if (sort) else (files_in_folder))

def even_filename_label(filename : str) -> bool:
    return (int(re.findall("[0-9]+$", os.path.splitext(os.path.basename(filename))[0])[0]) % 2 == 0)

def insert_dataframe_to_database(dataframe : pandas.DataFrame, verbose : Optional[ bool ] = True, file_percentage : Optional[ float ] = 100.0) -> None:
    global collection, DATAFRAME_COLUMN_NAME
    num_rows = len(dataframe)
    for row_idx, row_data in dataframe.iterrows():
        if (verbose):
            sys.stdout.write("\r[ Inserting ] [ {0:.1f}% ] [ {1:.1f}% ]".format(file_percentage, row_idx / num_rows * 100) + "  ")
            sys.stdout.flush()
        collection.insert_one(document = {
            "content" : ast.literal_eval(str(row_data[DATAFRAME_COLUMN_NAME]))
        })
    if (verbose):
        sys.stdout.write("\r[ Inserting ] [ {0:.1f}% ] [ 100.0% ]".format(file_percentage) + "  ")
        sys.stdout.flush()

def insert_dataframes_to_database(dataframe_files : List[ str ], verbose : Optional[ bool ] = True) -> None:
    num_dataframes = len(dataframe_files)
    for file_label, filename in enumerate(dataframe_files, start = 1):
        dataframe = pandas.read_csv(filename, encoding = "utf-8")
        insert_dataframe_to_database(dataframe, verbose, file_label / num_dataframes * 100)
    if (verbose):
        print("\n\n< Insertion Complete >")

if (__name__ == "__main__"):

    # >> CONSTANTS 

    INSERTION_NEU = 0

    INSERTION_POS = 1

    INSERTION_NEG = 2
    
    CURRENT_DIRECTORY = os.path.dirname(__file__)

    # << CONSTANTS

    # >> PARAMETERS 

    DATABASE_URI = "mongodb://localhost:27017"

    DATABASE_NAME = "sentiment-analysis"

    COLLECTION_NEU = "neutral"

    COLLECTION_POS = "positive"

    COLLECTION_NEG = "negative"

    DATAFRAME_FOLDER_NEU = os.path.join(CURRENT_DIRECTORY, "news_tokenized")

    DATAFRAME_FOLDER_SEN = os.path.join(CURRENT_DIRECTORY, "reviews_tokenized")

    DATAFRAME_COLUMN_NAME = "content"

    INSERTION_MODE = INSERTION_NEG

    # << PARAMETERS

    client = MongoClient(DATABASE_URI)

    database = client[DATABASE_NAME]

    if (INSERTION_MODE == INSERTION_NEU):

        collection = database[COLLECTION_NEU]

        csvs = csvs_in_folder(DATAFRAME_FOLDER_NEU)
    
    else:

        csvs = csvs_in_folder(DATAFRAME_FOLDER_SEN)

        if (INSERTION_MODE == INSERTION_POS):

            collection = database[COLLECTION_POS]

            csvs = list(filter(even_filename_label, csvs))

        if (INSERTION_MODE == INSERTION_NEG):

            collection = database[COLLECTION_NEG]

            csvs = list(filter(lambda x : not even_filename_label(x), csvs))

    insert_dataframes_to_database(csvs)

