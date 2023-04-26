from gensim.models import Word2Vec
import pandas, pickle, ast, sys, os
from typing import *

class DF2PKL:

    @staticmethod 
    def load_dataframe_tokens(training_data   : List[ List[ str ] ], 
                              src_filename    : str, 
                              column_name     : str, 
                              encoding        : Optional[ str  ] = "utf-8", 
                              error_bad_lines : Optional[ bool ] = False, 
                              verbose         : Optional[ bool ] = True,
                              verbose_label   : Optional[ str  ] = ""       ) -> None:
        
        dataframe = pandas.read_csv(src_filename, encoding = encoding, error_bad_lines = error_bad_lines)
        num_rows = len(dataframe)
        for row_idx, row_data in dataframe.iterrows():
            if (verbose):
                sys.stdout.write("\rProcessing{0}: {1:.1f}%".format(verbose_label, row_idx / num_rows * 100))
                sys.stdout.flush()
            training_data.append(ast.literal_eval(str(row_data[column_name])))
        if (verbose):
            sys.stdout.write("\rProcessing{0}: 100.0%".format(verbose_label))
            print("")

    @classmethod 
    def load_dataframe_tokens_from_folders(class_, folder_list     : List[ str ], 
                                                   column_name     : str, 
                                                   encoding        : Optional[ str  ] = "utf-8", 
                                                   error_bad_lines : Optional[ bool ] = False, 
                                                   verbose         : Optional[ bool ] = True     ) -> List[ List[ str ] ]:
        
        def dataframe_files_within_folder(folder_name : str, file_extension : Optional[ str ] = ".csv") -> List[ str ]:
            return [
                os.path.join(folder_name, filename) for filename in os.listdir(folder_name) 
                    if (os.path.splitext(filename)[1].lower() == file_extension)
            ]
        list_of_list_of_strings = []
        for folder_idx, folder_name in enumerate(folder_list):
            if (verbose):
                print("Processing Folder #{0}: \"{1}\"".format(folder_idx + 1, folder_name))
            for dataframe_idx, dataframe_name in enumerate(dataframe_files_within_folder(folder_name, file_extension = ".csv")):
                class_.load_dataframe_tokens(list_of_list_of_strings, dataframe_name, column_name, encoding, error_bad_lines, verbose, verbose_label = f" #{dataframe_idx + 1}")
            if (verbose):
                print("")
        if (verbose):
            print("\nProcessing Complete.")
        return list_of_list_of_strings

    @staticmethod 
    def dump(filename : str, training_data : List[ List[ str ] ], *args, **kwargs) -> None:
        with open(filename, "wb") as wf:
            pickle.dump(training_data, wf, protocol = pickle.HIGHEST_PROTOCOL, *args, **kwargs)

    @staticmethod 
    def load(filename : str, *args, **kwargs) -> List[ List[ str ] ]:
        with open(filename, "rb") as rf:
            return pickle.load(rf, *args, **kwargs)

if (__name__ == "__main__"):


    # >> CONSTANTS

    CASE_PREPROCESS = 0

    CASE_TRAINING   = 1

    # << CONSTANTS

    
    # >> PARAMETERS

    dataframe_folder_list = [
        os.path.join(os.path.dirname(__file__), "reviews_tokenized"),
        os.path.join(os.path.dirname(__file__), "news_tokenized")
    ]

    training_data_pickle_filename = os.path.join(os.path.dirname(__file__), "dictionary/training_data.pickle")

    word2vec_model_name = os.path.join(os.path.dirname(__file__), "dictionary/w2v_embedder.model")

    dataframe_column_name = "content"

    vector_size = 200

    num_workers = 4

    window_size = 4

    num_epochs  = 30

    CASE = CASE_TRAINING

    # << PARAMETERS


    if (CASE == CASE_PREPROCESS):

        training_data = DF2PKL.load_dataframe_tokens_from_folders(dataframe_folder_list, dataframe_column_name)

        DF2PKL.dump(training_data_pickle_filename, training_data)

        print("\nSaved Training Data: \"{}\"".format(training_data_pickle_filename))

    if (CASE == CASE_TRAINING):

        print("> Loading Training Data...")

        training_data = DF2PKL.load(training_data_pickle_filename)

        print("> Loading Success.\n")

        print("> Training Embedder Model...")

        word_embedder = Word2Vec(training_data, vector_size = vector_size, window = window_size, workers = num_workers, epochs = num_epochs)

        print("> Training Success.\n")

        print("> Saving Embedder Model...")

        word_embedder.save(word2vec_model_name)

        print("> Saving Success.")