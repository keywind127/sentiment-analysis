from nlp_utils import remove_symbols, DefaultVocabularySet, BasicStringTokenizer, LetterCaseOptimizer
import pandas, sys, os 
from typing import * 

class TokensOccurrenceAnalyzer:

    @staticmethod 
    def analyze_token_occurrences(tokens : List[ str ], result_dictionary : Dict[ str, int ] = None) -> Dict[ str, int ]:
        if (result_dictionary is None):
            result_dictionary = dict()
        for token in tokens:
            result_dictionary[token] = ((result_dictionary[token] + 1) if (token in result_dictionary) else (1))
        return result_dictionary

    @staticmethod 
    def save_occurrences(filename : str, token_occurrences : Dict[ str, int ], sort : Optional[ bool ] = True, *args, **kwargs) -> None:
        token_occurrences = list(token_occurrences.items())
        if (sort):
            token_occurrences.sort(key = lambda x : x[1], reverse = True)
        keys, values = zip(*token_occurrences)
        pandas.DataFrame({  "token" : keys, "occurrences" : values  }).to_csv(filename, *args, **kwargs)

    def __init__(self, pattern : str) -> None:
        DVS = DefaultVocabularySet
        DVS.initialize()
        self.tokenizer = BasicStringTokenizer(pattern)
        self.optimizer = LetterCaseOptimizer(DVS.stopwords.union(DVS.wordnet, DVS.words, DVS.names))
        self.dictionary = dict()
        self.dataframe_counter = 1

    def analyze(self, dataframe : pandas.DataFrame, column_name : str, verbose : Optional[ bool ] = True) -> None:
        num_rows = len(dataframe)
        for row_idx, row_data in dataframe.iterrows():
            if (verbose):
                sys.stdout.write("\rAnalyzing #{0}: {1:.1f}%".format(self.dataframe_counter, row_idx / num_rows * 100))
                sys.stdout.flush()
            self.analyze_token_occurrences(self.optimizer.optimize(remove_symbols(self.tokenizer.tokenize(str(row_data[column_name]))), return_unknown = True)[1], self.dictionary)
        if (verbose):
            sys.stdout.write("\rAnalyzing #{0}: 100.0%".format(self.dataframe_counter))
            print("")
        self.dataframe_counter += 1
        
    def analyze_from_folder(self, folder_name     : str, 
                                  column_name     : str, 
                                  verbose         : Optional[ bool ] = True,
                                  encoding        : Optional[ str  ] = "utf-8",
                                  error_bad_lines : Optional[ bool ] = False   ) -> None:
        if (verbose):
            print("Analyzing Folder: {0}\n".format(folder_name))
        for idx, dataframe_filename in enumerate( os.path.join(folder_name, filename) for filename in os.listdir(folder_name) if (filename[-4:].lower() == ".csv") ):
            self.analyze(pandas.read_csv(dataframe_filename, encoding = encoding, error_bad_lines = error_bad_lines), column_name, verbose)
        if (verbose):
            print("\nAnalysis Complete.")

    def filter(self, min_threshold : int) -> None:
        for token, occurrence in list(self.dictionary.items()):
            if (occurrence < min_threshold):
                del self.dictionary[token]

    def save(self, filename :           str, 
                   sort     : Optional[ bool ] = True, 
                   encoding : Optional[ str  ] = "utf-8", 
                   index    : Optional[ bool ] = False,   *args, **kwargs) -> None:
        
        self.save_occurrences(filename, self.dictionary, sort, encoding = encoding, index = index, *args, **kwargs)

    @staticmethod 
    def convert_csv_to_txt(dst_filename : str, src_filename : str, encoding : Optional[ str ] = "utf-8", error_bad_lines : Optional[ bool ] = False, *args, **kwargs) -> None:
        dataframe = pandas.read_csv(src_filename, encoding = encoding, error_bad_lines = error_bad_lines)
        with open(dst_filename, "w", encoding = encoding, *args, **kwargs) as wf:
            for row_idx, row_data in dataframe.iterrows():
                wf.write(f"{row_data['token']}\n")

if (__name__ == "__main__"):


    # >> PARAMETERS 

    training_data_folder = os.path.join(os.path.dirname(__file__), "news") # reviews / news

    custom_dict_folder   = os.path.join(os.path.dirname(__file__), "dictionary")

    if not (os.path.exists(custom_dict_folder)):
        os.makedirs(custom_dict_folder)

    custom_dict_filename     = os.path.join(custom_dict_folder, "custom_words_neut.txt") # custom_words_sent.txt / custom_words_neut.txt

    token_regular_expression = "[a-zA-Z0-9\']+"
    
    dataframe_column_name    = "content"

    thresh_occurrence        = 30

    # << PARAMETERS 


    occurrence_dataframe_filename = os.path.splitext(custom_dict_filename)[0] + ".csv" 

    tokens_occurrences_analyzer   = TokensOccurrenceAnalyzer(token_regular_expression)

    tokens_occurrences_analyzer.analyze_from_folder(training_data_folder, dataframe_column_name)

    tokens_occurrences_analyzer.filter(thresh_occurrence)

    tokens_occurrences_analyzer.save(occurrence_dataframe_filename) 

    tokens_occurrences_analyzer.convert_csv_to_txt(custom_dict_filename, occurrence_dataframe_filename)

    print(f"Saved New Tokens: \"{occurrence_dataframe_filename}\" & \"{custom_dict_filename}\"")