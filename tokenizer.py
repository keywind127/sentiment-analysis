from nlp_utils import load_word_list, TokenLemmatizer, StringCaseOptimizer
import pandas, nltk, sys, os
from typing import *

if (__name__ == "__main__"):

    # >> PARAMETERS

    custom_dictionaries = [  
        os.path.join(os.path.dirname(__file__), "custom_words_neut.txt"),
        os.path.join(os.path.dirname(__file__), "custom_words_sent.txt")
    ]

    training_data_folder = os.path.join(os.path.dirname(__file__), "reviews")

    custom_dict_filename = os.path.join(os.path.dirname(__file__), "custom_words_sent.txt")

    token_regular_expression = "[a-zA-Z0-9]+"
    
    dataframe_column_name = "content"

    min_tokens = 10

    # << PARAMETERS 

    training_data_output_folder = training_data_folder + "_tokenized"

    if not (os.path.exists(training_data_output_folder)):
        os.makedirs(training_data_output_folder)

    custom_dictionaries = set().union(
        *(set(load_word_list(filename)) 
            for filename in custom_dictionaries)
    )

    tokenizer = nltk.tokenize.RegexpTokenizer(token_regular_expression)

    case_optimizer = StringCaseOptimizer(custom_dictionaries)

    lemmatizer = TokenLemmatizer(custom_dictionaries)

    files_in_folder = sorted([  
        os.path.join(training_data_folder, filename) for filename in 
            os.listdir(training_data_folder) if (os.path.splitext(filename)[1].lower() == ".csv")
    ])

    for idx, filename in enumerate(files_in_folder):

        dataframe = pandas.read_csv(filename, encoding = "utf-8", error_bad_lines = False)

        num_rows = len(dataframe)

        rows_to_keep = []

        for row_idx, row_data in dataframe.iterrows():

            sys.stdout.write("\rTokenizing #{0}: {1:.1f}%".format(idx, row_idx / num_rows * 100))

            sys.stdout.flush()

            tokens = lemmatizer.lemmatize_tokens(
                case_optimizer.optimize_case(
                    tokenizer.tokenize(
                        str(row_data[dataframe_column_name])))
            )

            rows_to_keep.append(len(tokens) >= min_tokens)

            dataframe.loc[row_idx, dataframe_column_name] = str(tokens)

        sys.stdout.write(f"\rTokenizing #{idx}: 100%")

        print("")

        output_filename = os.path.join(training_data_output_folder, os.path.basename(filename))

        dataframe = dataframe[rows_to_keep]

        dataframe.to_csv(output_filename, encoding = "utf-8", index = False)

    print("\nTokenization Complete.")