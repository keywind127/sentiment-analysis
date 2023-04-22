from nlp_utils import StringCaseOptimizer, save_word_list
import pandas, nltk, sys, os 
from typing import * 

def analyze_word_occurrence(tokens : Union[ Set[ str ], List[ str ] ]) -> Dict[ str, int ]:
    tokens_occurrences = dict()
    for token in tokens:
        tokens_occurrences[token] = (
            (tokens_occurrences[token] + 1) 
                if (token in tokens_occurrences) else (1)
        )
    return tokens_occurrences 

def filter_rare_tokens(tokens : Dict[ str, int ], thresh_occurrence : Optional[ int ] = 10) -> Dict[ str, int ]:
    return {  
        token : occurrence for token, occurrence in tokens.items() 
            if (occurrence >= thresh_occurrence)  
    }

if (__name__ == "__main__"):

    # >> PARAMETERS 

    training_data_folder = os.path.join(os.path.dirname(__file__), "reviews")

    custom_dict_filename = os.path.join(os.path.dirname(__file__), "custom_words_sent.txt")

    token_regular_expression = "[a-zA-Z0-9]+"
    
    dataframe_column_name = "content"

    thresh_occurrence = 20

    # << PARAMETERS 

    files_in_folder = [  
        os.path.join(training_data_folder, filename) for filename in 
            os.listdir(training_data_folder) if (os.path.splitext(filename)[1].lower() == ".csv")
    ]

    tokenizer = nltk.tokenize.RegexpTokenizer(token_regular_expression)

    optimizer = StringCaseOptimizer()

    tokens = dict()

    for idx, filename in enumerate(files_in_folder):
        
        dataframe = pandas.read_csv(filename, error_bad_lines = False)

        num_rows = len(dataframe)

        for df_idx, df_row in dataframe.iterrows():

            sys.stdout.write("\rAnalyzing #{0}: {1:.1f}%".format(idx, df_idx / num_rows * 100))

            sys.stdout.flush()

            __tokens = analyze_word_occurrence(
                optimizer.filter_unknown_words(
                    tokenizer.tokenize(str(df_row[dataframe_column_name])))
            )

            for __token, __occ in __tokens.items():

                tokens[__token] = (
                    (tokens[__token] + __occ) if (__token in tokens) else (__occ)
                )

        sys.stdout.write(f"\rAnalyzing #{idx}: 100.0%")

        print("")

    save_word_list(custom_dict_filename, filter_rare_tokens(tokens, thresh_occurrence).keys())

    print(f"\nNew vocabulary list saved to: \"{custom_dict_filename}\"")