from typing import *
import nltk

class StringCaseOptimizer:

    _CORPUS_WORDNET = None 

    @classmethod 
    def initialize_wordnet(class_, custom_words : Set[ str ] = None) -> None:
        if (class_._CORPUS_WORDNET is None):
            nltk.download("stopwords")
            nltk.download("wordnet")
            nltk.download("names")
            class_._CORPUS_WORDNET = (
                set(nltk.corpus.wordnet.words()).union(
                    set(nltk.corpus.names.words())).union(
                        set(nltk.corpus.stopwords.words()))
            )
        if (custom_words is not None):
            class_._CORPUS_WORDNET.update(custom_words)
            
    @staticmethod 
    def enumerate_case(string : str) -> Iterator[ str ]:
        yield string;  yield string.lower();  yield string.upper();  yield string.capitalize()

    def __init__(self, custom_words : Set[ str ] = None) -> None:
        assert ((custom_words is None) or (isinstance(custom_words, set)))
        self.initialize_wordnet(custom_words)

    def optimize_case(self, tokens : List[ str ], in_place : Optional[ bool ] = False) -> List[ str ]:
        if not (in_place):
            tokens = [  token for token in tokens  ]
        for idx, token in enumerate(tokens):
            for token_case in self.enumerate_case(token):
                if (token_case in self._CORPUS_WORDNET):
                    tokens[idx] = token_case 
                    break 
        return tokens 

    def filter_unknown_words(self, tokens : List[ str ]) -> List[ str ]:
        return [
            token for token in tokens if not any(
                token_case in self._CORPUS_WORDNET 
                    for token_case in self.enumerate_case(token))
        ]

def save_word_list(filename : str, tokens : List[ str ]) -> None:
    with open(filename, mode = "w", encoding = "utf-8") as wf:
        for token in tokens:
            wf.write(f"{token}\n")

def load_word_list(filename : str) -> List[ str ]:
    with open(filename, mode = "r", encoding = "utf-8") as rf:
        return list(filter("".__ne__, rf.read().split("\n")))