from typing import *
import nltk

_CORPUS_WORDNET = None 

def initialize_wordnet(custom_words : Set[ str ] = None) -> None:
    global _CORPUS_WORDNET
    if (_CORPUS_WORDNET is None):
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("names")
        _CORPUS_WORDNET = (
            set(nltk.corpus.wordnet.words()).union(
                set(nltk.corpus.names.words())).union(
                    set(nltk.corpus.stopwords.words()))
        )
    if (custom_words is not None):
        _CORPUS_WORDNET.update(custom_words)

class StringCaseOptimizer:
            
    @staticmethod 
    def enumerate_case(string : str) -> Iterator[ str ]:
        yield string;  yield string.lower();  yield string.upper();  yield string.capitalize()

    def __init__(self, custom_words : Set[ str ] = None) -> None:
        assert ((custom_words is None) or (isinstance(custom_words, set)))
        initialize_wordnet(custom_words)

    def optimize_case(self, tokens : List[ str ], in_place : Optional[ bool ] = False) -> List[ str ]:
        if not (in_place):
            tokens = [  token for token in tokens  ]
        for idx, token in enumerate(tokens):
            for token_case in self.enumerate_case(token):
                if (token_case in _CORPUS_WORDNET):
                    tokens[idx] = token_case 
                    break 
        return tokens 

    def filter_unknown_words(self, tokens : List[ str ]) -> List[ str ]:
        return [
            token for token in tokens if not any(
                token_case in _CORPUS_WORDNET 
                    for token_case in self.enumerate_case(token))
        ]

def save_word_list(filename : str, tokens : List[ str ]) -> None:
    with open(filename, mode = "w", encoding = "utf-8") as wf:
        for token in tokens:
            wf.write(f"{token}\n")

def load_word_list(filename : str) -> List[ str ]:
    with open(filename, mode = "r", encoding = "utf-8") as rf:
        return list(filter("".__ne__, rf.read().split("\n")))

class TokenLemmatizer:

    @staticmethod 
    def wordnet_pos_tagging(tokens : List[ str ]) -> List[ Tuple[ str, str ] ]:
        tokens = nltk.pos_tag(tokens)
        for idx, (token, pos_tag) in enumerate(tokens):
            if (pos_tag.startswith("J")):
                tokens[idx] = (token, nltk.corpus.wordnet.ADJ)
                continue 
            if (pos_tag.startswith("V")):
                tokens[idx] = (token, nltk.corpus.wordnet.VERB)
                continue 
            if (pos_tag.startswith("R")):
                tokens[idx] = (token, nltk.corpus.wordnet.ADV)
                continue 
            tokens[idx] = (token, nltk.corpus.wordnet.NOUN)
        return tokens 

    def __init__(self, custom_words : Set[ str ] = None) -> None:
        assert ((custom_words is None) or (isinstance(custom_words, set)))
        initialize_wordnet(custom_words)
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_tokens(self, tokens : List[ str ]) -> List[ str ]:
        tokens = self.wordnet_pos_tagging(tokens)
        for idx, (token, pos_tag) in enumerate(tokens):
            tokens[idx] = self.lemmatizer.lemmatize(token, pos_tag)
        return tokens 

class StopWordFilter:

    # We observed stopwords removal may negatively impact the performance of sentiment analysis.

    _STOPWORDS = None 

    @classmethod 
    def initialize_stopwords(class_) -> None:
        if (class_._STOPWORDS is None):
            nltk.download("stopwords")
            class_._STOPWORDS = set(nltk.corpus.stopwords.words())

    def __init__(self) -> None:
        self.initialize_stopwords()

    def filter_stopwords(self, tokens : List[ str ]) -> List[ str ]:
        return [
            token for token in tokens 
                if token not in self._STOPWORDS
        ]

if (__name__ == "__main__"):

    tokenizer = nltk.tokenize.RegexpTokenizer("[a-zA-Z0-9]+")

    sentence = "It is well known to the world that computer science students may face unemployment after graduation due to the rise of artificial intelligence. "

    tokens = tokenizer.tokenize(sentence)

    case_optimizer = StringCaseOptimizer()

    tokens = case_optimizer.optimize_case(tokens)

    lemmatizer = TokenLemmatizer()

    tokens = lemmatizer.lemmatize_tokens(tokens)

    print(tokens)

    stopword_filter = StopWordFilter()

    tokens = stopword_filter.filter_stopwords(tokens)

    print(tokens)