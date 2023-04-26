import contractions, nltk, re
from typing import *

class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)

class DefaultVocabularySet:

    __corpus_word_sets = [
        None, None, None, None
    ]

    @classmethod 
    def initialize(class_) -> bool:
        if all(  (wordset is not None) for wordset in class_.__corpus_word_sets  ):
            return False 
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("names")
        nltk.download("words")
        class_.__corpus_word_sets = [
            set(nltk.corpus.stopwords.words()),
            set(nltk.corpus.wordnet.words()),
            set(nltk.corpus.names.words()),
            set(nltk.corpus.words.words())
        ]
        return True 

    @classproperty
    def stopwords(class_) -> Union[ Set[ str ], None ]:
        return class_.__corpus_word_sets[0]

    @classproperty
    def wordnet(class_) -> Union[ Set[ str ], None ]:
        return class_.__corpus_word_sets[1]

    @classproperty
    def names(class_) -> Union[ Set[ str ], None ]:
        return class_.__corpus_word_sets[2]

    @classproperty
    def words(class_) -> Union[ Set[ str ], None ]:
        return class_.__corpus_word_sets[3]
  
class LetterCaseOptimizer:

    def __init__(self, token_dictionary : Set[ str ]) -> None:
        self.token_dictionary = set( 
            token for token in token_dictionary 
        )

    @staticmethod 
    def enumerate_letter_cases(text : str) -> Iterator[ str ]:
        yield text;  yield text.upper();  yield text.capitalize();  yield text.lower()

    def optimize(self, tokens         : List[ str ], 
                       make_copy      : Optional[ bool ] = False, 
                       return_unknown : Optional[ bool ] = False,
                       filter_unknown : Optional[ bool ] = False) -> Union[ List[ str ], Tuple[ List[ str ], List[ str ] ] ]:
        if (make_copy):
            tokens = [  token for token in tokens  ]
        unknown_tokens = [];  known_tokens = []
        for idx, token in enumerate(tokens):
            for token in self.enumerate_letter_cases(token):
                if (token in self.token_dictionary):
                    known_tokens.append(token)
                    break 
            else:
                unknown_tokens.append(token)
            tokens[idx] = token
        if (filter_unknown):
            tokens = known_tokens
        return ((tokens, unknown_tokens) if (return_unknown) else (tokens))

class BasicStringTokenizer(nltk.tokenize.RegexpTokenizer):

    def __init__(self, pattern : Optional[ str ] = "[a-zA-Z0-9\']+", *args, **kwargs) -> None:
        assert ("'" in pattern)
        super(BasicStringTokenizer, self).__init__(pattern, *args, **kwargs)
 
    def expand_contractions_within_list(self, tokens : List[ str ], make_copy : Optional[ bool ] = False) -> List[ str ]:
        if (make_copy):
            tokens = [  token for token in tokens  ]
        for idx, token in reversed(list(enumerate(tokens))):
            if ("'" in token):
                tokens[ idx : idx + 1 ] = contractions.fix(token).split(" ")
        return tokens 

    def tokenize(self, text : str, *args, **kwargs) -> List[ str ]:
        return self.expand_contractions_within_list(
            super(BasicStringTokenizer, self).tokenize(text, *args, **kwargs)
        )

class TokensTaggingLemmatizer(nltk.stem.WordNetLemmatizer):
    
    nltk.download("averaged_perceptron_tagger")

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

    def __init__(self) -> None:
        super(TokensTaggingLemmatizer, self).__init__()

    def lemmatize(self, tokens : List[ str ]) -> List[ str ]:
        tokens = self.wordnet_pos_tagging(tokens)
        for idx, token_tag_pair in enumerate(tokens):
            tokens[idx] = super(TokensTaggingLemmatizer, self).lemmatize(*token_tag_pair)
        return tokens 

class StopWordRemover:

    def __init__(self, stopwords : Set[ str ]) -> None:
        self.stopwords = stopwords 

    def remove(self, tokens : List[ str ]) -> List[ str ]:
        return list(filter(lambda x : not self.stopwords.__contains__(x.lower()), tokens))

class CustomWordSet(set):

    def __init__(self, filename : str, *args, **kwargs) -> None:
        super(CustomWordSet, self).__init__(*args, **kwargs)
        self.update(set(filter("".__ne__, open(filename, "r", encoding = "utf-8").read().split("\n"))))
     
def remove_symbols(tokens : List[ str ]) -> List[ str ]:
    return re.findall("[a-zA-Z0-9]+", " ".join(tokens))