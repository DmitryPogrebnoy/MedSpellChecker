from re import search
from typing import final, Final, Generator, List

from nltk import download
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from pymorphy2.analyzer import Parse
from sacremoses import MosesTokenizer

from medspellchecker.tool.word import Word


@final
class PreProcessor:
    _stopwords_download_name: Final[str] = "stopwords"

    def __init__(self):
        self._version = 1
        self._tokenizer: Final[MosesTokenizer] = MosesTokenizer(lang="ru")
        self._lemmatizer: Final[MorphAnalyzer] = MorphAnalyzer()
        # Download nltk stopwords dict
        download(PreProcessor._stopwords_download_name, quiet=True)
        self._stopwords: Final[List[str]] = stopwords.words('russian')

    def is_valid_token(self, token: str) -> bool:
        """Checks if the correction token is valid.

        The token must not contain punctuation marks or other symbols,
        otherwise we cannot fix it correctly. And also all the letters together in the token are not capitalized -
        this is to exclude the correction of various abbreviations that are written with capital letters.
        """
        return (not search("[^а-яА-Я]", token)) and (not token.isupper()) and (
            not token in self._stopwords)

    def _is_proper_name(self, token: str, token_id: int, tokens: List[str]) -> bool:
        if token_id == 0:
            return False

        previous_token_id: int = token_id - 1
        previous_token: str = tokens[previous_token_id]

        return (not previous_token.endswith(".")) and token[0].isupper()

    def tokenize(self, string: str) -> List[str]:
        return self._tokenizer.tokenize(string)

    def lemmatize(self, string: str) -> str:
        return self._lemmatizer.parse(string)[0].normal_form

    def generate_words_from_tokens(self, tokens: List[str]) -> Generator[Word, None, None]:
        for id, token in enumerate(tokens):
            is_valid = self.is_valid_token(token)
            is_proper_name = self._is_proper_name(token, id, tokens)
            if is_valid and not is_proper_name:
                # if the token is valid for correction, extract a lemma from it
                parse: Parse = self._lemmatizer.parse(token)[0]
                if parse.is_known:
                    yield Word(id, token, True, parse.normal_form, parse.tag)
                else:
                    yield Word(id, token, True)
            else:
                # else just set corrected_value as original token
                yield Word(id, token, False, corrected_value=token)
