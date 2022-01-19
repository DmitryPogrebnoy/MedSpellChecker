from dataclasses import dataclass
from re import search
from typing import final, Final, Generator

from mosestokenizer import MosesTokenizer
from nltk import download
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from pymorphy2.analyzer import Parse


@final
@dataclass
class Word:
    source_value: str
    valid: bool
    lemma_normal_form: str = None
    lemma_tag: str = None
    corrected_value: str = None


@final
class NewPerfectSpellchecker:
    # TODO: Add description
    _stopwords_download_name: Final[str] = "stopwords"

    def __init__(self):
        # Download nltk stopwords dict
        download(NewPerfectSpellchecker._stopwords_download_name)
        self._word_dictionary: Final[dict] = dict()
        self._tokenizer: Final[MosesTokenizer] = MosesTokenizer(lang="ru")
        self._lemmatizer: Final[MorphAnalyzer] = MorphAnalyzer()
        self._stopwords: Final[list[str]] = stopwords.words('russian')

    def _is_invalid_token(self, token: str) -> bool:
        return (not search("[^а-яА-Я]", token)) & (not token.isupper()) & (
            not token in self._stopwords)

    def _generate_words_from_tokens(self, tokens: list[str]) -> Generator[Word, None, None]:
        for token in tokens:
            is_valid = self._is_invalid_token(token)
            if is_valid:
                # if the token is valid for correction, extract a lemma from it
                parse: Parse = self._lemmatizer.parse(token)[0]
                yield Word(token, is_valid, parse.normal_form, parse.tag)
            else:
                # else just set corrected_value as original token
                yield Word(token, is_valid, corrected_value=token)

    # TODO: Implement generating candidates
    def _generate_fixing_candidates(self, word: Word) -> list[str]:
        # Some work
        candidates_list = list(word.source_value)
        return candidates_list

    # TODO: Implement ranking candidates
    def _find_most_suitable_candidate(self, candidates: list[str]) -> str:
        # Some work
        return candidates[0]

    # TODO: Do I need to save the formatting after fixing?
    def fix_text(self, text: str) -> str:
        # Remove newlines as the MosesTokenizer fails on newlines.
        # So if we decide to keep original text formatting then it should be reworked.
        text_without_newline: str = text.replace("\n", " ")
        # Tokenize text
        tokens: list[str] = self._tokenizer(text_without_newline)
        # Build internal representation of words
        words: list[Word] = [word for word in self._generate_words_from_tokens(tokens)]
        valid_words: list[Word] = [word for word in words if word.valid]
        invalid_words: list[Word] = [word for word in words if not word.valid]

        for valid_word in valid_words:
            # Generate list of candidates for fix
            candidates_list = self._generate_fixing_candidates(valid_word)
            # Pick most suitable candidate as fixed word
            most_suitable_candidate = self._find_most_suitable_candidate(candidates_list)
            valid_word.corrected_value = most_suitable_candidate

        corrected_text = ' '.join(map(lambda word: word.corrected_value, words))
        return corrected_text
