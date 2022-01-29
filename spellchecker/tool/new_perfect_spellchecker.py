import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from re import search
from typing import final, Final, Generator, Set, List, Dict, Optional, Union, IO

from mosestokenizer import MosesTokenizer
from nltk import download
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from pymorphy2.analyzer import Parse

from editdistance import EditDistanceAlgo, EditDistance

logger = logging.getLogger(__name__)


@final
@dataclass
class Word:
    """Input word representation.

        Args:
            source_value: The original word.
            should_correct: Should this word be corrected?
            lemma_normal_form: The normal form of the word
            lemma_tag: Tag line of the original word form
            corrected_value: Final corrected word
    """
    original_value: str
    should_correct: bool
    lemma_normal_form: Optional[str] = None
    lemma_tag: Optional[str] = None
    corrected_value: Optional[str] = None


@final
@dataclass
class WordWindow:
    """Internal representation of window for candidate ranking.

       Args:
           center_word: word for correction
           left_words: words on the left side (maybe empty for first word in text)
           right_words: words on the right side (maybe empty for last word in text)
   """
    center_word: Word
    left_words: List[Word]
    right_words: List[Word]


@final
@dataclass
class CandidateWord:
    """Spelling suggestion.

        Args:
            value: The suggested word.
            distance: Edit distance from search word.
    """

    value: str
    distance: int
    # TODO: Add more features for ranking if needed


@final
class NewPerfectSpellchecker:
    # TODO: Add description
    _stopwords_download_name: Final[str] = "stopwords"

    # TODO: Need ability to init spellchecker from saved state
    def __init__(self,
                 words_list: Optional[Union[Path, str, IO[str], List[str]]] = None,
                 encoding: Optional[str] = None,
                 edit_distance_algo: EditDistanceAlgo = EditDistanceAlgo.DAMERAU_OSA_FAST,
                 max_dictionary_edit_distance: int = 2,
                 saved_state: Optional[Union[Path, str, IO[str]]] = None):
        self._version = 1

        if saved_state is not None:
            self._load_sate(saved_state)
        else:
            self._edit_distance_comparer: EditDistance = EditDistance(edit_distance_algo)
            self._max_dictionary_edit_distance: Final[int] = max_dictionary_edit_distance

            self._max_dictionary_word_length: int = 0
            self._words_dictionary: Set[str] = set()
            self._deletes_word_dictionary: Dict[str, List[str]] = defaultdict(list)
            # Fill _words_dictionary and build _deletes_word_dictionary
            if words_list is None:
                raise ValueError(f"word_list and saved_state cannot be None at the same time. Pass one of these "
                                 f"arguments!")
            self._create_dictionary_from_words(words_list, encoding)

        self._tokenizer: Final[MosesTokenizer] = MosesTokenizer(lang="ru")
        self._lemmatizer: Final[MorphAnalyzer] = MorphAnalyzer()
        # Download nltk stopwords dict
        download(NewPerfectSpellchecker._stopwords_download_name)
        self._stopwords: Final[List[str]] = stopwords.words('russian')

    def _load_sate(self, saved_state: Union[Path, str, IO[str]]):
        if isinstance(saved_state, (Path, str)):
            state_path = Path(saved_state)
            if not state_path.exists():
                raise ValueError(f"State not found at {state_path}.")
            with open(state_path, "r") as infile:
                data = json.load(infile)
        else:
            data = json.load(saved_state)

        if data["_version"] != self._version:
            raise ValueError(f"Incompatible version of loaded state! Loaded version is {data['_version']}, but "
                             f"spellchecker version is {self._version}")

        self._edit_distance_comparer = EditDistance(data["_edit_distance_algo"])
        # noinspection PyFinal
        self._max_dictionary_edit_distance = data["_max_dictionary_edit_distance"]
        self._max_dictionary_word_length = data["_max_dictionary_word_length"]
        self._words_dictionary = set(data["_words_dictionary"])
        self._deletes_word_dictionary = data["_deletes_word_dictionary"]

    def save_state(self, path: Union[Path, str]):
        data = {
            "_version": self._version,
            "_edit_distance_algo": self._edit_distance_comparer.algorithm,
            "_max_dictionary_edit_distance": self._max_dictionary_edit_distance,
            "_max_dictionary_word_length": self._max_dictionary_word_length,
            "_words_dictionary": list(self._words_dictionary),
            "_deletes_word_dictionary": self._deletes_word_dictionary
        }
        with open(Path(path), "w") as file:
            file.write(json.dumps(data))

    def _create_dictionary_from_words(
            self,
            words_list: Union[Path, str, IO[str], List[str]],
            encoding: Optional[str] = None) -> bool:
        """Create dictionary words from a file containing words. One word per line.

        Args:
            words_list: The path+filename of the file or afile object of the
                word-list or list of strings.
            encoding: Text encoding of the corpus file.

        Returns:
            ``True`` if file loaded, or ``False`` if file not found.
        """
        if isinstance(words_list, (Path, str)):
            corpus = Path(words_list)
            if not corpus.exists():
                logger.error(f"Corpus not found at {corpus}.")
                return False
            with open(corpus, "r", encoding=encoding) as infile:
                for line in infile.read().splitlines():
                    self._create_dictionary_entry(line)
        else:
            for line in words_list:
                self._create_dictionary_entry(line)
        return True

    def _create_dictionary_entry(self, key: str) -> None:
        """Creates/updates an entry in the dictionary.

        For every word there are deletes with an edit distance of
        1..max_edit_distance created and added to the dictionary. Every delete
        entry has a suggestions list, which points to the original term(s) it was
        created from. The dictionary may be dynamically updated (word frequency
        and new words) at any time by calling create_dictionary_entry.

        Args:
            key: The word to add to dictionary.

        Returns:
            None
        """
        # TODO: Should we lemmatise words in order to glue together several different forms
        #  of the same word and thereby reduce the size of the dictionary?
        if key not in self._words_dictionary:
            self._words_dictionary.add(key)

            # edits/suggestions are created only once, no matter how often word
            # occurs. edits/suggestions are created as soon as the word occurs in the
            # corpus, even if the same term existed before in the dictionary as an
            # edit from another word
            if len(key) > self._max_dictionary_word_length:
                self._max_dictionary_word_length = len(key)

            # create deletes
            edits = self._generate_delete_words(key)
            logger.info(edits)
            for delete in edits:
                self._deletes_word_dictionary[delete].append(key)
        return

    def _is_invalid_token(self, token: str) -> bool:
        """Checks if the correction token is valid.

        The token must not contain punctuation marks or other symbols,
        otherwise we cannot fix it correctly. And also all the letters together in the token are not capitalized -
        this is to exclude the correction of various abbreviations that are written with capital letters.
        """
        return (not search("[^а-яА-Я]", token)) & (not token.isupper()) & (
            not token in self._stopwords)

    def _generate_words_from_tokens(self, tokens: List[str]) -> Generator[Word, None, None]:
        for token in tokens:
            is_valid = self._is_invalid_token(token)
            if is_valid:
                # if the token is valid for correction, extract a lemma from it
                parse: Parse = self._lemmatizer.parse(token)[0]
                yield Word(token, is_valid, parse.normal_form, parse.tag)
            else:
                # else just set corrected_value as original token
                yield Word(token, is_valid, corrected_value=token)

    def _recursive_generate_delete_words(
            self,
            word: str,
            edit_distance: int,
            delete_words: Set[str],
            current_distance: int = 0,
    ) -> Set[str]:
        """SymDel algorithm in the recursion style uses only deletes for simulate
        the four expensive operations (transposes, replaces, deletions, addition).
        """
        edit_distance += 1
        if not word:
            return delete_words
        for i in range(current_distance, len(word)):
            delete = word[:i] + word[i + 1:]
            if delete in delete_words:
                continue
            delete_words.add(delete)
            # start recursion, if maximum edit distance not yet reached
            if edit_distance < self._max_dictionary_edit_distance:
                self._recursive_generate_delete_words(delete, edit_distance, delete_words, i)
        return delete_words

    def _generate_delete_words(self, word: str) -> Set[str]:
        hash_set = {word}
        if len(word) <= self._max_dictionary_edit_distance:
            hash_set.add("")
        return self._recursive_generate_delete_words(word, 0, hash_set)

    # TODO: Test this implementation
    def _generate_fixing_candidates(
            self,
            word: Word,
            max_edit_distance: Optional[int] = None,
            include_unknown: bool = True) -> List[CandidateWord]:

        if max_edit_distance is None:
            max_edit_distance = self._max_dictionary_edit_distance
        if max_edit_distance > self._max_dictionary_edit_distance:
            raise ValueError("Passed edit distance too big!")
        suggestions: List[CandidateWord] = list()

        original_word = word.original_value
        original_word_length = len(original_word)

        def early_return():
            if include_unknown and not suggestions:
                suggestions.append(CandidateWord(original_word, 0))
            return suggestions

        # early return - word is too big to possibly match any words
        if original_word_length - max_edit_distance > self._max_dictionary_word_length:
            return early_return()

        # early_return - quick check for exact match
        if original_word in self._words_dictionary:
            # return exact match
            return early_return()

        # early return - if we only want to check if word in dictionary
        if max_edit_distance == 0:
            return early_return()

        considered_deletes: Set[str] = set()
        considered_candidates: Set[str] = set()
        # we considered the original word already in the 'original_word in self._words_dictionary' above
        considered_candidates.add(original_word)

        candidate_pointer: int = 0
        candidates: List[str] = [original_word]

        while candidate_pointer < len(candidates):
            candidate = candidates[candidate_pointer]
            candidate_pointer += 1
            candidate_len = len(candidate)
            len_diff = abs(original_word_length - candidate_len)

            # early return: if candidate distance is already higher than
            # suggestion distance, then there are no better suggestions to be expected
            if len_diff > max_edit_distance:
                break  # "peephole" optimization, http://bugs.python.org/issue2506

            if candidate in self._deletes_word_dictionary:
                candidates_from_delete_dictionary = self._deletes_word_dictionary[candidate]
                for suggestion in candidates_from_delete_dictionary:
                    if suggestion == original_word:
                        continue
                    suggestion_len = len(suggestion)
                    # original word length and suggestion lengths diff > allowed/current best
                    # distance
                    if (
                            abs(suggestion_len - original_word_length) > max_edit_distance
                            # suggestion must be for a different delete string, in
                            # same bin only because of hash collision
                            or suggestion_len < candidate_len
                            # if suggestion len = delete len, then it either equals
                            # delete or is in same bin only because of hash collision
                            or (suggestion_len == candidate_len and suggestion != candidate)):
                        continue  # "peephole" optimization, http://bugs.python.org/issue2506

                    # True Damerau-Levenshtein Edit Distance: adjust distance,
                    # if both distances>0. We allow simultaneous edits (deletes)
                    # of max_edit_distance on both the dictionary and the
                    # original term. For replaces and adjacent transposes the
                    # resulting edit distance stays <= max_edit_distance. For
                    # inserts and deletes the resulting edit distance might
                    # exceed max_edit_distance. To prevent suggestions of a
                    # higher edit distance, we need to calculate the resulting
                    # edit distance, if there are simultaneous edits on both
                    # sides. Example: (bank==bnak and bank==bink, but bank!=kanb
                    # and bank!=xban and bank!=baxn for max_edit_distance=1).
                    # Two deletes on each side of a pair makes them all equal,
                    # but the first two pairs have edit distance=1, the others
                    # edit distance=2.
                    distance = 0
                    min_distance = 0
                    if candidate_len == 0:
                        # suggestions which have no common chars with phrase
                        # (phrase_len<=max_edit_distance && suggestion_len<=max_edit_distance)
                        distance = max(original_word_length, suggestion_len)
                        if distance > max_edit_distance or suggestion in considered_candidates:
                            continue
                    elif suggestion_len == 1:
                        # This should always be phrase_len - 1? Since
                        # suggestions are generated from deletes of the input
                        # phrase
                        distance = (
                            original_word_length
                            if original_word.index(suggestion[0]) < 0
                            else original_word_length - 1
                        )
                        # `suggestion` only gets added to
                        # `considered_candidates` when `suggestion_len>1`.
                        # Given the max_dictionary_edit_distance and
                        # prefix_length restrictions, `distance`` should never
                        # be >max_edit_distance_2
                        if (
                                distance > max_edit_distance
                                or suggestion in considered_candidates
                        ):
                            continue
                    # number of edits in prefix ==maxeditdistance AND no
                    # identical suffix, then editdistance>max_edit_distance and
                    # no need for Levenshtein calculation
                    # (phraseLen >= prefixLength) &&
                    # (suggestionLen >= prefixLength)
                    else:
                        # delete_in_suggestion_prefix is somewhat expensive
                        if suggestion in considered_candidates:
                            continue
                        considered_candidates.add(suggestion)
                        distance = self._edit_distance_comparer.compare(
                            original_word, suggestion, max_edit_distance
                        )
                        if distance < 0:
                            continue
                    # do not process higher distances than those already found,
                    # if verbosity<ALL (note: max_edit_distance_2 will always
                    # equal max_edit_distance when Verbosity.ALL)
                    if distance <= max_edit_distance:  # pragma: no branch
                        item = CandidateWord(suggestion, distance)
                        suggestions.append(item)

            # add edits: derive edits (deletes) from candidate (phrase) and add
            # them to candidates list. this is a recursive process until the
            # maximum edit distance has been reached
            if len_diff < max_edit_distance:
                for i in range(candidate_len):
                    delete = candidate[:i] + candidate[i + 1:]
                    if delete not in considered_deletes:
                        considered_deletes.add(delete)
                        candidates.append(delete)

        early_return()
        return suggestions

    # TODO: Implement ranking candidates
    def _find_most_suitable_candidate(self, candidates: List[CandidateWord]) -> CandidateWord:
        # Some work
        return candidates[0]

    # TODO: Do I need to save the formatting after fixing?
    def fix_text(self, text: str) -> str:
        # Remove newlines as the MosesTokenizer fails on newlines.
        # So if we decide to keep original text formatting then it should be reworked.
        text_without_newline: str = text.replace("\n", " ")
        # Tokenize text
        tokens: List[str] = self._tokenizer(text_without_newline)
        # Build internal representation of words
        words: List[Word] = [word for word in self._generate_words_from_tokens(tokens)]
        valid_words: List[Word] = [word for word in words if word.should_correct]
        invalid_words: List[Word] = [word for word in words if not word.should_correct]

        # TODO: Need to implement the window words for ranking candidates
        for valid_word in valid_words:
            # Generate list of candidates for fix
            candidates_list: List[CandidateWord] = self._generate_fixing_candidates(valid_word)
            # Pick most suitable candidate as fixed word
            most_suitable_candidate: CandidateWord = self._find_most_suitable_candidate(candidates_list)
            # TODO: Need to restore original case and word form from tags
            valid_word.corrected_value = most_suitable_candidate.value

        corrected_text = ' '.join(map(lambda word: word.corrected_value, words))
        return corrected_text
