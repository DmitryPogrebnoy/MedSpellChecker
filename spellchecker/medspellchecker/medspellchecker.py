import logging
import os
from pathlib import Path
from typing import final, List, Optional, Union, IO, Set, Tuple, Dict

import numpy as np

from abstract_candidate_ranker import AbstractCandidateRanker
from candidate_generator import CandidateGenerator
from candidate_word import CandidateWord
from edit_distance import EditDistanceAlgo
from mistake_type import MistakeType
from pre_post_processor import PreProcessor
from word import Word

logger = logging.getLogger(__name__)


@final
class MedSpellchecker:
    def __init__(self, candidate_ranker: AbstractCandidateRanker,
                 words_list: Optional[Union[Path, str, IO[str], List[str]]] = None,
                 encoding: Optional[str] = None,
                 edit_distance_algo: EditDistanceAlgo = EditDistanceAlgo.DAMERAU_OSA_FAST,
                 max_dictionary_edit_distance: int = 2,
                 handle_compound_words: bool = False,
                 saved_state_folder: Optional[Union[Path, str]] = None):
        self._version = 1
        self._handle_compound_words = handle_compound_words
        if isinstance(words_list, (Path, str)):
            corpus = Path(words_list)
            if not corpus.exists():
                logger.error(f"Corpus not found at {corpus}.")
            with open(corpus, "r", encoding=encoding) as infile:
                self.words: Set[str] = set(infile.read().splitlines())
        else:
            self.words: Set[str] = set(words_list)

        self._pre_processor: PreProcessor = PreProcessor()

        if saved_state_folder is not None:
            self._candidate_generator: CandidateGenerator = CandidateGenerator(saved_state_folder=saved_state_folder)
        else:
            self._candidate_generator: CandidateGenerator = CandidateGenerator(
                words_list, encoding, edit_distance_algo, max_dictionary_edit_distance
            )

        self._candidate_ranker: AbstractCandidateRanker = candidate_ranker

    def save_state(self, path: Union[Path, str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._candidate_generator.save_state(path)

    def _pick_top_missing_space_candidate(self, word: Word,
                                          valid_words_before: List[Word],
                                          valid_words_after: List[Word]) -> Optional[Tuple[CandidateWord, float]]:
        candidate_words = []
        for i in range(len(word.original_value)):
            first_word = self._pre_processor.lemmatize(word.original_value[:i])
            second_part = self._pre_processor.lemmatize(word.original_value[i:])
            if first_word in self.words and second_part in self.words:
                candidate_word = f"{first_word} {second_part}"
                score = self._candidate_ranker.predict_score(word, valid_words_before, valid_words_after,
                                                             candidate_word)
                if score:
                    candidate_words.append((CandidateWord(candidate_word, 1), score))
        if candidate_words:
            candidate_words.sort(key=lambda x: x[1])
            return candidate_words[0]
        else:
            return None

    # TODO: Do I need to save the formatting after fixing? -- no for now,
    #  usually formatting is not necessary for ml tasks
    def fix_text(self, text: str) -> str:
        # Remove newlines as the MosesTokenizer fails on newlines.
        # So if we decide to keep original text formatting then it should be reworked.
        text_without_newline: str = text.replace("\n", " ")
        # Tokenize text
        tokens: List[str] = self._pre_processor.tokenize(text_without_newline)
        # Build internal representation of words
        words: List[Word] = [word for word in self._pre_processor.generate_words_from_tokens(tokens)]

        current_word_idx = 0
        while current_word_idx < len(words):
            current_word = words[current_word_idx]

            if not current_word.should_correct or current_word.original_value in self.words:
                current_word.corrected_value = current_word.original_value
                current_word_idx += 1
                continue

            valid_words_before: List[Word] = [word for word in words[:current_word_idx] if word.should_correct]
            valid_words_after: List[Word] = [word for word in words[current_word_idx + 1:] if word.should_correct]

            # Consider non-spacing types of mistakes
            # Generate list of candidates for fix
            single_candidates_list: List[CandidateWord] = self._candidate_generator.generate_fixing_candidates(
                current_word)
            top_single_word_candidate_pair: Optional[Tuple[CandidateWord, float]] = \
                self._candidate_ranker.pick_top_candidate(current_word, valid_words_before,
                                                          valid_words_after, single_candidates_list)

            # If we shouldn't handle spacing mistakes then that's it
            if not self._handle_compound_words:
                if top_single_word_candidate_pair:
                    current_word.corrected_value = top_single_word_candidate_pair[0].value
                else:
                    current_word.corrected_value = current_word.original_value
                current_word_idx += 1
                continue

            # Else we need to process handle cases, but first of all create candidate dict
            candidate_by_types: Dict[MistakeType, Tuple[CandidateWord, float]] = {}
            if top_single_word_candidate_pair:
                candidate_by_types[MistakeType.SINGLE_WORD_MISTAKE] = top_single_word_candidate_pair

            # Consider missing space mistake
            top_split_candidate_pair: Optional[Tuple[CandidateWord, float]] = \
                self._pick_top_missing_space_candidate(current_word, valid_words_before, valid_words_after)
            if top_split_candidate_pair:
                candidate_by_types[MistakeType.MISSING_SPACE_MISTAKE] = top_split_candidate_pair

            # Consider extra space mistake
            new_word: Optional[Word] = None
            if current_word_idx + 1 < len(words):
                next_word: Word = words[current_word_idx + 1]
                new_word_str: str = current_word.original_value + next_word.original_value
                new_word: Word = next(self._pre_processor.generate_words_from_tokens([new_word_str]))
                if new_word.should_correct and new_word.original_value in self.words:
                    extra_space_candidate: CandidateWord = CandidateWord(new_word_str, 1)
                    top_extra_space_candidate_pair: Optional[Tuple[CandidateWord, float]] = \
                        self._candidate_ranker.pick_top_candidate(
                            new_word, valid_words_before, valid_words_after[1:], [extra_space_candidate])
                    if top_extra_space_candidate_pair:
                        candidate_by_types[MistakeType.EXTRA_SPACE_MISTAKE] = top_extra_space_candidate_pair

            # Now we have collected all possible candidates. So let's pick most suitable one
            candidate_by_types_list: List[Tuple[MistakeType, Tuple[CandidateWord, float]]] = \
                list(candidate_by_types.items())

            if not candidate_by_types_list:
                current_word.corrected_value = current_word.original_value
                current_word_idx += 1
                continue

            min_distance_candidate = min(candidate_by_types_list, key=lambda x: x[1][0].distance)
            filtered_candidate_by_types_list: List[Tuple[MistakeType, Tuple[CandidateWord, float]]] = \
                list(filter(lambda x: x[1][0].distance == min_distance_candidate[1][0].distance,
                            candidate_by_types_list))
            candidate_scores: List[float] = list(map(lambda x: x[1][1], filtered_candidate_by_types_list))
            top_candidate_idx: int = np.argmax(candidate_scores)
            top_candidate: Tuple[MistakeType, Tuple[CandidateWord, float]] = \
                filtered_candidate_by_types_list[top_candidate_idx]

            top_candidate_type: MistakeType = top_candidate[0]
            top_candidate_word: CandidateWord = top_candidate[1][0]

            if top_candidate_type == MistakeType.SINGLE_WORD_MISTAKE:
                current_word.corrected_value = top_candidate_word.value
                current_word_idx += 1
                continue
            elif top_candidate_type == MistakeType.MISSING_SPACE_MISTAKE:
                # Here we didn't create two separate words due to the performance reasons
                current_word.corrected_value = top_candidate_word.value
                current_word_idx += 1
                continue
            elif top_candidate_type == MistakeType.EXTRA_SPACE_MISTAKE:
                del words[current_word_idx + 1]
                new_word.corrected_value = top_candidate_word.value
                words[current_word_idx] = new_word
                current_word_idx += 1
                continue

            current_word.corrected_value = current_word.original_value
            current_word_idx += 1

        corrected_text = ' '.join(map(lambda word: word.corrected_value, words))
        return corrected_text
