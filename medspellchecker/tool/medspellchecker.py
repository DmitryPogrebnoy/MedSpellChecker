import logging
import os
from pathlib import Path
from typing import Dict, final, IO, List, Optional, Set, Tuple, Union

import numpy as np
from pkg_resources import resource_filename
from pymorphy2.tagset import OpencorporaTag

from medspellchecker.tool.abstract_candidate_ranker import AbstractCandidateRanker
from medspellchecker.tool.candidate_generator import CandidateGenerator
from medspellchecker.tool.candidate_word import CandidateWord
from medspellchecker.tool.edit_distance import EditDistanceAlgo
from medspellchecker.tool.mistake_type import MistakeType
from medspellchecker.tool.pre_post_processor import PreProcessor
from medspellchecker.tool.word import Word

logger = logging.getLogger(__name__)


@final
class MedSpellchecker:
    def __init__(self, candidate_ranker: AbstractCandidateRanker,
                 words_list: Optional[Union[Path, str, IO[str], List[str]]] = resource_filename(
                     __name__, "data/processed_lemmatized_all_dict.txt"),
                 encoding: Optional[str] = None,
                 edit_distance_algo: EditDistanceAlgo = EditDistanceAlgo.DAMERAU_OSA_FAST,
                 max_dictionary_edit_distance: int = 1,
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
            first_part: Tuple[str, OpencorporaTag] = self._pre_processor.lemmatize_and_tag(word.original_value[:i])
            second_part: Tuple[str, OpencorporaTag] = self._pre_processor.lemmatize_and_tag(word.original_value[i:])
            if first_part[0] in self.words and second_part[0] in self.words:
                candidate_word = f"{first_part[0]} {second_part[0]}"
                first_word: Word = Word(word.position, first_part[0], True, corrected_value=first_part[0])
                second_word: Word = Word(word.position, second_part[0], True, corrected_value=second_part[0])
                first_score = self._candidate_ranker.predict_score(
                    valid_words_before,
                    [second_word] + valid_words_after,
                    first_part[0]
                )
                second_score = self._candidate_ranker.predict_score(
                    valid_words_before + [first_word],
                    valid_words_after,
                    second_part[0]
                )
                if first_score and second_score:
                    candidate_words.append(
                        (CandidateWord(candidate_word, 1),
                         (first_score + second_score) / 2,
                         first_part[1],
                         second_part[1])
                    )
        if candidate_words:
            candidate_words.sort(key=lambda x: x[1])
            best_candidate_word_tuple = candidate_words[0]
            best_candidate_word_split = best_candidate_word_tuple[0].value.split()
            best_candidate_word_part_first: str = best_candidate_word_split[0]
            best_candidate_word_part_second: str = best_candidate_word_split[1]

            restored_first_part: str = self._pre_processor.str_restore_original_form(
                best_candidate_word_part_first, best_candidate_word_tuple[2])
            restored_second_part: str = self._pre_processor.str_restore_original_form(
                best_candidate_word_part_second, best_candidate_word_tuple[3])

            return CandidateWord(f"{restored_first_part} {restored_second_part}", 1), best_candidate_word_tuple[1]
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

            if (not current_word.should_correct) or (current_word.original_value in self.words) or \
                    (current_word.lemma_normal_form in self.words):
                current_word.corrected_value = current_word.original_value
                current_word_idx += 1
                continue

            valid_words_before: List[Word] = [word for word in words[:current_word_idx] if word.should_correct]
            valid_words_after: List[Word] = [word for word in words[current_word_idx + 1:] if word.should_correct]

            mistake_type_to_candidate: Dict[MistakeType, Tuple[CandidateWord, float]] = {}

            # Compute score of original word
            original_word_score: Optional[float] = self._candidate_ranker.predict_score(
                valid_words_before, valid_words_after, current_word.get_lemma_normal_or_original_form())

            if original_word_score is None:
                original_word_score = 1.0

            # If we shouldn't handle spacing mistakes then that's it
            if self._handle_compound_words:
                # Else we need to process handle cases
                # Consider missing space mistake
                if len(current_word.get_lemma_normal_or_original_form()) > 3:
                    top_split_candidate_pair: Optional[Tuple[CandidateWord, float]] = \
                        self._pick_top_missing_space_candidate(current_word, valid_words_before, valid_words_after)
                    if top_split_candidate_pair:
                        mistake_type_to_candidate[MistakeType.MISSING_SPACE_MISTAKE] = top_split_candidate_pair

                # Consider extra space mistake
                compound_word: Optional[Word] = self._create_compound_word(current_word_idx, current_word, words)
                top_extra_space_candidate_pair: Optional[Tuple[CandidateWord, float]] = \
                    self._get_top_extra_space_candidate(compound_word, valid_words_before, valid_words_after)
                if top_extra_space_candidate_pair:
                    mistake_type_to_candidate[MistakeType.EXTRA_SPACE_MISTAKE] = top_extra_space_candidate_pair
            else:
                compound_word = None

            # Consider non-spacing types of mistakes
            # Generate list of candidates for fix
            if len(current_word.get_lemma_normal_or_original_form()) > 3:
                single_candidates_list: List[CandidateWord] = self._candidate_generator.generate_fixing_candidates(
                    current_word, include_unknown=False)

                if self._handle_compound_words and current_word_idx + 1 < len(words):
                    top_single_word_candidate_pair: Optional[Tuple[CandidateWord, float]] = \
                        self._candidate_ranker.pick_top_candidate_with_next_word(
                            current_word,
                            words[current_word_idx + 1].get_lemma_normal_or_original_form(),
                            valid_words_before,
                            valid_words_after,
                            single_candidates_list
                        )
                else:
                    top_single_word_candidate_pair: Optional[Tuple[CandidateWord, float]] = \
                        self._candidate_ranker.pick_top_candidate(current_word, valid_words_before,
                                                                  valid_words_after, single_candidates_list)
                if top_single_word_candidate_pair:
                    mistake_type_to_candidate[MistakeType.SINGLE_WORD_MISTAKE] = top_single_word_candidate_pair

            # Early skip
            if not mistake_type_to_candidate:
                current_word.corrected_value = current_word.get_lemma_normal_or_original_form()
                current_word_idx += 1
                continue

            # Now we have collected all possible candidates. So let's pick most suitable one
            top_candidate: Tuple[MistakeType, Tuple[CandidateWord, float]] = \
                self._find_top_candidate(mistake_type_to_candidate)
            self._fix_word_by_top_candidate(
                top_candidate,
                original_word_score,
                current_word,
                current_word_idx,
                words,
                compound_word
            )
            current_word_idx += 1

        corrected_text = self._build_result_text(words)
        return corrected_text

    def _create_compound_word(self, current_word_idx: int, current_word: Word, words: List[Word]) -> Optional[Word]:
        if current_word_idx + 1 < len(words):
            next_word: Word = words[current_word_idx + 1]
            new_word_str: str = current_word.original_value + next_word.original_value
            new_word: Word = next(self._pre_processor.generate_words_from_tokens([new_word_str]))
            return new_word
        return None

    def _get_top_extra_space_candidate(self, compound_word: Optional[Word],
                                       valid_words_before: List[Word],
                                       valid_words_after: List[Word]) -> Optional[Tuple[CandidateWord, float]]:
        if compound_word and compound_word.should_correct and compound_word.get_lemma_normal_or_original_form() in self.words:
            extra_space_candidate: CandidateWord = CandidateWord(compound_word.get_lemma_normal_or_original_form(), 1)
            top_extra_space_candidate_pair: Optional[Tuple[CandidateWord, float]] = \
                self._candidate_ranker.pick_top_candidate(
                    compound_word, valid_words_before, valid_words_after[1:], [extra_space_candidate])
            return top_extra_space_candidate_pair
        return None

    def _fix_word_by_top_candidate(self,
                                   top_candidate: Tuple[MistakeType, Tuple[CandidateWord, float]],
                                   original_word_score: float,
                                   current_word: Word,
                                   current_word_idx: int,
                                   words: List[Word],
                                   compound_word: Optional[Word]):
        top_candidate_type: MistakeType = top_candidate[0]
        top_candidate_word: CandidateWord = top_candidate[1][0]
        top_candidate_score: float = top_candidate[1][1]

        if original_word_score < top_candidate_score:
            if top_candidate_type == MistakeType.SINGLE_WORD_MISTAKE:
                current_word.corrected_value = top_candidate_word.value
                return
            elif self._handle_compound_words:
                if top_candidate_type == MistakeType.MISSING_SPACE_MISTAKE:
                    # Here we didn't create two separate words due to the performance reasons
                    current_word.corrected_value = top_candidate_word.value
                    return
                elif top_candidate_type == MistakeType.EXTRA_SPACE_MISTAKE:
                    del words[current_word_idx + 1]
                    compound_word.corrected_value = top_candidate_word.value
                    words[current_word_idx] = compound_word
                    return

        current_word.corrected_value = current_word.get_lemma_normal_or_original_form()

    def _find_top_candidate(self, mistake_type_to_candidate: Dict[MistakeType, Tuple[CandidateWord, float]]) \
            -> Tuple[MistakeType, Tuple[CandidateWord, float]]:
        mistake_type_candidate_list: List[Tuple[MistakeType, Tuple[CandidateWord, float]]] = \
            list(mistake_type_to_candidate.items())

        candidate_scores: List[float] = list(map(lambda x: x[1][1], mistake_type_candidate_list))
        top_candidate_idx: int = np.argmax(candidate_scores)
        return mistake_type_candidate_list[top_candidate_idx]

    def _build_result_text(self, words: List[Word]) -> str:
        result_list = [self._pre_processor.word_restore_original_form(word) for word in words]
        return ' '.join(result_list)
