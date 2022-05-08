import logging
import os
from pathlib import Path
from typing import final, List, Optional, Union, IO, Set

from candidate_generator import CandidateGenerator
from candidate_ranker import CandidateRanker, CandidateRankerType
from candidate_word import CandidateWord
from edit_distance import EditDistanceAlgo
from pre_post_processor import PrePostProcessor
from word import Word

logger = logging.getLogger(__name__)


@final
class MedSpellchecker:
    def __init__(self,
                 words_list: Optional[Union[Path, str, IO[str], List[str]]] = None,
                 encoding: Optional[str] = None,
                 edit_distance_algo: EditDistanceAlgo = EditDistanceAlgo.DAMERAU_OSA_FAST,
                 max_dictionary_edit_distance: int = 2,
                 candidate_ranker_type: CandidateRankerType = CandidateRankerType.RU_ROBERTA_LARGE_CANDIDATE_RANKER,
                 saved_state_folder: Optional[Union[Path, str]] = None):
        self._version = 1
        if isinstance(words_list, (Path, str)):
            corpus = Path(words_list)
            if not corpus.exists():
                logger.error(f"Corpus not found at {corpus}.")
            with open(corpus, "r", encoding=encoding) as infile:
                self.words: Set[str] = set(infile.read().splitlines())
        else:
            self.words: Set[str] = set(words_list)

        self._pre_pos_processor: PrePostProcessor = PrePostProcessor()

        if saved_state_folder is not None:
            self._candidate_generator = CandidateGenerator(saved_state_folder=saved_state_folder)
        else:
            self._candidate_generator = CandidateGenerator(words_list,
                                                           encoding,
                                                           edit_distance_algo,
                                                           max_dictionary_edit_distance)
            self._candidate_ranker = CandidateRanker(candidate_ranker_type)

    def save_state(self, path: Union[Path, str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._candidate_generator.save_state(path)

    # TODO: Do I need to save the formatting after fixing? -- no for now,
    #  usually formatting is not necessary for ml tasks
    def fix_text(self, text: str) -> str:
        # Remove newlines as the MosesTokenizer fails on newlines.
        # So if we decide to keep original text formatting then it should be reworked.
        text_without_newline: str = text.replace("\n", " ")
        # Tokenize text
        tokens: List[str] = self._pre_pos_processor.tokenize(text_without_newline)
        # Build internal representation of words
        words: List[Word] = [word for word in self._pre_pos_processor.generate_words_from_tokens(tokens)]
        valid_words: List[Word] = [word for word in words if word.should_correct]
        invalid_words: List[Word] = [word for word in words if not word.should_correct]

        for valid_word in valid_words:
            if valid_word.original_value in self.words:
                valid_word.corrected_value = valid_word.original_value
            else:
                # Generate list of candidates for fix
                candidates_list: List[CandidateWord] = self._candidate_generator.generate_fixing_candidates(valid_word)
                # Pick most suitable candidate as fixed word
                most_suitable_candidate: CandidateWord = self._candidate_ranker.pick_most_suitable_candidate(
                    valid_word, valid_words, candidates_list)
                # TODO: Need to restore original case and word form from tags -- no for now,
                #  lemmatization is common activity for preprocessing
                if most_suitable_candidate:
                    valid_word.corrected_value = most_suitable_candidate.value
                else:
                    valid_word.corrected_value = valid_word.original_value

        corrected_text = ' '.join(map(lambda word: word.corrected_value, words))
        return corrected_text
