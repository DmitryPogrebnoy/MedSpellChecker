import logging
from abc import abstractmethod, ABC
from typing import List, Optional, Tuple

from candidate_word import CandidateWord
from word import Word


class AbstractCandidateRanker(ABC):
    @abstractmethod
    def prepare_text_for_prediction(self, current_word: Word,
                                    all_correct_words: List[Word]) -> str:
        """Returns text prepared for prediction

            Args:
                current_word: The words that need correction
                all_correct_words: All correct words in texts

            Returns:
                Text prepared for prediction
        """

    @abstractmethod
    def predict_score(self, text_for_prediction: str,
                      candidate_value: str) -> Optional[float]:
        """Returns score of candidate word

            Args:
                text_for_prediction: Text for computing score
                candidate_value: Value for computing score

            Returns:
                Score of candidate word or None if it is under treshold
        """

    @abstractmethod
    def rank_candidates(self, current_word: Word,
                        all_correct_words: List[Word],
                        candidates: List[CandidateWord]) -> List[CandidateWord]:
        """Returns ranked candidate words for correction

        Args:
            current_word: The words that need correction
            all_correct_words: All correct words in texts
            candidates: Candidates for right word

        Returns:
            Ranked candidates. First element is the most suitable for fixing words
        """

    @abstractmethod
    def pick_most_suitable_candidate(self, current_word: Word,
                                     all_correct_words: List[Word],
                                     candidates: List[CandidateWord]) -> Optional[Tuple[CandidateWord, float]]:
        """Returns most suitable candidate for fixing incorrect words

            Args:
                current_word: The words that need correction
                all_correct_words: All correct words in texts
                candidates: Candidates for right word

            Returns:
                Most suitable candidate and it's score
        """


logger = logging.getLogger(__name__)


def _pick_correct_word_form(word: Word) -> str:
    if word.lemma_normal_form:
        return word.lemma_normal_form
    else:
        return word.original_value