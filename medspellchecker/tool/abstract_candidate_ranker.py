from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from medspellchecker.tool.candidate_word import CandidateWord
from medspellchecker.tool.word import Word


class AbstractCandidateRanker(ABC):
    @abstractmethod
    def predict_score(self, correct_words_before: List[Word], words_after: List[Word], candidate_value: str) -> \
    Optional[float]:
        """Returns score of candidate word

            Args:
                current_word: current word
                correct_words_before: corrected words before
                words_after: words after (not corrected yet)
                candidate_value: Value for computing score

            Returns:
                Score of candidate word or None if it is under treshold
        """

    @abstractmethod
    def rank_candidates(self, current_word: Word,
                        correct_words_before: List[Word],
                        words_after: List[Word],
                        candidates: List[CandidateWord]) -> List[CandidateWord]:
        """Returns ranked candidate words for correction

        Args:
            current_word: The words that need correction
            correct_words_before: correct words before current word
            words_after: not processed yet words after current word
            candidates: Candidates for right word

        Returns:
            Ranked candidates. First element is the most suitable for fixing words
        """

    @abstractmethod
    def pick_top_candidate(self, current_word: Word,
                           correct_words_before: List[Word],
                           words_after: List[Word],
                           candidates: List[CandidateWord]) -> Optional[Tuple[CandidateWord, float]]:
        """Returns most suitable candidate for fixing incorrect words

            Args:
                current_word: The words that need correction
                correct_words_before: correct words before current word
                words_after: not processed yet words after current word
                candidates: Candidates for right word

            Returns:
                Most suitable candidate and it's score
        """

    @abstractmethod
    def pick_top_candidate_with_next_word(self, current_word: Word,
                                          next_word: Word,
                                          correct_words_before: List[Word],
                                          words_after: List[Word],
                                          candidates: List[CandidateWord]) -> Optional[Tuple[CandidateWord, float]]:
        """Returns most suitable candidate (with next word) for fixing incorrect words

            Args:
                current_word: The words that need correction
                next_word: Next word
                correct_words_before: correct words before current word
                words_after: not processed yet words after current word
                candidates: Candidates for right word

            Returns:
                Most suitable candidate and it's score
        """
