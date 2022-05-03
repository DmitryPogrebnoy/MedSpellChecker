from abc import abstractmethod, ABC
from enum import IntEnum
from typing import List

from candidate_word import CandidateWord
from ru_roberta_candidate_ranker import RuRobertaCandidateRanker
from word import Word


class CandidateRankerType(IntEnum):
    """Supported candidate ranking algorithms."""

    RU_ROBERTA_LARGE_CANDIDATE_RANKER = 0


class CandidateRanker:
    """Candidate word ranker.

        Args:
            rank_method: rank algorithm to use.

        Attributes:
            _algorithm (:class:`CandidateRankerType`): The rank algorithm to use.
            _candidate_ranker (:class:`AbstractCandidateRanker`): An object to
                rank the candidate words. The concrete object will be chosen based
                on the value of :attr:`_algorithm`.

        Raises:
            ValueError: If `algorithm` specifies an invalid rank algorithm.
    """

    def __init__(self, rank_method: CandidateRankerType = CandidateRankerType.RU_ROBERTA_LARGE_CANDIDATE_RANKER):
        self._algorithm = rank_method
        if rank_method == CandidateRankerType.RU_ROBERTA_LARGE_CANDIDATE_RANKER:
            self._candidate_ranker = RuRobertaCandidateRanker()
        else:
            raise ValueError("Unknown candidate word rank type!")

    def rank_candidates(self, current_word: Word,
                        all_correct_words: List[Word],
                        candidates: List[CandidateWord]) -> List[CandidateWord]:
        return self._candidate_ranker.rank_candidates(current_word, all_correct_words, candidates)


class AbstractCandidateRanker(ABC):

    @abstractmethod
    def rank_candidates(self, current_word: Word,
                        all_correct_words: List[Word],
                        candidates: List[CandidateWord]) -> List[CandidateWord]:
        """Returns ranked candidate words for correction

        Args:


        Returns:
            Ranked candidates. First element is the most suitable for fixing words
        """
