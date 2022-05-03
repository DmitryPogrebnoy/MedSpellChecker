from typing import final, List

from transformers import AutoModelForMaskedLM, AutoTokenizer

from candidate_ranker import AbstractCandidateRanker
from candidate_word import CandidateWord
from word import Word


@final
class RuRobertaCandidateRanker(AbstractCandidateRanker):
    _pretrained_model_checkpoint = "DmitryPogrebnoy/MedRuRobertaLarge"

    def __init__(self):
        self._version = 1
        self._tokenizer = AutoTokenizer.from_pretrained(RuRobertaCandidateRanker._pretrained_model_checkpoint)
        self._model = AutoModelForMaskedLM.from_pretrained(RuRobertaCandidateRanker._pretrained_model_checkpoint)

    def rank_candidates(self, current_word: Word,
                        all_correct_words: List[Word],
                        candidates: List[CandidateWord]) -> List[CandidateWord]:
        """Returns ranked candidate words for correction

        Args:


        Returns:
            Ranked candidates. First element is the most suitable for fixing words
        """
