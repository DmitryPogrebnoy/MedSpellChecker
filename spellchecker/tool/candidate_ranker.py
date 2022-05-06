import logging
from abc import abstractmethod, ABC
from enum import IntEnum
from typing import List, final, Optional, Tuple

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForMaskedLM

from candidate_word import CandidateWord
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

    def pick_most_suitable_candidate(self, current_word: Word,
                                     all_correct_words: List[Word],
                                     candidates: List[CandidateWord]) -> CandidateWord:
        return self._candidate_ranker.pick_most_suitable_candidate(current_word, all_correct_words, candidates)


class AbstractCandidateRanker(ABC):

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

    def pick_most_suitable_candidate(self, current_word: Word,
                                     all_correct_words: List[Word],
                                     candidates: List[CandidateWord]) -> CandidateWord:
        """Returns most suitable candidate for fixing incorrect words

            Args:
                current_word: The words that need correction
                all_correct_words: All correct words in texts
                candidates: Candidates for right word

            Returns:
                Most suitable candidate
        """


logger = logging.getLogger(__name__)


def _pick_correct_word_form(word: Word) -> str:
    if word.lemma_normal_form:
        return word.lemma_normal_form
    else:
        return word.original_value


@final
class RuRobertaCandidateRanker(AbstractCandidateRanker):
    _pretrained_model_checkpoint = "DmitryPogrebnoy/MedRuRobertaLarge"

    def __init__(self, use_gpu: bool = False):
        self._version = 1
        self._use_gpu = use_gpu

        if self._use_gpu:
            accelerator = Accelerator(fp16=True)
            self._tokenizer = accelerator.prepare(
                AutoTokenizer.from_pretrained(RuRobertaCandidateRanker._pretrained_model_checkpoint))
            self._model = accelerator.prepare(
                AutoModelForMaskedLM.from_pretrained(RuRobertaCandidateRanker._pretrained_model_checkpoint))
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(RuRobertaCandidateRanker._pretrained_model_checkpoint)
            self._model = AutoModelForMaskedLM.from_pretrained(RuRobertaCandidateRanker._pretrained_model_checkpoint)

    def _build_text_for_prediction(self, current_word: Word,
                                   all_correct_words: List[Word], ):
        return ' '.join(map(
            lambda x: _pick_correct_word_form(
                x) if not x.position == current_word.position else self._tokenizer.mask_token,
            all_correct_words))

    def _process_candidates(self, text_for_prediction: str,
                            candidate: CandidateWord) -> Tuple[CandidateWord, float]:
        inputs = self._tokenizer(text_for_prediction, return_tensors='pt')
        logger.debug(inputs)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["input_ids"]
        if self._use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        token_logits = self._model(input_ids, attention_mask).logits
        logger.debug(f"Token logits: {token_logits}")

        mask_token_index = torch.where(inputs["input_ids"] == self._tokenizer.mask_token_id)[1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        softmax_mask_token_logits = torch.softmax(mask_token_logits, dim=1)
        candidate_token_ids = self._tokenizer.encode(candidate.value, add_special_tokens=False)
        logger.debug(f"Candidate token ids: {candidate_token_ids}")

        if logger.getEffectiveLevel() == logging.DEBUG:
            top_20 = torch.topk(softmax_mask_token_logits, 20, dim=1)
            top_20_tokens = zip(top_20.indices[0].tolist(), top_20.values[0].tolist())
            logger.debug("----Top 20 most suitable tokens----")
            for token, score in top_20_tokens:
                logger.debug(f"{self._tokenizer.decode([token])}, score: {score}")

        ## Sum score of candidate word parts
        candidate_score = 0.0
        for token in candidate_token_ids:
            candidate_score = softmax_mask_token_logits[:, token]
        logger.debug(f"Candidate - {candidate}, score {candidate_score}")
        return candidate, candidate_score

    def rank_candidates(self, current_word: Word,
                        all_correct_words: List[Word],
                        candidates: List[CandidateWord]) -> List[CandidateWord]:
        # candidates must be ordered by edit distance

        if not candidates:
            return candidates

        text_for_prediction = self._build_text_for_prediction(current_word, all_correct_words)

        logger.debug(f"Text for prediction {text_for_prediction}")

        candidate_score_pairs = []

        for candidate in candidates:
            processed_candidate_score = self._process_candidates(text_for_prediction, candidate)
            candidate_score_pairs.append(processed_candidate_score)

        ranked_candidate_score_pairs = sorted(candidate_score_pairs, key=lambda x: x[1])
        ranked_candidates = map(lambda x: x[0], ranked_candidate_score_pairs)
        return list(ranked_candidates)

    def pick_most_suitable_candidate(self, current_word: Word,
                                     all_correct_words: List[Word],
                                     candidates: List[CandidateWord]) -> Optional[CandidateWord]:
        # candidates must be ordered by edit distance

        if not candidates:
            return None

        text_for_prediction = self._build_text_for_prediction(current_word, all_correct_words)

        logger.debug(f"Text for prediction {text_for_prediction}")

        candidate_score_pairs = []
        previous_edit_distance_candidate = candidates[0].distance

        for candidate in candidates:
            if candidate.distance > previous_edit_distance_candidate:
                break

            processed_candidate_score = self._process_candidates(text_for_prediction, candidate)
            candidate_score_pairs.append(processed_candidate_score)
            previous_edit_distance_candidate = candidate.distance

        ranked_candidate_score_pairs = sorted(candidate_score_pairs, key=lambda x: x[1])
        return ranked_candidate_score_pairs[0][0]
