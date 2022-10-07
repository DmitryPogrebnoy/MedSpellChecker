import logging
from abc import abstractmethod
from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForMaskedLM

from abstract_candidate_ranker import AbstractCandidateRanker
from candidate_word import CandidateWord
from gpu_utils import set_device
from word import Word


class AbstractBertCandidateRanker(AbstractCandidateRanker):

    def __init__(self, pretrained_model_checkpoint: str,
                 use_treshold: bool = True, treshold: float = 0.0000001,
                 use_gpu: bool = True):
        self._use_treshold: bool = use_treshold
        self._treshold: float = treshold
        self._use_gpu: bool = set_device() if use_gpu else False

        accelerator = Accelerator(cpu=not self._use_gpu)
        self._tokenizer = accelerator.prepare(
            AutoTokenizer.from_pretrained(pretrained_model_checkpoint))
        self._model = accelerator.prepare(
            AutoModelForMaskedLM.from_pretrained(pretrained_model_checkpoint))

    @property
    @abstractmethod
    def _version(self) -> int:
        """
        :return: version of ranker
        """

    def prepare_text_for_prediction(self, current_word: Word,
                                    all_correct_words: List[Word]) -> str:
        return ' '.join(map(
            lambda x: _pick_correct_word_form(
                x) if not x.position == current_word.position else self._tokenizer.mask_token,
            all_correct_words))

    def predict_score(self, text_for_prediction: str,
                      candidate_value: str) -> Optional[float]:
        inputs = self._tokenizer(text_for_prediction, return_tensors='pt')
        logger.debug(inputs)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if self._use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        token_logits = self._model(input_ids, attention_mask).logits
        logger.debug(f"Token logits: {token_logits}")

        mask_token_index = torch.where(inputs["input_ids"] == self._tokenizer.mask_token_id)[1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        softmax_mask_token_logits = torch.softmax(mask_token_logits, dim=1)
        candidate_token_ids = self._tokenizer.encode(candidate_value, add_special_tokens=False)
        logger.debug(f"Candidate token ids: {candidate_token_ids}")

        if logger.getEffectiveLevel() == logging.DEBUG:
            top_20 = torch.topk(softmax_mask_token_logits, 20, dim=1)
            top_20_tokens = zip(top_20.indices[0].tolist(), top_20.values[0].tolist())
            logger.debug("----Top 20 most suitable tokens----")
            for token, score in top_20_tokens:
                logger.debug(f"{self._tokenizer.decode([token])}, score: {score}")

        ## Mean score of candidate word parts
        candidate_score = 0.0
        for token in candidate_token_ids:
            candidate_score += softmax_mask_token_logits[:, token][0]
        candidate_score = candidate_score / len(candidate_token_ids)
        logger.debug(f"Candidate - {candidate_value}, score {candidate_score}")
        if self._use_treshold and candidate_score > self._treshold or not self._use_treshold:
            return candidate_score
        else:
            return None

    def _process_candidates(self, text_for_prediction: str,
                            candidate: CandidateWord) -> Optional[Tuple[CandidateWord, float]]:
        candidate_score = self.predict_score(text_for_prediction, candidate.value)
        if candidate_score:
            return candidate, candidate_score
        else:
            return None

    def rank_candidates(self, current_word: Word,
                        all_correct_words: List[Word],
                        candidates: List[CandidateWord]) -> List[CandidateWord]:
        if not candidates:
            return candidates

        # candidates must be ordered by edit distance
        candidates.sort(key=lambda x: x.distance)

        text_for_prediction = self.prepare_text_for_prediction(current_word, all_correct_words)

        logger.debug(f"Text for prediction {text_for_prediction}")

        candidate_score_pairs = []

        for candidate in candidates:
            processed_candidate_score = self._process_candidates(text_for_prediction, candidate)
            if processed_candidate_score:
                candidate_score_pairs.append(processed_candidate_score)

        ranked_candidate_score_pairs = sorted(candidate_score_pairs, key=lambda x: x[1])
        ranked_candidates = map(lambda x: x[0], ranked_candidate_score_pairs)
        return list(ranked_candidates)

    def pick_most_suitable_candidate(self, current_word: Word,
                                     all_correct_words: List[Word],
                                     candidates: List[CandidateWord]) -> Optional[Tuple[CandidateWord, float]]:
        if not candidates:
            return None

        # candidates must be ordered by edit distance
        candidates.sort(key=lambda x: x.distance)

        text_for_prediction = self.prepare_text_for_prediction(current_word, all_correct_words)

        logger.debug(f"Text for prediction {text_for_prediction}")

        candidate_score_pairs = []
        previous_edit_distance_candidate = candidates[0].distance

        for candidate in candidates:
            if candidate.distance > previous_edit_distance_candidate:
                break

            processed_candidate_score = self._process_candidates(text_for_prediction, candidate)
            if processed_candidate_score:
                candidate_score_pairs.append(processed_candidate_score)
                previous_edit_distance_candidate = candidate.distance

        if candidate_score_pairs:
            ranked_candidate_score_pairs = sorted(candidate_score_pairs, key=lambda x: x[1])
            return ranked_candidate_score_pairs[0]
        else:
            return None


logger = logging.getLogger(__name__)


def _pick_correct_word_form(word: Word) -> str:
    if word.lemma_normal_form:
        return word.lemma_normal_form
    else:
        return word.original_value
