from abc import abstractmethod
import dataclasses
import itertools
import logging
from typing import List, Optional, Tuple

from accelerate import Accelerator
import torch
from torch import IntTensor
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from medspellchecker.tool.abstract_candidate_ranker import AbstractCandidateRanker
from medspellchecker.tool.candidate_word import CandidateWord
from medspellchecker.tool.gpu_utils import set_device
from medspellchecker.tool.word import Word

logger = logging.getLogger(__name__)


class AbstractBertCandidateRanker(AbstractCandidateRanker):

    def __init__(self, pretrained_model_checkpoint: str,
                 use_treshold: bool = True, treshold: float = 0.0000001,
                 use_gpu: bool = True):
        self._use_treshold: bool = use_treshold
        self._treshold: float = treshold
        self._use_gpu: bool = set_device() if use_gpu else False

        accelerator: Accelerator = Accelerator(cpu=not self._use_gpu)
        self._tokenizer: PreTrainedTokenizer = accelerator.prepare(
            AutoTokenizer.from_pretrained(pretrained_model_checkpoint))
        self._model: PreTrainedModel = accelerator.prepare(
            AutoModelForMaskedLM.from_pretrained(pretrained_model_checkpoint))

    @property
    @abstractmethod
    def _version(self) -> int:
        """
        :return: version of ranker
        """

    def prepare_text_for_prediction(self, correct_words_before: List[Word],
                                    words_after: List[Word]) -> str:
        str_before = ' '.join([word.corrected_value for word in correct_words_before])
        str_word_token = self._tokenizer.mask_token
        str_after = ' '.join([word.original_value for word in words_after])

        return ' '.join([str_before, str_word_token, str_after])

    def create_texts_for_prediction_with_ids(self, input_ids: List[int],
                                             candidate_ids: List[int]) -> List[Tuple[IntTensor, IntTensor]]:
        results: List[Tuple[IntTensor, IntTensor]] = []
        input_ids_mask_pos: int = input_ids.index(self._tokenizer.mask_token_id)
        input_ids_before: List[int] = input_ids[:input_ids_mask_pos]
        input_ids_after: List[int] = input_ids[input_ids_mask_pos + 1:]
        for i in range(len(candidate_ids)):
            candidate_ids_before: List[int] = candidate_ids[:i]
            candidate_ids_after: List[int] = candidate_ids[i + 1:]
            patched_input_ids: IntTensor = IntTensor([list(itertools.chain.from_iterable(
                [input_ids_before, candidate_ids_before,
                 [self._tokenizer.mask_token_id], candidate_ids_after, input_ids_after]))])
            patched_attention_mask: IntTensor = IntTensor([list(itertools.repeat(1, patched_input_ids.size()[1]))])
            results.append((patched_input_ids, patched_attention_mask))
        return results

    def predict_score(self, correct_words_before: List[Word], words_after: List[Word], candidate_value: str) -> \
            Optional[float]:
        text_with_mask = self.prepare_text_for_prediction(correct_words_before, words_after)
        candidate_token_ids = self._tokenizer.encode(candidate_value, add_special_tokens=False)
        logger.debug(f"Candidate token ids: {candidate_token_ids}")

        inputs = self._tokenizer(text_with_mask, return_tensors='pt')
        logger.debug(f"Input token ids: {inputs}")

        original_input_ids = inputs["input_ids"][0].tolist()
        patched_inputs_data = self.create_texts_for_prediction_with_ids(original_input_ids, candidate_token_ids)

        ## Mean score of candidate word parts
        candidate_mean_score = 0.0
        for i, patched_item in enumerate(patched_inputs_data):
            patched_inputs_ids, patched_attention_mask = patched_item
            if self._use_gpu:
                patched_inputs_ids = patched_inputs_ids.cuda()
                patched_attention_mask = patched_attention_mask.cuda()

            token_logits = self._model(patched_inputs_ids, patched_attention_mask).logits
            logger.debug(f"Token logits: {token_logits}")

            mask_token_index = torch.where(inputs["input_ids"] == self._tokenizer.mask_token_id)[1]
            mask_token_logits = token_logits[0, mask_token_index, :]
            softmax_mask_token_logits = torch.softmax(mask_token_logits, dim=1)

            if logger.getEffectiveLevel() == logging.DEBUG:
                top_20 = torch.topk(softmax_mask_token_logits, 20, dim=1)
                top_20_tokens = zip(top_20.indices[0].tolist(), top_20.values[0].tolist())
                logger.debug("----Top 20 most suitable tokens----")
                for token, score in top_20_tokens:
                    logger.debug(f"{self._tokenizer.decode([token])}, score: {score}")

            candidate_mean_score += softmax_mask_token_logits[:, candidate_token_ids[i]][0].item()
        candidate_mean_score = candidate_mean_score / len(candidate_token_ids)
        logger.debug(f"Candidate - {candidate_value}, score {candidate_mean_score}")
        if self._use_treshold and candidate_mean_score > self._treshold or not self._use_treshold:
            return candidate_mean_score
        else:
            return None

    def rank_candidates(self, current_word: Word,
                        correct_words_before: List[Word],
                        words_after: List[Word],
                        candidates: List[CandidateWord]) -> List[CandidateWord]:
        if not candidates:
            return candidates

        # candidates must be ordered by edit distance
        candidates.sort(key=lambda x: x.distance)

        candidate_score_pairs = []

        for candidate in candidates:
            candidate_score = self.predict_score(correct_words_before, words_after, candidate.value)
            if candidate_score:
                candidate_score_pairs.append((candidate, candidate_score))

        ranked_candidate_score_pairs = sorted(candidate_score_pairs, key=lambda x: x[1])
        ranked_candidates = map(lambda x: x[0], ranked_candidate_score_pairs)
        return list(ranked_candidates)

    def pick_top_candidate(self, current_word: Word,
                           correct_words_before: List[Word],
                           words_after: List[Word],
                           candidates: List[CandidateWord]) -> Optional[Tuple[CandidateWord, float]]:
        if not candidates:
            return None

        # candidates must be ordered by edit distance
        candidates.sort(key=lambda x: x.distance)

        candidate_score_pairs = []
        candidates = list(filter(lambda x: x.distance == candidates[0].distance, candidates))

        for candidate in candidates:
            candidate_score = self.predict_score(correct_words_before, words_after, candidate.value)
            if candidate_score:
                candidate_score_pairs.append((candidate, candidate_score))

        if candidate_score_pairs:
            ranked_candidate_score_pairs = sorted(candidate_score_pairs, key=lambda x: x[1])
            return ranked_candidate_score_pairs[0]
        else:
            return None

    def pick_top_candidate_with_next_word(self, current_word: Word,
                                          next_word_str: str,
                                          correct_words_before: List[Word],
                                          words_after: List[Word],
                                          candidates: List[CandidateWord]) -> Optional[Tuple[CandidateWord, float]]:
        if not candidates:
            return None

        # candidates must be ordered by edit distance
        candidates.sort(key=lambda x: x.distance)

        candidate_score_pairs = []
        candidates = list(filter(lambda x: x.distance == candidates[0].distance, candidates))

        for candidate in candidates:
            candidate_score = self.predict_score(correct_words_before, words_after, candidate.value)
            corrected_current_word = dataclasses.replace(current_word)
            corrected_current_word.corrected_value = candidate.value
            before = correct_words_before + [corrected_current_word]
            after = words_after[1:]
            next_word_score = self.predict_score(before, after, next_word_str)
            if candidate_score and next_word_score:
                candidate_score_pairs.append((candidate, (candidate_score + next_word_score) / 2))

        if candidate_score_pairs:
            ranked_candidate_score_pairs = sorted(candidate_score_pairs, key=lambda x: x[1])
            return ranked_candidate_score_pairs[0]
        else:
            return None
