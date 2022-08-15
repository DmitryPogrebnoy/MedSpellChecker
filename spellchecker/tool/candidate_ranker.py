import logging
from abc import abstractmethod, ABC
from enum import IntEnum
from typing import List, final, Optional, Tuple

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForMaskedLM

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


@final
class RuRobertaCandidateRanker(AbstractCandidateRanker):
    _pretrained_model_checkpoint = "DmitryPogrebnoy/MedRuRobertaLarge"

    def __init__(self, use_treshold: bool = True, use_gpu: bool = True):
        self._version = 1
        self._use_treshold = use_treshold
        self._treshold = 0.000001
        # Required gpu memory in Mb
        self._required_gpu_memory: int = 8192
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        self._use_gpu = use_gpu and torch.cuda.is_available() and gpu_memory > self._required_gpu_memory

        if torch.cuda.is_available():
            print("This machine has Cuda available!")
        else:
            print("This machine hasn't Cuda available!")

        print(f"Cuda device has {int(gpu_memory)} Mb memory")

        if gpu_memory > self._required_gpu_memory:
            print(f"Memory of the Cuda device is enough (>{self._required_gpu_memory}Mb) to use this model on the GPU.")
        else:
            print(f"Memory of the Cuda device is NOT enough (<{self._required_gpu_memory}Mb) to use this model on the "
                  "GPU.")

        if self._use_gpu:
            print("Model RuRobertaCandidateRanker use GPU")
            accelerator = Accelerator(fp16=True)
            self._tokenizer = accelerator.prepare(
                AutoTokenizer.from_pretrained(RuRobertaCandidateRanker._pretrained_model_checkpoint))
            self._model = accelerator.prepare(
                AutoModelForMaskedLM.from_pretrained(RuRobertaCandidateRanker._pretrained_model_checkpoint))
        else:
            print("Model RuRobertaCandidateRanker use CPU")
            self._tokenizer = AutoTokenizer.from_pretrained(RuRobertaCandidateRanker._pretrained_model_checkpoint)
            self._model = AutoModelForMaskedLM.from_pretrained(RuRobertaCandidateRanker._pretrained_model_checkpoint)

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
        attention_mask = inputs["input_ids"]
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

        ## Sum score of candidate word parts
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
