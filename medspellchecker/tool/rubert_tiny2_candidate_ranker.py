import logging
from typing import final

from medspellchecker.tool.abstract_bert_candidate_ranker import AbstractBertCandidateRanker

logger = logging.getLogger(__name__)


@final
class RuBertTiny2CandidateRanker(AbstractBertCandidateRanker):
    _pretrained_model_checkpoint: str = "DmitryPogrebnoy/MedRuBertTiny2"

    def __init__(self, use_treshold: bool = True, use_gpu: bool = True):
        super().__init__(RuBertTiny2CandidateRanker._pretrained_model_checkpoint, use_treshold, 0.0000001, use_gpu)

    @property
    def _version(self) -> int:
        return 1
