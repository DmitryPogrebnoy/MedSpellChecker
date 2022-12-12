from typing import List

import pytest

from abstract_candidate_ranker import AbstractCandidateRanker
from distilbert_candidate_ranker import RuDistilBertCandidateRanker
from medspellchecker import MedSpellchecker
from roberta_candidate_ranker import RuRobertaCandidateRanker


@pytest.fixture
def one_word_list() -> List[str]:
    return ["домик"]


@pytest.fixture
def two_word_list() -> List[str]:
    return ["дом", "ром"]


@pytest.fixture(params=[RuRobertaCandidateRanker(), RuDistilBertCandidateRanker()])
def ranker(request) -> AbstractCandidateRanker:
    return request.param


@pytest.fixture
def spellchecker_by_one_word_list_1(one_word_list: List[str], ranker: AbstractCandidateRanker) -> MedSpellchecker:
    return MedSpellchecker(candidate_ranker=ranker, words_list=one_word_list, max_dictionary_edit_distance=1)


@pytest.fixture
def spellchecker_by_one_word_list_2(one_word_list: List[str], ranker: AbstractCandidateRanker) -> MedSpellchecker:
    return MedSpellchecker(candidate_ranker=ranker, words_list=one_word_list, max_dictionary_edit_distance=2)


@pytest.fixture
def spellchecker_by_one_word_list_3(one_word_list: List[str], ranker: AbstractCandidateRanker) -> MedSpellchecker:
    return MedSpellchecker(candidate_ranker=ranker, words_list=one_word_list, max_dictionary_edit_distance=3)


@pytest.fixture
def spellchecker_by_two_word_list_1(two_word_list: List[str], ranker: AbstractCandidateRanker) -> MedSpellchecker:
    return MedSpellchecker(candidate_ranker=ranker, words_list=two_word_list, max_dictionary_edit_distance=1)


@pytest.fixture
def spellchecker_by_two_word_list_2(two_word_list: List[str], ranker: AbstractCandidateRanker) -> MedSpellchecker:
    return MedSpellchecker(candidate_ranker=ranker, words_list=two_word_list, max_dictionary_edit_distance=2)


@pytest.fixture
def spellchecker_by_two_word_list_3(two_word_list: List[str], ranker: AbstractCandidateRanker) -> MedSpellchecker:
    return MedSpellchecker(candidate_ranker=ranker, words_list=two_word_list, max_dictionary_edit_distance=3)
