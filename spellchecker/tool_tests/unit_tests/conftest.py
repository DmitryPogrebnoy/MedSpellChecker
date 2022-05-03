from typing import List

import pytest

from med_spellchecker import MedSpellchecker


@pytest.fixture
def one_word_list() -> List[str]:
    return ["домик"]


@pytest.fixture
def two_word_list() -> List[str]:
    return ["дом", "ром"]


@pytest.fixture
def spellchecker_by_one_word_list_1(one_word_list: List[str]) -> MedSpellchecker:
    return MedSpellchecker(one_word_list, max_dictionary_edit_distance=1)


@pytest.fixture
def spellchecker_by_one_word_list_2(one_word_list: List[str]) -> MedSpellchecker:
    return MedSpellchecker(one_word_list, max_dictionary_edit_distance=2)


@pytest.fixture
def spellchecker_by_one_word_list_3(one_word_list: List[str]) -> MedSpellchecker:
    return MedSpellchecker(one_word_list, max_dictionary_edit_distance=3)


@pytest.fixture
def spellchecker_by_two_word_list_1(two_word_list: List[str]) -> MedSpellchecker:
    return MedSpellchecker(two_word_list, max_dictionary_edit_distance=1)


@pytest.fixture
def spellchecker_by_two_word_list_2(two_word_list: List[str]) -> MedSpellchecker:
    return MedSpellchecker(two_word_list, max_dictionary_edit_distance=2)


@pytest.fixture
def spellchecker_by_two_word_list_3(two_word_list: List[str]) -> MedSpellchecker:
    return MedSpellchecker(two_word_list, max_dictionary_edit_distance=3)
