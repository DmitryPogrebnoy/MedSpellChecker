from collections import defaultdict

from new_perfect_spellchecker import NewPerfectSpellchecker


def test_delete_word_dict_by_one_word_list_1(spellchecker_by_one_word_list_1: NewPerfectSpellchecker):
    answer: defaultdict[str, list] = defaultdict(list,
                                                 {'дмик': ['домик'], 'доик': ['домик'], 'доми': ['домик'],
                                                  'домик': ['домик'], 'домк': ['домик'], 'омик': ['домик']})
    assert spellchecker_by_one_word_list_1._deletes_word_dictionary == answer


def test_word_dict_by_one_word_list_1(spellchecker_by_one_word_list_1: NewPerfectSpellchecker):
    assert spellchecker_by_one_word_list_1._words_dictionary == {"домик"}


def test_delete_word_dict_by_one_word_list_2(spellchecker_by_one_word_list_2: NewPerfectSpellchecker):
    answer: defaultdict[str, list] = defaultdict(list,
                                                 {'дик': ['домик'], 'дми': ['домик'],
                                                  'дмик': ['домик'], 'дмк': ['домик'],
                                                  'дои': ['домик'], 'доик': ['домик'],
                                                  'док': ['домик'], 'дом': ['домик'],
                                                  'доми': ['домик'], 'домик': ['домик'],
                                                  'домк': ['домик'], 'мик': ['домик'],
                                                  'оик': ['домик'], 'оми': ['домик'],
                                                  'омик': ['домик'], 'омк': ['домик']})
    assert spellchecker_by_one_word_list_2._deletes_word_dictionary == answer


def test_word_dict_by_one_word_list_2(spellchecker_by_one_word_list_2: NewPerfectSpellchecker):
    assert spellchecker_by_one_word_list_2._words_dictionary == {"домик"}


def test_delete_word_dict_by_one_word_list_3(spellchecker_by_one_word_list_3: NewPerfectSpellchecker):
    answer: defaultdict[str, list] = defaultdict(list, {'ди': ['домик'], 'дик': ['домик'], 'дк': ['домик'],
                                                        'дм': ['домик'], 'дми': ['домик'], 'дмик': ['домик'],
                                                        'дмк': ['домик'], 'до': ['домик'], 'дои': ['домик'],
                                                        'доик': ['домик'], 'док': ['домик'], 'дом': ['домик'],
                                                        'доми': ['домик'], 'домик': ['домик'], 'домк': ['домик'],
                                                        'ик': ['домик'], 'ми': ['домик'], 'мик': ['домик'],
                                                        'мк': ['домик'], 'ои': ['домик'], 'оик': ['домик'],
                                                        'ок': ['домик'], 'ом': ['домик'], 'оми': ['домик'],
                                                        'омик': ['домик'], 'омк': ['домик']})
    assert spellchecker_by_one_word_list_3._deletes_word_dictionary == answer


def test_word_dict_by_one_word_list_3(spellchecker_by_one_word_list_3: NewPerfectSpellchecker):
    assert spellchecker_by_one_word_list_3._words_dictionary == {"домик"}


def test_delete_word_dict_by_two_word_list_1(spellchecker_by_two_word_list_1: NewPerfectSpellchecker):
    answer: defaultdict[str, list] = defaultdict(list, {'дм': ['дом'], 'до': ['дом'], 'дом': ['дом'],
                                                        'ом': ['дом', 'ром'], 'рм': ['ром'], 'ро': ['ром'],
                                                        'ром': ['ром']})
    assert spellchecker_by_two_word_list_1._deletes_word_dictionary == answer


def test_word_dict_by_two_word_list_1(spellchecker_by_two_word_list_1: NewPerfectSpellchecker):
    assert spellchecker_by_two_word_list_1._words_dictionary == {'ром', 'дом'}


def test_delete_word_dict_by_two_word_list_2(spellchecker_by_two_word_list_2: NewPerfectSpellchecker):
    answer: defaultdict[str, list] = defaultdict(list, {'д': ['дом'], 'дм': ['дом'], 'до': ['дом'],
                                                        'дом': ['дом'], 'м': ['дом', 'ром'], 'о': ['дом', 'ром'],
                                                        'ом': ['дом', 'ром'], 'р': ['ром'], 'рм': ['ром'],
                                                        'ро': ['ром'], 'ром': ['ром']})
    assert spellchecker_by_two_word_list_2._deletes_word_dictionary == answer


def test_word_dict_by_two_word_list_2(spellchecker_by_two_word_list_2: NewPerfectSpellchecker):
    assert spellchecker_by_two_word_list_2._words_dictionary == {'ром', 'дом'}


def test_delete_word_dict_by_two_word_list_3(spellchecker_by_two_word_list_3: NewPerfectSpellchecker):
    answer: defaultdict[str, list] = defaultdict(list, {'': ['дом', 'ром'], 'д': ['дом'], 'дм': ['дом'],
                                                        'до': ['дом'], 'дом': ['дом'], 'м': ['дом', 'ром'],
                                                        'о': ['дом', 'ром'], 'ом': ['дом', 'ром'],
                                                        'р': ['ром'], 'рм': ['ром'], 'ро': ['ром'],
                                                        'ром': ['ром']})
    assert spellchecker_by_two_word_list_3._deletes_word_dictionary == answer


def test_word_dict_by_two_word_list_3(spellchecker_by_two_word_list_3: NewPerfectSpellchecker):
    assert spellchecker_by_two_word_list_3._words_dictionary == {'ром', 'дом'}
