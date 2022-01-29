from new_perfect_spellchecker import NewPerfectSpellchecker


def test_load_save(spellchecker_by_two_word_list_2: NewPerfectSpellchecker):
    path = "test_load_save_state.json"
    spellchecker_by_two_word_list_2.save_state(path)
    spellchecker = NewPerfectSpellchecker(saved_state=path)
    assert spellchecker._version == 1
    assert spellchecker._words_dictionary == {'дом', 'ром'}
    assert spellchecker._deletes_word_dictionary == {
        'о': ['дом', 'ром'], 'м': ['дом', 'ром'], 'д': ['дом'], 'ом': ['дом', 'ром'],
        'дом': ['дом'], 'до': ['дом'], 'дм': ['дом'], 'ро': ['ром'], 'ром': ['ром'],
        'р': ['ром'], 'рм': ['ром']}
    assert spellchecker._edit_distance_comparer.algorithm == 1
    assert spellchecker._max_dictionary_word_length == 3
    assert spellchecker._max_dictionary_edit_distance == 2
