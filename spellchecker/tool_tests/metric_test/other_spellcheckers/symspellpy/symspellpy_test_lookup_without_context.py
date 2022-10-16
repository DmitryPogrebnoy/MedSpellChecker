from symspellpy import SymSpell, Verbosity
from tqdm import tqdm

from common.metric_test_without_context import MetricTestWithoutContext
from other_spellcheckers.utils import ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT

basic_frequency_dict = '../../../../../data/other_spellcheckers/symspell/ru-100k.txt'


def symspell_py_lookup_test(frequency_dict_path, input_word_list):
    sym_spell_py = SymSpell()
    sym_spell_py.load_dictionary(frequency_dict_path, 0, 1, encoding="UTF8")

    result = []
    timer = tqdm(input_word_list)
    for word in timer:
        suggestions = sym_spell_py.lookup(word, Verbosity.TOP, max_edit_distance=2, include_unknown=True)
        result.append(suggestions[0].term)
    return {"elapsed": timer.format_dict["elapsed"], "corrected_word_list": result}


def perform_test():
    metric_test_without_context = MetricTestWithoutContext()
    return metric_test_without_context.compute_all_metrics(
        ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT,
        lambda x: symspell_py_lookup_test(basic_frequency_dict, x),
        lambda x: symspell_py_lookup_test(basic_frequency_dict, x))


if __name__ == '__main__':
    """
    Run test without context for SymSpellPy (lookup method)
    """
    test_result = perform_test()
    print(test_result)
