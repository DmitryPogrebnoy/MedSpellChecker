from symspellpy import SymSpell, Verbosity
from tqdm import tqdm

from common.metric_test_with_context import MetricTestWithContext
from other_spellcheckers.utils import ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT

basic_frequency_dict = '../../../../../data/other_spellcheckers/symspell/ru-100k.txt'


def symspell_py_lookup_test(frequency_dict_path, input_sentence):
    sym_spell_py = SymSpell()
    sym_spell_py.load_dictionary(frequency_dict_path, 0, 1, encoding="UTF8")

    result = []
    timer = tqdm(input_sentence)
    for sentence in timer:
        corrected_sentence = []
        for word in sentence:
            suggestions = sym_spell_py.lookup(word, Verbosity.TOP, max_edit_distance=2, include_unknown=True)
            corrected_sentence.append(suggestions[0].term)
        result.append(corrected_sentence)
    return timer.format_dict["elapsed"], result


def perform_test():
    metric_test_with_context = MetricTestWithContext()
    return metric_test_with_context.compute_all_metrics(
        ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        lambda x: symspell_py_lookup_test(basic_frequency_dict, x))


if __name__ == '__main__':
    """
    Run test with context for SymSpellPy (lookup method)
    """
    test_result = perform_test()
    print(test_result)
