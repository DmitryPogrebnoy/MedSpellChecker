from symspellpy import SymSpell
from tqdm import tqdm

from common.metric_test_with_context import MetricTestWithContext

basic_frequency_dict = '../../../../../data/other_spellcheckers/symspell/ru-100k.txt'


def symspell_py_lookup_compound_test(frequency_dict_path, input_sentence):
    sym_spell_py = SymSpell()
    sym_spell_py.load_dictionary(frequency_dict_path, 0, 1, encoding="UTF8")

    result = []
    timer = tqdm(input_sentence)
    for sentence in timer:
        corrected_sentence = []
        for word in sentence:
            suggestions = sym_spell_py.lookup_compound(word, max_edit_distance=2)
            corrected_sentence.append(suggestions[0].term)
        result.append(corrected_sentence)
    return {"elapsed": timer.format_dict["elapsed"], "corrected_batch": result}


def perform_test():
    metric_test_with_context = MetricTestWithContext(
        "../../../../../data/test/with_context/data_for_test_with_context.csv")
    return metric_test_with_context.compute_all_metrics(
        lambda x: symspell_py_lookup_compound_test(basic_frequency_dict, x))


if __name__ == '__main__':
    """
    Run test with context for SymSpellPy (lookup_compound method)
    """
    test_result = perform_test()
    print(test_result)