from symspellpy import SymSpell
from tqdm import tqdm

from medspellchecker.tool.pre_post_processor import PreProcessor
from medspellchecker.tool_tests.anamnesis_fixing_test.common.anamnesis_fixing_test import perform_anamnesis_fixing_test


def symspell_py_lookup_compound_test(input_sentence):
    basic_frequency_dict = '../../../../data/other_spellcheckers/symspell/ru-100k.txt'
    sym_spell_py = SymSpell()
    sym_spell_py.load_dictionary(basic_frequency_dict, 0, 1, encoding="UTF8")
    pre_processor = PreProcessor()

    result = []
    timer = tqdm(input_sentence)
    for sentence in timer:
        corrected_sentence = []
        for word in sentence:
            if pre_processor.is_valid_token(word):
                suggestions = sym_spell_py.lookup_compound(word, max_edit_distance=2)
                if len(suggestions) == 0:
                    corrected_sentence.append(word)
                else:
                    corrected_sentence.append(suggestions[0].term)
            else:
                corrected_sentence.append(word)
        result.append(" ".join(corrected_sentence))
    return result


if __name__ == '__main__':
    """
    Run test for SymSpellPy (lookup_compound method)
    """
    perform_anamnesis_fixing_test(symspell_py_lookup_compound_test, "symspellpy_compound_fix.csv")
