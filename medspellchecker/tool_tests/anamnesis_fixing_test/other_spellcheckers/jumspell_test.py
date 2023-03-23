from jamspell import TSpellCorrector
from tqdm import tqdm

from medspellchecker.tool.pre_post_processor import PreProcessor
from medspellchecker.tool_tests.anamnesis_fixing_test.common.anamnesis_fixing_test import perform_anamnesis_fixing_test

jumspell_model_lib = "../../../../../data/other_spellcheckers/jumspell/ru_small.bin"


def jumspell_test(input_sentence):
    jamspell = TSpellCorrector()
    jamspell.LoadLangModel(jumspell_model_lib)
    pre_processor = PreProcessor()

    result = []
    timer = tqdm(input_sentence)
    for sentence in timer:
        corrected_sentence = []
        for word in sentence:
            if pre_processor.is_valid_token(word):
                fixed_word = jamspell.FixFragment(word)
                corrected_sentence.append(fixed_word)
            else:
                corrected_sentence.append(word)
        result.append(" ".join(corrected_sentence))
    return result


if __name__ == '__main__':
    """
    Run test for Jumspell

    For installing jumspell you need to install swig3.0 (sudo apt-get install swig3.0)
    """
    perform_anamnesis_fixing_test(jumspell_test, "jumspell_fix.csv")
