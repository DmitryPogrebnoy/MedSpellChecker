from spellchecker import SpellChecker
from tqdm import tqdm

from medspellchecker.tool.pre_post_processor import PreProcessor
from medspellchecker.tool_tests.anamnesis_fixing_test.common.anamnesis_fixing_test import perform_anamnesis_fixing_test


def pyspellchecker_tool_test(input_sentences):
    pre_processor = PreProcessor()
    speller = SpellChecker(language='ru')
    result = []
    timer = tqdm(input_sentences)
    for sentence in timer:
        corrected_sentence = []
        for word in sentence:
            if pre_processor.is_valid_token(word):
                fixed_word = speller.correction(word)
                corrected_sentence.append(fixed_word)
            else:
                corrected_sentence.append(word)
        result.append(" ".join(corrected_sentence))
    return result


if __name__ == '__main__':
    """
    Run test with context for Pyspellchecker wrapper (pyspellchecker package)
    """
    perform_anamnesis_fixing_test(pyspellchecker_tool_test, "pyspellchecker_fix.csv")
