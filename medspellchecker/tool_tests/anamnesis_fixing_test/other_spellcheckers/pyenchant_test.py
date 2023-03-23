import enchant
from tqdm import tqdm

from medspellchecker.tool.pre_post_processor import PreProcessor
from medspellchecker.tool_tests.anamnesis_fixing_test.common.anamnesis_fixing_test import perform_anamnesis_fixing_test


def pyenchant_tool_test(input_sentence):
    pre_processor = PreProcessor()
    speller = enchant.Dict("ru")
    result = []
    timer = tqdm(input_sentence)
    for sentence in timer:
        corrected_sentence = []
        for word in sentence:
            if pre_processor.is_valid_token(word):
                suggestions = speller.suggest(word)
                if len(suggestions) == 0:
                    corrected_sentence.append(word)
                else:
                    corrected_sentence.append(suggestions[0])
            else:
                corrected_sentence.append(word)
        result.append(" ".join(corrected_sentence))
    return result


if __name__ == '__main__':
    """
    Run test for PyEnchant wrapper (pyenchant package)
    """
    perform_anamnesis_fixing_test(pyenchant_tool_test, "pyenchant_fix.csv")
