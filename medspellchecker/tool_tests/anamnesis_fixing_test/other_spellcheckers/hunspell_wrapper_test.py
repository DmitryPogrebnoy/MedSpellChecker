from hunspell import HunSpell
from tqdm import tqdm

from medspellchecker.tool.pre_post_processor import PreProcessor
from medspellchecker.tool_tests.anamnesis_fixing_test.common.anamnesis_fixing_test import perform_anamnesis_fixing_test


def hunspell_tool_test(input_sentence):
    hunspell_dic = '../../../../data/other_spellcheckers/hunspell/index.dic'
    hunspell_aff = '../../../../data/other_spellcheckers/hunspell/index.aff'
    pre_processor = PreProcessor()
    speller = HunSpell(hunspell_dic, hunspell_aff)
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
    Run test for Hunspell wrapper (hunspell package)

    Firstly you need to install libhunspell-dev, hunspell and hunspell-ru (sudo apt-get libhunspell-dev hunspell hunspell-ru)
    """
    perform_anamnesis_fixing_test(hunspell_tool_test, "hunspell_fix.csv")
