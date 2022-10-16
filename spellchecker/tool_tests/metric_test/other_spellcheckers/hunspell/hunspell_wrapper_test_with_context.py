from hunspell import HunSpell
from tqdm import tqdm

from common.metric_test_with_context import MetricTestWithContext
from other_spellcheckers.utils import ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT

hunspell_dic = '../../../../../data/other_spellcheckers/hunspell/index.dic'
hunspell_aff = '../../../../../data/other_spellcheckers/hunspell/index.aff'


def hunspell_tool_test(input_sentence):
    speller = HunSpell(hunspell_dic, hunspell_aff)
    result = []
    timer = tqdm(input_sentence)
    for sentence in timer:
        corrected_sentence = []
        for word in sentence:
            suggestions = speller.suggest(word)
            if len(suggestions) == 0:
                corrected_sentence.append(word)
            else:
                corrected_sentence.append(suggestions[0])
        result.append(corrected_sentence)
    return timer.format_dict["elapsed"], result


def perform_test():
    metric_test_with_context = MetricTestWithContext()
    return metric_test_with_context.compute_all_metrics(ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, hunspell_tool_test)


if __name__ == '__main__':
    """
    Run test without context for Hunspell wrapper (hunspell package)

    Firstly you need to install libhunspell-dev, hunspell and hunspell-ru (sudo apt-get libhunspell-dev hunspell hunspell-ru)
    """
    test_result = perform_test()
    print(test_result)
