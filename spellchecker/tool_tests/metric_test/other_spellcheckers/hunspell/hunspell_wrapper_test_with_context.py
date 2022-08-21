from hunspell import HunSpell
from tqdm import tqdm

from common.metric_test_with_context import MetricTestWithContext

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
    return {"elapsed": timer.format_dict["elapsed"], "corrected_batch": result}


def perform_test():
    metric_test_with_context = MetricTestWithContext(
        "../../../../../data/test/with_context/data_for_test_with_context.csv")
    return metric_test_with_context.compute_all_metrics(hunspell_tool_test)


if __name__ == '__main__':
    """
    Run test without context for Hunspell wrapper (hunspell package)

    Firstly you need to install libhunspell-dev, hunspell and hunspell-ru (sudo apt-get libhunspell-dev hunspell hunspell-ru)
    """
    test_result = perform_test()
    print(test_result)
