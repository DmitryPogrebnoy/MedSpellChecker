from tqdm import tqdm

from common.metric_test_without_context import MetricTestWithoutContext
from jamspell import TSpellCorrector

jumspell_model_lib = "../../../../../data/other_spellcheckers/jumspell/ru_small.bin"


def jumspell_test(input_word_list):
    jamspell = TSpellCorrector()
    jamspell.LoadLangModel(jumspell_model_lib)

    result = []
    timer = tqdm(input_word_list)
    for word in timer:
        suggestions = jamspell.FixFragment(word)
        result.append(suggestions)
    return {"elapsed": timer.format_dict["elapsed"], "corrected_word_list": result}


def perform_test():
    metric_test_without_context = MetricTestWithoutContext(
        '../../../../../data/test/without_context/error_precision_words.txt',
        '../../../../../data/test/without_context/lexical_precision_words.txt')
    return metric_test_without_context.compute_all_metrics(
        lambda x: jumspell_test(x),
        lambda x: jumspell_test(x))


if __name__ == '__main__':
    """
    Run test without context for Jumspell
    
    For installing jumspell you need to install swig3.0 (sudo apt-get install swig3.0)
    """
    test_result = perform_test()
    print(test_result)
