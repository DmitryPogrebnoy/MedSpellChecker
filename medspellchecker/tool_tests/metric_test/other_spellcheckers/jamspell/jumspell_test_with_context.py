from jamspell import TSpellCorrector
from tqdm import tqdm

from medspellchecker.tool_tests.metric_test.common.metric_test_with_context import MetricTestWithContext
from medspellchecker.tool_tests.metric_test.utils import EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, \
    MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT

jumspell_model_lib = "../../../../../data/other_spellcheckers/jumspell/ru_small.bin"


def jumspell_test(input_sentence):
    jamspell = TSpellCorrector()
    jamspell.LoadLangModel(jumspell_model_lib)

    result = []
    timer = tqdm(input_sentence)
    for sentence in timer:
        result.append([jamspell.FixFragment(word) for word in sentence])
    return timer.format_dict["elapsed"], result


def perform_test():
    metric_test_with_context = MetricTestWithContext()
    return metric_test_with_context.compute_all_metrics(
        SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        lambda x: jumspell_test(x))


if __name__ == '__main__':
    """
    Run test with context for Jumspell

    For installing jumspell you need to install swig3.0 (sudo apt-get install swig3.0)
    """
    test_result = perform_test()
    print(test_result)
