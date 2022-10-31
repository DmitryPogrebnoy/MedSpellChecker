from spellchecker import SpellChecker
from tqdm import tqdm

from common.metric_test_with_context import MetricTestWithContext
from other_spellcheckers.utils import SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, \
    MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT


def pyspellchecker_tool_test(input_sentences):
    speller = SpellChecker(language='ru')
    result = []
    timer = tqdm(input_sentences)
    for sentences in timer:
        result.append([speller.correction(word) for word in sentences])
    return timer.format_dict["elapsed"], result


def perform_test():
    metric_test_with_context = MetricTestWithContext()
    return metric_test_with_context.compute_all_metrics(
        SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        pyspellchecker_tool_test)


if __name__ == '__main__':
    """
    Run test with context for Pyspellchecker wrapper (pyspellchecker package)
    """
    test_result = perform_test()
    print(test_result)
