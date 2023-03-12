from spellchecker import SpellChecker
from tqdm import tqdm

from medspellchecker.tool_tests.metric_test.common.metric_test_without_context import MetricTestWithoutContext
from medspellchecker.tool_tests.metric_test.utils import ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT


def pyspellchecker_tool_test(input_word_list):
    speller = SpellChecker(language='ru')
    result = []
    timer = tqdm(input_word_list)
    for word in timer:
        suggestions = speller.correction(word)
        result.append(suggestions)
    return timer.format_dict["elapsed"], result


def perform_test():
    metric_test_without_context = MetricTestWithoutContext()
    return metric_test_without_context.compute_all_metrics(ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT,
                                                           pyspellchecker_tool_test, pyspellchecker_tool_test)


if __name__ == '__main__':
    """
    Run test without context for Pyspellchecker wrapper (pyspellchecker package)
    """
    test_result = perform_test()
    print(test_result)
