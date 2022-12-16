import enchant
from tqdm import tqdm

from medspellchecker.tool_tests.metric_test.common.metric_test_without_context import MetricTestWithoutContext
from medspellchecker.tool_tests.metric_test.other_spellcheckers.utils import ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT


def pyenchant_tool_test(input_word_list):
    speller = enchant.Dict("ru")
    result = []
    timer = tqdm(input_word_list)
    for word in timer:
        suggestions = speller.suggest(word)
        if len(suggestions) == 0:
            result.append(word)
        else:
            result.append(suggestions[0])
    return timer.format_dict["elapsed"], result


def perform_test():
    metric_test_without_context = MetricTestWithoutContext()
    return metric_test_without_context.compute_all_metrics(ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT,
                                                           pyenchant_tool_test, pyenchant_tool_test)


if __name__ == '__main__':
    """
    Run test without context for PyEnchant wrapper (pyenchant package)
    """
    test_result = perform_test()
    print(test_result)
