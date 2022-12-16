import enchant
from tqdm import tqdm

from medspellchecker.tool_tests.metric_test.common.metric_test_with_context import MetricTestWithContext
from medspellchecker.tool_tests.metric_test.other_spellcheckers.utils import \
    SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, \
    MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT


def pyenchant_tool_test(input_sentence):
    speller = enchant.Dict("ru")
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
    return metric_test_with_context.compute_all_metrics(
        SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        pyenchant_tool_test)


if __name__ == '__main__':
    """
    Run test with context for PyEnchant wrapper (pyenchant package)
    """
    test_result = perform_test()
    print(test_result)
