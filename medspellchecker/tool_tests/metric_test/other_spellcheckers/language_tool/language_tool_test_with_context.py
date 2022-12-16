from language_tool_python import LanguageTool
from tqdm import tqdm

from medspellchecker.tool_tests.metric_test.common.metric_test_with_context import MetricTestWithContext
from medspellchecker.tool_tests.metric_test.other_spellcheckers.utils import \
    SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, \
    MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT


def language_tool_test(input_sentence):
    tool = LanguageTool('ru')
    result = []
    timer = tqdm(input_sentence)
    for sentence in timer:
        result.append([tool.correct(word) for word in sentence])
    return timer.format_dict["elapsed"], result


def perform_test():
    metric_test_with_context = MetricTestWithContext()
    return metric_test_with_context.compute_all_metrics(
        SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        language_tool_test)


if __name__ == '__main__':
    """
    Run test with context for Language Tool (language-tool-python package)
    
    Firstly you need to install Java SDK (sudo apt-get install openjdk-17-jdk)
    """
    test_result = perform_test()
    print(test_result)