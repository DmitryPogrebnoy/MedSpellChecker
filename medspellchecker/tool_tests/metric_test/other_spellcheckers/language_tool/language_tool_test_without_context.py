from language_tool_python import LanguageTool
from tqdm import tqdm

from medspellchecker.tool_tests.metric_test.common.metric_test_without_context import MetricTestWithoutContext
from medspellchecker.tool_tests.metric_test.utils import ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT


def language_tool_test(input_word_list):
    tool = LanguageTool('ru')
    result = []
    timer = tqdm(input_word_list)
    for word in timer:
        suggestions = tool.correct(word)
        result.append(suggestions.lower())
    return timer.format_dict["elapsed"], result


def perform_test():
    metric_test_without_context = MetricTestWithoutContext(True)
    return metric_test_without_context.compute_all_metrics(ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT,
                                                           language_tool_test, language_tool_test)


if __name__ == '__main__':
    """
    Run test without context for Language Tool (language-tool-python package)
    
    Firstly you need to install Java SDK (sudo apt-get install openjdk-17-jdk)
    """
    test_result = perform_test()
    print(test_result)
