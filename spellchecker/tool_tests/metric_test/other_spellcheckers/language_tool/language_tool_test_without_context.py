from tqdm import tqdm

from common.metric_test_without_context import MetricTestWithoutContext
from language_tool_python import LanguageTool


def language_tool_test(input_word_list):
    tool = LanguageTool('ru')
    result = []
    timer = tqdm(input_word_list)
    for word in timer:
        suggestions = tool.correct(word)
        result.append(suggestions)
    return {"elapsed": timer.format_dict["elapsed"], "corrected_word_list": result}


def perform_test():
    metric_test_without_context = MetricTestWithoutContext(
        '../../../../../data/test/without_context/error_precision_words.txt',
        '../../../../../data/test/without_context/lexical_precision_words.txt')
    return metric_test_without_context.compute_all_metrics(language_tool_test, language_tool_test)


if __name__ == '__main__':
    """
    Run test without context for Language Tool (language-tool-python package)
    
    Firstly you need to install Java SDK (sudo apt-get install openjdk-17-jdk)
    """
    test_result = perform_test()
    print(test_result)
