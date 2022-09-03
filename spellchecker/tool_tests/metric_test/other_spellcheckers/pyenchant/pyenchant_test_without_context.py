from tqdm import tqdm

from common.metric_test_without_context import MetricTestWithoutContext
import enchant

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
    return {"elapsed": timer.format_dict["elapsed"], "corrected_word_list": result}


def perform_test():
    metric_test_without_context = MetricTestWithoutContext(
        '../../../../../data/test/without_context/error_precision_words.txt',
        '../../../../../data/test/without_context/lexical_precision_words.txt')
    return metric_test_without_context.compute_all_metrics(pyenchant_tool_test, pyenchant_tool_test)


if __name__ == '__main__':
    """
    Run test without context for PyEnchant wrapper (pyenchant package)
    """
    test_result = perform_test()
    print(test_result)