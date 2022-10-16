from aspell import Speller
from tqdm import tqdm

from common.metric_test_without_context import MetricTestWithoutContext
from other_spellcheckers.utils import ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT


def aspell_tool_test(input_word_list):
    speller = Speller(("lang", "ru"))
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
    metric_test_without_context = MetricTestWithoutContext()
    return metric_test_without_context.compute_all_metrics(ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT,
                                                           aspell_tool_test, aspell_tool_test)


if __name__ == '__main__':
    """
    Run test without context for Aspell wrapper (aspell-python-py3 package)
    
    Firstly you need to install libaspell-dev, aspell and aspell-ru (sudo apt-get libaspell-dev aspell aspell-ru)
    """
    test_result = perform_test()
    print(test_result)
