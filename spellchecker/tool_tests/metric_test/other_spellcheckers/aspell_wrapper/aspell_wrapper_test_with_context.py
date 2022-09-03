from aspell import Speller
from tqdm import tqdm

from common.metric_test_with_context import MetricTestWithContext


def aspell_tool_test(input_sentence):
    speller = Speller(("lang", "ru"))
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
    return {"elapsed": timer.format_dict["elapsed"], "corrected_batch": result}


def perform_test():
    metric_test_with_context = MetricTestWithContext(
        "../../../../../data/test/with_context/data_for_test_with_context.csv")
    return metric_test_with_context.compute_all_metrics(aspell_tool_test)


if __name__ == '__main__':
    """
    Run test with context for Aspell wrapper (aspell-python-py3 package)

    Firstly you need to install libaspell-dev, aspell and aspell-ru (sudo apt-get libaspell-dev aspell aspell-ru)
    """
    test_result = perform_test()
    print(test_result)