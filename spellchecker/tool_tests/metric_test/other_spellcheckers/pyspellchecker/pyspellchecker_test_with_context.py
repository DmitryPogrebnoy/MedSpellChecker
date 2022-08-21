from spellchecker import SpellChecker
from tqdm import tqdm

from common.metric_test_with_context import MetricTestWithContext


def pyspellchecker_tool_test(input_sentences):
    speller = SpellChecker(language='ru')
    result = []
    timer = tqdm(input_sentences)
    for sentences in timer:
        result.append([speller.correction(word) for word in sentences])
    return {"elapsed": timer.format_dict["elapsed"], "corrected_batch": result}


def perform_test():
    metric_test_with_context = MetricTestWithContext(
        "../../../../../data/test/with_context/data_for_test_with_context.csv")
    return metric_test_with_context.compute_all_metrics(pyspellchecker_tool_test)


if __name__ == '__main__':
    """
    Run test with context for Pyspellchecker wrapper (pyspellchecker package)
    """
    test_result = perform_test()
    print(test_result)
