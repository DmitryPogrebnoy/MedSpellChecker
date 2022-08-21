from jamspell import TSpellCorrector
from tqdm import tqdm

from common.metric_test_with_context import MetricTestWithContext

jumspell_model_lib = "../../../../../data/other_spellcheckers/jumspell/ru_small.bin"


def jumspell_test(input_sentence):
    jamspell = TSpellCorrector()
    jamspell.LoadLangModel(jumspell_model_lib)

    result = []
    timer = tqdm(input_sentence)
    for sentence in timer:
        result.append([jamspell.FixFragment(word) for word in sentence])
    return {"elapsed": timer.format_dict["elapsed"], "corrected_batch": result}


def perform_test():
    metric_test_with_context = MetricTestWithContext(
        "../../../../../data/test/with_context/data_for_test_with_context.csv")
    return metric_test_with_context.compute_all_metrics(lambda x: jumspell_test(x))


if __name__ == '__main__':
    """
    Run test with context for Jumspell

    For installing jumspell you need to install swig3.0 (sudo apt-get install swig3.0)
    """
    test_result = perform_test()
    print(test_result)
