from tqdm import tqdm

from candidate_ranker import RuRobertaCandidateRanker
from med_spellchecker import MedSpellchecker
from common.metric_test_without_context import MetricTestWithoutContext


def med_spellchecker_test(input_word_list):
    med_spellchecker = MedSpellchecker(
        words_list="../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", candidate_ranker=RuRobertaCandidateRanker(True)
    )
    result = []
    timer = tqdm(input_word_list)
    for word in timer:
        fixed_text = med_spellchecker.fix_text(word)
        result.append(fixed_text)
    return {"elapsed": timer.format_dict["elapsed"], "corrected_word_list": result}


def perform_test():
    metric_test_without_context = MetricTestWithoutContext(
        '../../../../data/test/without_context/error_precision_words.txt',
        '../../../../data/test/without_context/lexical_precision_words.txt')
    test_med_spellchecker_result = metric_test_without_context.compute_all_metrics(
        med_spellchecker_test, med_spellchecker_test)
    return test_med_spellchecker_result


if __name__ == '__main__':
    """
    Run test without context for MedSpellchecker
    """
    test_result = perform_test()
    print(test_result)
