from tqdm import tqdm

from candidate_ranker import RuRobertaCandidateRanker
from med_spellchecker import MedSpellchecker
from metric_test_with_context import MetricTestWithContext


def med_spellchecker_test(input_batches):
    med_spellchecker = MedSpellchecker(
        words_list="../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", candidate_ranker=RuRobertaCandidateRanker(True)
    )
    result = []
    timer = tqdm(input_batches)
    for batch in timer:
        fixed_text = med_spellchecker.fix_text(' '.join(batch))
        result.append(fixed_text.split())
    return {"elapsed": timer.format_dict["elapsed"], "corrected_batch": result}


def perform_test():
    metric_test_with_context = MetricTestWithContext("../../../data/test/with_context/data_for_test_with_context.csv")
    test_med_spellchecker_result = metric_test_with_context.compute_all_metrics(
        med_spellchecker_test)
    return test_med_spellchecker_result


if __name__ == '__main__':
    """
    Run test with context for MedSpellchecker
    """
    test_result = perform_test()
    print(test_result)
