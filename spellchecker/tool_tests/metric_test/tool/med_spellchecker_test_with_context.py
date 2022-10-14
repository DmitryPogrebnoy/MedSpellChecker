from tqdm import tqdm

from distilbert_candidate_ranker import RuDistilBertCandidateRanker
from med_spellchecker import MedSpellchecker
from metric_test_with_context import MetricTestWithContext
from roberta_candidate_ranker import RuRobertaCandidateRanker

med_spellchecker_ru_roberta = MedSpellchecker(candidate_ranker=RuRobertaCandidateRanker(True),
                                              words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
                                              encoding="UTF-8")

med_spellchecker_ru_distilbert = MedSpellchecker(candidate_ranker=RuDistilBertCandidateRanker(True),
                                                 words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
                                                 encoding="UTF-8")


def med_spellchecker_roberta_test(input_batches):
    return apply_model_to_test(input_batches, med_spellchecker_ru_roberta)


def med_spellchecker_distilbert_test(input_batches):
    return apply_model_to_test(input_batches, med_spellchecker_ru_distilbert)


def apply_model_to_test(input_batches, med_spellchecker):
    result = []
    timer = tqdm(input_batches)
    for batch in timer:
        fixed_text = med_spellchecker.fix_text(' '.join(batch))
        result.append(fixed_text.split())
    return timer.format_dict["elapsed"], result


def run_test(spellchecker_function):
    metric_test_with_context = MetricTestWithContext()
    test_med_spellchecker_result = metric_test_with_context.compute_all_metrics(spellchecker_function)
    return test_med_spellchecker_result


if __name__ == '__main__':
    """
    Run test with context for MedSpellchecker
    """
    test_result_roberta = run_test(med_spellchecker_roberta_test)
    print()
    print("MedSpellChecker with RoBERTa")
    print(test_result_roberta)
    test_result_distilbert = run_test(med_spellchecker_distilbert_test)
    print()
    print("MedSpellChecker with DistilBERT")
    print(test_result_distilbert)
