from tqdm import tqdm

from distilbert_candidate_ranker import RuDistilBertCandidateRanker
from med_spellchecker import MedSpellchecker
from metric_test_with_context import MetricTestWithContext
from roberta_candidate_ranker import RuRobertaCandidateRanker


def med_spellchecker_roberta_test(input_batches):
    med_spellchecker = MedSpellchecker(
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", candidate_ranker=RuRobertaCandidateRanker(True)
    )
    return apply_model_to_test(input_batches, med_spellchecker)


def med_spellchecker_distilbert_test(input_batches):
    med_spellchecker = MedSpellchecker(
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", candidate_ranker=RuDistilBertCandidateRanker(True)
    )
    return apply_model_to_test(input_batches, med_spellchecker)


def apply_model_to_test(input_batches, med_spellchecker):
    result = []
    timer = tqdm(input_batches)
    for batch in timer:
        fixed_text = med_spellchecker.fix_text(' '.join(batch))
        result.append(fixed_text.split())
    return {"elapsed": timer.format_dict["elapsed"], "corrected_batch": result}


def run_test(spellchecker_function):
    metric_test_with_context = MetricTestWithContext(
        "../../../../data/test/with_context/data_for_test_with_context.csv")
    test_med_spellchecker_result = metric_test_with_context.compute_all_metrics(spellchecker_function)
    return test_med_spellchecker_result


if __name__ == '__main__':
    """
    Run test with context for MedSpellchecker
    """
    test_result_roberta = run_test(med_spellchecker_roberta_test)
    print("MedSpellChecker with RoBERTa")
    print(test_result_roberta)
    test_result_distilbert = run_test(med_spellchecker_distilbert_test)
    print("MedSpellChecker with DistilBERT")
    print(test_result_distilbert)
