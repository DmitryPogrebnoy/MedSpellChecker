from tqdm import tqdm

from distilbert_candidate_ranker import RuDistilBertCandidateRanker
from med_spellchecker import MedSpellchecker
from metric_test_without_context import MetricTestWithoutContext
from roberta_candidate_ranker import RuRobertaCandidateRanker


def med_spellchecker_roberta_test(input_word_list):
    med_spellchecker = MedSpellchecker(
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", candidate_ranker=RuRobertaCandidateRanker(True)
    )
    return apply_spellchecker_to_test_data(input_word_list, med_spellchecker)


def med_spellchecker_distilbert_test(input_word_list):
    med_spellchecker = MedSpellchecker(
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", candidate_ranker=RuDistilBertCandidateRanker(True)
    )
    return apply_spellchecker_to_test_data(input_word_list, med_spellchecker)


def apply_spellchecker_to_test_data(input_word_list, med_spellchecker):
    result = []
    timer = tqdm(input_word_list)
    for word in timer:
        fixed_text = med_spellchecker.fix_text(word)
        result.append(fixed_text)
    return {"elapsed": timer.format_dict["elapsed"], "corrected_word_list": result}


def run_test(error_precision_spellchecker_function, lexical_precision_spellchecker_function):
    metric_test_without_context = MetricTestWithoutContext(
        '../../../../data/test/without_context/error_precision_words.txt',
        '../../../../data/test/without_context/lexical_precision_words.txt')
    test_med_spellchecker_result = metric_test_without_context.compute_all_metrics(
        error_precision_spellchecker_function, lexical_precision_spellchecker_function)
    return test_med_spellchecker_result


if __name__ == '__main__':
    """
    Run test without context for MedSpellchecker
    """
    test_result_roberta = run_test(med_spellchecker_roberta_test, med_spellchecker_roberta_test)
    print("MedSpellChecker with RoBERTa")
    print(test_result_roberta)
    test_result_distilbert = run_test(med_spellchecker_distilbert_test, med_spellchecker_distilbert_test)
    print("MedSpellChecker with DistilBERT")
    print(test_result_distilbert)
