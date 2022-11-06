from tqdm import tqdm

from distilbert_candidate_ranker import RuDistilBertCandidateRanker
from med_spellchecker import MedSpellchecker
from metric_test_without_context import MetricTestWithoutContext
from roberta_candidate_ranker import RuRobertaCandidateRanker
from tool.utils import ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT


def med_spellchecker_roberta_no_space_handling_test(input_word_list):
    med_spellchecker_ru_roberta_no_space_handling = MedSpellchecker(
        candidate_ranker=RuRobertaCandidateRanker(True),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8")
    return apply_spellchecker_to_test_data(input_word_list, med_spellchecker_ru_roberta_no_space_handling)


def med_spellchecker_distilbert_no_space_handling_test(input_word_list):
    med_spellchecker_ru_distilbert_no_space_handling = MedSpellchecker(
        candidate_ranker=RuDistilBertCandidateRanker(True),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8")
    return apply_spellchecker_to_test_data(input_word_list, med_spellchecker_ru_distilbert_no_space_handling)


def med_spellchecker_roberta_missing_space_handling_test(input_word_list):
    med_spellchecker_ru_roberta_missing_space_handling = MedSpellchecker(
        candidate_ranker=RuRobertaCandidateRanker(True),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", handle_compound_words=True)
    return apply_spellchecker_to_test_data(input_word_list, med_spellchecker_ru_roberta_missing_space_handling)


def med_spellchecker_distilbert_missing_space_handling_test(input_word_list):
    med_spellchecker_ru_distilbert_missing_space_handling = MedSpellchecker(
        candidate_ranker=RuDistilBertCandidateRanker(True),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", handle_compound_words=True)
    return apply_spellchecker_to_test_data(input_word_list, med_spellchecker_ru_distilbert_missing_space_handling)


def apply_spellchecker_to_test_data(input_word_list, med_spellchecker):
    result = []
    timer = tqdm(input_word_list)
    for word in timer:
        fixed_text = med_spellchecker.fix_text(word)
        result.append(fixed_text)
    return timer.format_dict["elapsed"], result


def run_test(error_precision_spellchecker_function, lexical_precision_spellchecker_function):
    metric_test_without_context = MetricTestWithoutContext()
    test_med_spellchecker_result = metric_test_without_context.compute_all_metrics(
        ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT,
        error_precision_spellchecker_function, lexical_precision_spellchecker_function)
    return test_med_spellchecker_result


if __name__ == '__main__':
    """
    Run test without context for MedSpellchecker
    """
    # test_result_roberta = run_test(med_spellchecker_roberta_no_space_handling_test,
    #                                med_spellchecker_roberta_no_space_handling_test)
    # print()
    # print("MedSpellChecker with RoBERTa no space handling")
    # print(test_result_roberta)
    # test_result_distilbert = run_test(med_spellchecker_distilbert_no_space_handling_test,
    #                                   med_spellchecker_distilbert_no_space_handling_test)
    # print()
    # print("MedSpellChecker with DistilBERT no space handling")
    # print(test_result_distilbert)
    test_result_roberta_missing_space_handling = run_test(med_spellchecker_roberta_missing_space_handling_test,
                                                          med_spellchecker_roberta_missing_space_handling_test)
    print()
    print("MedSpellChecker with RoBERTa space handling")
    print(test_result_roberta_missing_space_handling)
    test_result_distilbert_missing_space_handling = run_test(med_spellchecker_distilbert_missing_space_handling_test,
                                                             med_spellchecker_distilbert_missing_space_handling_test)
    print()
    print("MedSpellChecker with DistilBERT space handling")
    print(test_result_distilbert_missing_space_handling)
