from tqdm import tqdm

from medspellchecker.tool.distilbert_candidate_ranker import RuDistilBertCandidateRanker
from medspellchecker.tool.medspellchecker import MedSpellchecker
from medspellchecker.tool.roberta_candidate_ranker import RuRobertaCandidateRanker
from medspellchecker.tool.rubert_tiny2_candidate_ranker import RuBertTiny2CandidateRanker
from medspellchecker.tool.rubioberta_candidate_ranker import RuBioBertCandidateRanker
from medspellchecker.tool.rubioroberta_candidate_ranker import RuBioRobertCandidateRanker
from medspellchecker.tool_tests.metric_test.common.metric_test_with_context import MetricTestWithContext
from medspellchecker.tool_tests.metric_test.utils import EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, \
    MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT


def med_spellchecker_roberta_test_no_space_handling(input_batches):
    med_spellchecker_ru_roberta_no_space_handling = MedSpellchecker(
        candidate_ranker=RuRobertaCandidateRanker(True),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8")
    return apply_model_to_test(input_batches, med_spellchecker_ru_roberta_no_space_handling)


def med_spellchecker_distilbert_test_no_space_handling(input_batches):
    med_spellchecker_ru_distilbert_no_space_handling = MedSpellchecker(
        candidate_ranker=RuDistilBertCandidateRanker(True),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8")
    return apply_model_to_test(input_batches, med_spellchecker_ru_distilbert_no_space_handling)


def med_spellchecker_rubert_tiny2_test_no_space_handling(input_batches):
    med_spellchecker_rubert_tiny2_no_space_handling = MedSpellchecker(
        candidate_ranker=RuBertTiny2CandidateRanker(True),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8")
    return apply_model_to_test(input_batches, med_spellchecker_rubert_tiny2_no_space_handling)


def med_spellchecker_rubiobert_test_no_space_handling(input_batches):
    med_spellchecker_rubiobert_no_space_handling = MedSpellchecker(
        candidate_ranker=RuBioBertCandidateRanker(False),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", handle_compound_words=False)
    return apply_model_to_test(input_batches, med_spellchecker_rubiobert_no_space_handling)


def med_spellchecker_rubiorobert_test_no_space_handling(input_batches):
    med_spellchecker_rubiorobert_no_space_handling = MedSpellchecker(
        candidate_ranker=RuBioRobertCandidateRanker(False),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", handle_compound_words=False)
    return apply_model_to_test(input_batches, med_spellchecker_rubiorobert_no_space_handling)


def med_spellchecker_roberta_test_missing_space_handling(input_batches):
    med_spellchecker_ru_roberta_missing_space_handling = MedSpellchecker(
        candidate_ranker=RuRobertaCandidateRanker(True),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", handle_compound_words=True)
    return apply_model_to_test(input_batches, med_spellchecker_ru_roberta_missing_space_handling)


def med_spellchecker_distilbert_test_missing_space_handling(input_batches):
    med_spellchecker_ru_distilbert_missing_space_handling = MedSpellchecker(
        candidate_ranker=RuDistilBertCandidateRanker(False),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", handle_compound_words=True)
    return apply_model_to_test(input_batches, med_spellchecker_ru_distilbert_missing_space_handling)


def med_spellchecker_rubert_tiny2_test_missing_space_handling(input_batches):
    med_spellchecker_rubert_tiny2_missing_space_handling = MedSpellchecker(
        candidate_ranker=RuBertTiny2CandidateRanker(False),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", handle_compound_words=True)
    return apply_model_to_test(input_batches, med_spellchecker_rubert_tiny2_missing_space_handling)


def med_spellchecker_rubiobert_test_missing_space_handling(input_batches):
    med_spellchecker_rubiobert_missing_space_handling = MedSpellchecker(
        candidate_ranker=RuBioBertCandidateRanker(False),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", handle_compound_words=True)
    return apply_model_to_test(input_batches, med_spellchecker_rubiobert_missing_space_handling)


def med_spellchecker_rubiorobert_test_missing_space_handling(input_batches):
    med_spellchecker_rubiorobert_missing_space_handling = MedSpellchecker(
        candidate_ranker=RuBioRobertCandidateRanker(False),
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", handle_compound_words=True)
    return apply_model_to_test(input_batches, med_spellchecker_rubiorobert_missing_space_handling)


def apply_model_to_test(input_batches, med_spellchecker):
    result = []
    timer = tqdm(input_batches)
    for batch in timer:
        fixed_text = med_spellchecker.fix_text(' '.join(batch))
        result.append(fixed_text.split())
    return timer.format_dict["elapsed"], result


def run_test(spellchecker_function):
    metric_test_with_context = MetricTestWithContext()
    test_med_spellchecker_result = metric_test_with_context.compute_all_metrics(
        SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        spellchecker_function)
    return test_med_spellchecker_result


if __name__ == '__main__':
    """
    Run test with context for MedSpellchecker
    """
    # test_result_roberta_no_space_handling = run_test(med_spellchecker_roberta_test_no_space_handling)
    # print()
    # print("MedSpellChecker with RoBERTa no space handling")
    # print(test_result_roberta_no_space_handling)
    #
    # test_result_distilbert_no_space_handling = run_test(med_spellchecker_distilbert_test_no_space_handling)
    # print()
    # print("MedSpellChecker with DistilBERT no space handling")
    # print(test_result_distilbert_no_space_handling)
    #
    # test_result_rubert_tiny2_no_space_handling = run_test(med_spellchecker_rubert_tiny2_test_no_space_handling)
    # print()
    # print("MedSpellChecker with RuBertTiny2 no space handling")
    # print(test_result_rubert_tiny2_no_space_handling)
    #
    # test_result_rubiobert_no_space_handling = run_test(med_spellchecker_rubiobert_test_no_space_handling)
    # print()
    # print("MedSpellChecker with RuBioBert no space handling")
    # print(test_result_rubiobert_no_space_handling)
    #
    # test_result_rubioroberta_no_space_handling = run_test(med_spellchecker_rubiorobert_test_no_space_handling)
    # print()
    # print("MedSpellChecker with RuBioRoBerta no space handling")
    # print(test_result_rubioroberta_no_space_handling)

    # test_result_roberta_space_handling = run_test(med_spellchecker_roberta_test_missing_space_handling)
    # print()
    # print("MedSpellChecker with RoBERTa space handling")
    # print(test_result_roberta_space_handling)
    #
    # test_result_distilbert_space_handling = run_test(med_spellchecker_distilbert_test_missing_space_handling)
    # print()
    # print("MedSpellChecker with DistilBERT space handling")
    # print(test_result_distilbert_space_handling)
    #
    # test_result_rubert_tiny2_space_handling = run_test(med_spellchecker_rubert_tiny2_test_missing_space_handling)
    # print()
    # print("MedSpellChecker with RuBertTiny2 space handling")
    # print(test_result_rubert_tiny2_space_handling)

    test_result_rubiobert_space_handling = run_test(med_spellchecker_rubiobert_test_missing_space_handling)
    print()
    print("MedSpellChecker with RuBioBert space handling")
    print(test_result_rubiobert_space_handling)

    test_result_rubioroberta_space_handling = run_test(med_spellchecker_rubiorobert_test_missing_space_handling)
    print()
    print("MedSpellChecker with RuBioRoBerta space handling")
    print(test_result_rubioroberta_space_handling)
