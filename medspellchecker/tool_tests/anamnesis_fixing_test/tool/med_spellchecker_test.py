from tqdm import tqdm

from medspellchecker.tool.distilbert_candidate_ranker import RuDistilBertCandidateRanker
from medspellchecker.tool.medspellchecker import MedSpellchecker
from medspellchecker.tool.roberta_candidate_ranker import RuRobertaCandidateRanker
from medspellchecker.tool.rubert_tiny2_candidate_ranker import RuBertTiny2CandidateRanker
from medspellchecker.tool.rubioberta_candidate_ranker import RuBioBertCandidateRanker
from medspellchecker.tool.rubioroberta_candidate_ranker import RuBioRobertCandidateRanker
from medspellchecker.tool_tests.anamnesis_fixing_test.common.anamnesis_fixing_test import perform_anamnesis_fixing_test


def med_spellchecker_roberta_test_missing_space_handling(input_batches):
    med_spellchecker_ru_roberta_missing_space_handling = MedSpellchecker(
        candidate_ranker=RuRobertaCandidateRanker(False),
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
        result.append(fixed_text)
    return result


if __name__ == '__main__':
    """
    Run test for MedSpellchecker
    """
    perform_anamnesis_fixing_test(med_spellchecker_roberta_test_missing_space_handling,
                                  "medspellchecker_roberta_fix.csv")
    perform_anamnesis_fixing_test(med_spellchecker_distilbert_test_missing_space_handling,
                                  "medspellchecker_distilbert_fix.csv")
    perform_anamnesis_fixing_test(med_spellchecker_rubert_tiny2_test_missing_space_handling,
                                  "medspellchecker_rubert_tiny2_fix.csv")
    perform_anamnesis_fixing_test(med_spellchecker_rubiobert_test_missing_space_handling,
                                  "medspellchecker_rubiobert_fix.csv")
    perform_anamnesis_fixing_test(med_spellchecker_rubiorobert_test_missing_space_handling,
                                  "medspellchecker_rubiorobert_fix.csv")
