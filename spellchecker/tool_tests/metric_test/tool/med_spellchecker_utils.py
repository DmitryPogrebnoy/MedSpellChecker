from med_spellchecker import MedSpellchecker
from roberta_candidate_ranker import RuRobertaCandidateRanker


def create_med_spellchecker_roberta():
    return MedSpellchecker(
        words_list="../../../../data/dictionaries/processed/processed_lemmatized_all_dict.txt",
        encoding="UTF-8", candidate_ranker=RuRobertaCandidateRanker()
    )
