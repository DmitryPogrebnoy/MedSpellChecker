from tqdm import tqdm

from common.metric_test_without_context import MetricTestWithoutContext
from other_spellcheckers.utils import ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT
from spellchecker_prototype_v2.spell_checker import SpellChecker


def spellchecker_prototype_v2_test(input_word_list):
    spellchecker_prototype = SpellChecker()
    word_list = " ".join(input_word_list)
    timer = tqdm()
    corrected_word_list = spellchecker_prototype.correct_words(word_list)
    return {"elapsed": timer.format_dict["elapsed"], "corrected_word_list": corrected_word_list}


def perform_test():
    metric_test_without_context = MetricTestWithoutContext()
    return metric_test_without_context.compute_all_metrics(
        ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT,
        spellchecker_prototype_v2_test, spellchecker_prototype_v2_test)


if __name__ == '__main__':
    """
    Run test without context for spellchecker prototype v2 from this article https://arxiv.org/abs/2004.04987
    
    For run this test download models (all files) from link 
    (https://drive.google.com/drive/folders/1ubVgiIC2pqDOpA-CbjEqV8Jo8uBbk0qi?usp=sharing) 
    and move it to data/other_spellcheckers/spellchecker_prototype_v2/models folder
    
    Also need to install python-levenshtein and stringdist packages (pip3 install python-levenshtein stringdist)
    """
    test_result = perform_test()
    print(test_result)
