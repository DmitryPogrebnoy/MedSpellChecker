from tqdm import tqdm

from common.metric_test_with_context import MetricTestWithContext
from other_spellcheckers.utils import ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT
from spellchecker_prototype_v2.spell_checker import SpellChecker


def spellchecker_prototype_v2_test(input_sentences):
    spellchecker_prototype = SpellChecker()
    timer = tqdm(input_sentences)
    result = []
    for sentence in timer:
        corrected_sentence = spellchecker_prototype.correct_words(" ".join(sentence))
        result.append(corrected_sentence)
    return timer.format_dict["elapsed"], result


def perform_test():
    metric_test_with_context = MetricTestWithContext()
    return metric_test_with_context.compute_all_metrics(ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
                                                        spellchecker_prototype_v2_test)


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
