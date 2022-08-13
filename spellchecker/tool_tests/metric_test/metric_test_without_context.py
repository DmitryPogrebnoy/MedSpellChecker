from typing import final

import pandas as pd


@final
class MetricTestWithoutContext:
    def __init__(self, path_to_error_precision_data, path_to_lexical_precision_data):
        # Load word data for error precision test
        # Format - {word}{several spaces}{answer-words separeted by comma}
        with open(path_to_error_precision_data) as file:
            error_precision_test_incorrect_lines = file.readlines()
        error_precision_splitted_test_incorrect_lines = list(map(lambda x: x.split(maxsplit=1),
                                                                 error_precision_test_incorrect_lines))
        error_precision_test_incorrect_words = list(map(lambda x: x[0], error_precision_splitted_test_incorrect_lines))
        error_precision_test_incorrect_answers = list(map(lambda x: list(map(lambda y: y.strip(), x[1].split(', '))),
                                                          error_precision_splitted_test_incorrect_lines))
        error_precision_test_df = pd.DataFrame(error_precision_test_incorrect_words, columns=["test_word"])
        error_precision_test_df["answers"] = error_precision_test_incorrect_answers
        self._error_precision_test_word_list = error_precision_test_df["test_word"].tolist()
        self._error_precision_test_answers = error_precision_test_df["answers"].tolist()
        # Load word list for lexical precision test
        with open(path_to_lexical_precision_data) as file:
            lexical_precision_test_word_list = file.readlines()
        self.lexical_precision_test_word_list = [line.rstrip() for line in lexical_precision_test_word_list]

    def _compute_error_precision(self, original_word_list, corrected_word_list, answer_word_list):
        words_number = len(corrected_word_list)
        correct_words_number = 0
        print("Error precision")
        print("original_word_list --- corrected_word --- answer_word_list")
        for i, corrected_word in enumerate(corrected_word_list):
            if corrected_word in answer_word_list[i]:
                correct_words_number += 1
            else:
                print(f"{original_word_list[i]} --- {corrected_word} --- {answer_word_list[i]}")
        print(f"Right corrected words count - {correct_words_number} of {words_number} total")
        return correct_words_number / words_number

    def _compute_lexical_precision(self, original_word_list, corrected_word_list):
        words_number = len(corrected_word_list)
        correct_words_number = 0
        print("Lexical precision")
        print("original_word_list --- corrected_word")
        for i, corrected_word in enumerate(corrected_word_list):
            if corrected_word == original_word_list[i]:
                correct_words_number += 1
            else:
                print(f"{original_word_list[i]} --- {corrected_word}")
        print(f"Right corrected words count - {correct_words_number} of {words_number} total")
        return correct_words_number / words_number

    def compute_all_metrics(self, error_precision_spellchecker_function,
                            lexical_precision_spellchecker_function):

        error_precision_result = error_precision_spellchecker_function(self._error_precision_test_word_list)
        error_precision_elapsed_time = error_precision_result["elapsed"]
        error_precision_corrected_word_list = error_precision_result["corrected_word_list"]
        error_precision = self._compute_error_precision(self._error_precision_test_word_list,
                                                        error_precision_corrected_word_list,
                                                        self._error_precision_test_answers)

        lexical_precision_result = lexical_precision_spellchecker_function(self.lexical_precision_test_word_list)
        lexical_precision_elapsed_time = lexical_precision_result["elapsed"]
        lexical_precision_corrected_word_list = lexical_precision_result["corrected_word_list"]
        lexical_precision = self._compute_lexical_precision(self.lexical_precision_test_word_list,
                                                            lexical_precision_corrected_word_list)

        return {"words_per_second": (len(self._error_precision_test_word_list) +
                                     len(self.lexical_precision_test_word_list)) /
                                    (error_precision_elapsed_time + lexical_precision_elapsed_time),
                "error_precision": error_precision,
                "lexical_precision": lexical_precision,
                "overall_precision": (error_precision + lexical_precision) / 2.0
                # "error_precision_original_word_list": error_precision_original_word_list,
                # "error_precision_corrected_word_list": error_precision_corrected_word_list,
                # "error_precision_answer_word_list": error_precision_answer_word_list,
                # "lexical_precision_original_word_list":lexical_precision_original_word_list
                }
