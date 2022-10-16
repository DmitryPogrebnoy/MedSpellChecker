import logging
from typing import final, List

import pandas as pd
from tabulate import tabulate


@final
class MetricTestWithoutContext:
    # First path to data for error precision test, second path to data for lexical precision test
    ERROR_TYPE_TO_DATA_PATH = {
        "Wrong_character": ('../../../../data/test/without_context/data_wrong_char_without_context.csv',
                            '../../../../data/test/without_context/lexical_precision_words.txt'),
        "Missing_character": ('../../../../data/test/without_context/data_missing_char_without_context.csv',
                              '../../../../data/test/without_context/lexical_precision_words.txt'),
        "Extra_character": ('../../../../data/test/without_context/data_extra_char_without_context.csv',
                            '../../../../data/test/without_context/lexical_precision_words.txt'),
        "Shuffled_character": ('../../../../data/test/without_context/data_shuffled_char_without_context.csv',
                               '../../../../data/test/without_context/lexical_precision_words.txt')}

    WORD_PER_SECOND_KEY: str = "words_per_second"
    ERROR_PRECISION_KEY: str = "error_precision"
    LEXICAL_PRECISION_KEY: str = "lexical_precision"
    OVERALL_PRECISION_KEY: str = "overall_precision"

    ERROR_TYPE_HEADER: str = "Error_Type"
    WORDS_PER_SECOND_HEADER: str = "Words_Per_Second"
    ERROR_PRECISION_HEADER: str = "Error_Precision"
    LEXICAL_PRECISION_HEADER: str = "Lexical_Precision"
    OVERALL_PRECISION_HEADER: str = "Overall_Precision"
    HEADER: List[str] = [ERROR_TYPE_HEADER, WORDS_PER_SECOND_HEADER, ERROR_PRECISION_HEADER,
                         LEXICAL_PRECISION_HEADER, OVERALL_PRECISION_HEADER]

    OVERALL_PRECISION_LABEL: str = "Overall"

    def __init__(self, additional_info=False):
        self.logger: logging.Logger = logging.getLogger()
        if additional_info:
            self.logger.setLevel(logging.INFO)

    def _compute_error_precision(self, original_word_list, corrected_word_list, answer_word_list):
        words_number = len(corrected_word_list)
        correct_words_number = 0
        self.logger.info("Error precision")
        self.logger.info("original_word_list --- corrected_word --- answer_word_list")
        for i, corrected_word in enumerate(corrected_word_list):
            if corrected_word in answer_word_list[i]:
                correct_words_number += 1
            else:
                self.logger.info(f"{original_word_list[i]} --- {corrected_word} --- {answer_word_list[i]}")
        self.logger.info(f"Right corrected words count - {correct_words_number} of {words_number} total")
        return correct_words_number / words_number

    def _compute_lexical_precision(self, original_word_list, corrected_word_list):
        words_number = len(corrected_word_list)
        correct_words_number = 0
        self.logger.info("Lexical precision")
        self.logger.info("original_word_list --- corrected_word")
        for i, corrected_word in enumerate(corrected_word_list):
            if corrected_word == original_word_list[i]:
                correct_words_number += 1
            else:
                self.logger.info(f"{original_word_list[i]} --- {corrected_word}")
        self.logger.info(f"Right corrected words count - {correct_words_number} of {words_number} total")
        return correct_words_number / words_number

    # TODO: Rebuild datasets for tests in simple way and rewrite this method
    def _load_data(self, error_path_to_data, lexical_path_to_data):
        # Load word data for error precision test
        # Format - {word}{several spaces}{answer-words separeted by comma}
        df = pd.read_csv(error_path_to_data)
        error_word_list = df.incorrect_word
        error_answer_list = df.correct_word

        # Load word list for lexical precision test
        with open(lexical_path_to_data) as file:
            lexical_word_list = [line.rstrip() for line in file.readlines()]

        return error_word_list, error_answer_list, lexical_word_list

    def compute_all_metrics(self, error_precision_spellchecker_function, lexical_precision_spellchecker_function):
        table = []

        for metric_name, path_to_data in self.ERROR_TYPE_TO_DATA_PATH.items():
            error_path_to_data, lexical_path_to_data = path_to_data
            # Load data
            error_word_list, error_answer_list, lexical_word_list = self._load_data(
                error_path_to_data, lexical_path_to_data)
            metric_result = self.compute_metric(error_word_list, error_answer_list, lexical_word_list,
                                                error_precision_spellchecker_function,
                                                lexical_precision_spellchecker_function)

            row = [metric_name,
                   metric_result[self.WORD_PER_SECOND_KEY],
                   metric_result[self.ERROR_PRECISION_KEY],
                   metric_result[self.LEXICAL_PRECISION_KEY],
                   metric_result[self.OVERALL_PRECISION_KEY]]
            table.append(row)

        df_table = pd.DataFrame(table, columns=self.HEADER)

        df_table.loc[len(df_table.index)] = [self.OVERALL_PRECISION_LABEL,
                                             df_table[self.WORDS_PER_SECOND_HEADER].mean(),
                                             df_table[self.ERROR_PRECISION_HEADER].mean(),
                                             df_table[self.LEXICAL_PRECISION_HEADER].mean(),
                                             df_table[self.OVERALL_PRECISION_HEADER].mean()]

        return tabulate(df_table, headers="keys", showindex=False)

    def compute_metric(self, error_word_list, error_answer_list, lexical_word_list,
                       error_precision_spellchecker_function, lexical_precision_spellchecker_function):
        error_elapsed_time, error_corrected_word_list = error_precision_spellchecker_function(error_word_list)
        error_precision = self._compute_error_precision(error_word_list,
                                                        error_corrected_word_list,
                                                        error_answer_list)

        lexical_elapsed_time, lexical_corrected_word_list = lexical_precision_spellchecker_function(lexical_word_list)
        lexical_precision = self._compute_lexical_precision(lexical_word_list,
                                                            lexical_corrected_word_list)

        return {self.WORD_PER_SECOND_KEY: (len(error_word_list) + len(lexical_word_list)) /
                                          (error_elapsed_time + lexical_elapsed_time),
                self.ERROR_PRECISION_KEY: error_precision,
                self.LEXICAL_PRECISION_KEY: lexical_precision,
                self.OVERALL_PRECISION_KEY: (error_precision + lexical_precision) / 2.0}
