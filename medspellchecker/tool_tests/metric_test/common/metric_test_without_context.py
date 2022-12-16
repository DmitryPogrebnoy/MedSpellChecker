import logging
from typing import final, List

import pandas as pd
from tabulate import tabulate

from medspellchecker.tool.pre_post_processor import PreProcessor


@final
class MetricTestWithoutContext:
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

    def __init__(self, ignore_lemmatization=True, additional_info=False):
        self._ignore_lemmatization = ignore_lemmatization
        self._pre_processor = PreProcessor()
        self.logger: logging.Logger = logging.getLogger()
        if additional_info:
            self.logger.setLevel(logging.INFO)

    def _compute_error_precision(self, original_word_list, corrected_word_list, answer_word_list):
        words_number = len(corrected_word_list)
        correct_words_number = 0
        self.logger.info("Error precision")
        self.logger.info("original_word_list --- corrected_word --- answer_word_list")
        for i, corrected_word in enumerate(corrected_word_list):
            if corrected_word == answer_word_list[i]:
                correct_words_number += 1
                self.logger.info(f"CORRECT {original_word_list[i]} --- {corrected_word} --- {answer_word_list[i]}")
            else:
                self.logger.info(f"ERROR {original_word_list[i]} --- {corrected_word} --- {answer_word_list[i]}")
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
                self.logger.info(f"CORRECT {original_word_list[i]} --- {corrected_word}")
            else:
                self.logger.info(f"ERROR {original_word_list[i]} --- {corrected_word}")
        self.logger.info(f"Right corrected words count - {correct_words_number} of {words_number} total")
        return correct_words_number / words_number

    def _load_data(self, path_to_data):
        # Load word data for error precision test
        # Format - {word}{several spaces}{answer-words separeted by comma}
        df = pd.read_csv(path_to_data)
        error_word_list = df.incorrect_word
        error_answer_list = df.correct_word

        return error_word_list, error_answer_list, error_answer_list

    def compute_all_metrics(self, simple_test_data_path,
                            error_precision_spellchecker_function,
                            lexical_precision_spellchecker_function):
        table = []

        for metric_name, path_to_data in simple_test_data_path.items():
            # Load data
            error_word_list, error_answer_list, lexical_word_list = self._load_data(path_to_data)
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
