import logging
from typing import final, List, Dict

import pandas as pd
from tabulate import tabulate

from pre_post_processor import PreProcessor


@final
class MetricTestWithContext:

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

    def compute_all_metrics(self, error_type_to_data_path, spellchecker_function):
        table = []

        for metric_name, path_to_data in error_type_to_data_path.items():
            # Load data
            dataframe = pd.read_csv(path_to_data)
            dataframe.reset_index(drop=True, inplace=True)
            original_data = [batch.split() for batch in dataframe.original]
            answer_data = [batch.split() for batch in dataframe.answer]
            incorrect_word_pos_data = [int(pos) for pos in dataframe.pos_incorrect_word]

            metric_result = self.compute_metric(original_data, answer_data,
                                                incorrect_word_pos_data, spellchecker_function)

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

    def compute_metric(self, original_data: List[List[str]], answer_data: List[List[str]],
                       incorrect_word_pos_data: List[int], spellchecker_function) -> Dict[str, float]:
        """
        Compute metric for one of these mistakes - extra char, wrong char, missing char, shuffled char

        :param original_data: original data in format [word, word, word, ...], [word, word, word, ...], ...]
        :param answer_data: answer data in format [word, word, word, ...], [word, word, word, ...], ...]
        :param incorrect_word_pos_data: list with numbers of incorrect words in format [int, int, int, ...]
        :param spellchecker_function: function to correct original data,
         should return two value (elapsed_time, corrected_batches)
        :return: (word_per_second, error_precision, lexical_precision, overall_precision)
        """

        elapsed_time, result_batches = spellchecker_function(original_data)

        total_count_words = 0
        lexical_precision_correct_words_count = 0
        lexical_precision_incorrect_words_count = 0
        error_precision_correct_words_count = 0
        error_precision_incorrect_words_count = 0

        # Don't work with compound words!
        self.logger.info("Incorrect words!")
        self.logger.info(f"Batch id --- Corrected --- Answer")
        for i, corrected_batch in enumerate(result_batches):
            self.logger.info(f"Batch {i}")
            self.logger.info(f"Original {answer_data[i]}")
            self.logger.info(f"Corrected {corrected_batch}")
            for j, corrected_word in enumerate(corrected_batch):
                total_count_words += 1
                if incorrect_word_pos_data[i] != j:
                    # For test tools which returns lemmatized and non lemmatized words
                    if answer_data[i][j] == corrected_word or \
                            (self._ignore_lemmatization and
                             answer_data[i][j] == self._pre_processor.lemmatize(corrected_word)):
                        lexical_precision_correct_words_count += 1
                    else:
                        self.logger.info(f"{i} --- {corrected_word} --- {answer_data[i][j]} --- LEXICAL")
                        lexical_precision_incorrect_words_count += 1
                else:
                    # For test tools which returns lemmatized and non lemmatized words
                    if answer_data[i][j] == corrected_word or \
                            (self._ignore_lemmatization and
                             answer_data[i][j] == self._pre_processor.lemmatize(corrected_word)):
                        error_precision_correct_words_count += 1
                    else:
                        self.logger.info(f"{i} --- {corrected_word} --- {answer_data[i][j]} --- ERROR")
                        error_precision_incorrect_words_count += 1

        error_precision = error_precision_correct_words_count / \
                          (error_precision_correct_words_count + error_precision_incorrect_words_count)
        lexical_precision = lexical_precision_correct_words_count / \
                            (lexical_precision_correct_words_count + lexical_precision_incorrect_words_count)

        return {self.WORD_PER_SECOND_KEY: total_count_words / elapsed_time,
                self.ERROR_PRECISION_KEY: error_precision,
                self.LEXICAL_PRECISION_KEY: lexical_precision,
                self.OVERALL_PRECISION_KEY: (error_precision + lexical_precision) / 2.0}
