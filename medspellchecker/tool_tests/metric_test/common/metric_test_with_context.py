import logging
from difflib import SequenceMatcher
from typing import final, List, Dict

import pandas as pd
from tabulate import tabulate

from medspellchecker.tool.pre_post_processor import PreProcessor


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

    def compute_all_metrics(self, simple_test_data_path,
                            missing_space_test_data_path,
                            extra_space_test_data_path, spellchecker_function):
        table = []

        for metric_name, path_to_data in simple_test_data_path.items():
            # Load data
            dataframe = pd.read_csv(path_to_data)
            original_data = [batch.split() for batch in dataframe.original]
            answer_data = [batch.split() for batch in dataframe.answer]
            incorrect_word_pos_data = [int(pos) for pos in dataframe.pos_incorrect_word]

            metric_result = self.compute_simple_metric(original_data, answer_data,
                                                       incorrect_word_pos_data, spellchecker_function)
            row = [metric_name,
                   metric_result[self.WORD_PER_SECOND_KEY],
                   metric_result[self.ERROR_PRECISION_KEY],
                   metric_result[self.LEXICAL_PRECISION_KEY],
                   metric_result[self.OVERALL_PRECISION_KEY]]
            table.append(row)

        # Compute metric for missing space error
        missing_space_metric_name, missing_space_path_to_data = missing_space_test_data_path
        missing_space_df = pd.read_csv(missing_space_path_to_data)
        original_data = [batch.split() for batch in missing_space_df.original]
        answer_data = [batch.split() for batch in missing_space_df.answer]
        incorrect_word_pos_data = [int(pos) for pos in missing_space_df.pos]

        missing_space_metric_result = self.compute_missing_space_metric(original_data, answer_data,
                                                                        incorrect_word_pos_data, spellchecker_function)
        row = [missing_space_metric_name,
               missing_space_metric_result[self.WORD_PER_SECOND_KEY],
               missing_space_metric_result[self.ERROR_PRECISION_KEY],
               missing_space_metric_result[self.LEXICAL_PRECISION_KEY],
               missing_space_metric_result[self.OVERALL_PRECISION_KEY]]
        table.append(row)

        # Compute metric for extra space error
        extra_space_metric_name, extra_space_path_to_data = extra_space_test_data_path
        extra_space_df = pd.read_csv(extra_space_path_to_data)
        original_data = [batch.split() for batch in extra_space_df.original]
        answer_data = [batch.split() for batch in extra_space_df.answer]
        incorrect_word_pos_data = [int(pos) for pos in extra_space_df.pos]

        extra_space_metric_result = self.compute_extra_space_metric(original_data, answer_data,
                                                                    incorrect_word_pos_data, spellchecker_function)
        row = [extra_space_metric_name,
               extra_space_metric_result[self.WORD_PER_SECOND_KEY],
               extra_space_metric_result[self.ERROR_PRECISION_KEY],
               extra_space_metric_result[self.LEXICAL_PRECISION_KEY],
               extra_space_metric_result[self.OVERALL_PRECISION_KEY]]
        table.append(row)

        df_table = pd.DataFrame(table, columns=self.HEADER)

        df_table.loc[len(df_table.index)] = [self.OVERALL_PRECISION_LABEL,
                                             df_table[self.WORDS_PER_SECOND_HEADER].mean(),
                                             df_table[self.ERROR_PRECISION_HEADER].mean(),
                                             df_table[self.LEXICAL_PRECISION_HEADER].mean(),
                                             df_table[self.OVERALL_PRECISION_HEADER].mean()]

        return tabulate(df_table, headers="keys", showindex=False)

    def compute_simple_metric(self, original_data: List[List[str]], answer_data: List[List[str]],
                              incorrect_word_pos_data: List[int], spellchecker_function) -> Dict[str, float]:
        """
            Compute metric for missing space mistake

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

        for i, corrected_batch in enumerate(result_batches):
            answer = answer_data[i]
            corrected_batch = " ".join(corrected_batch).split()
            if self._ignore_lemmatization:
                corrected_batch = [self._pre_processor.lemmatize(word) for word in corrected_batch]
            total_count_words += len(answer)
            incorrect_pos = incorrect_word_pos_data[i]

            opcodes = SequenceMatcher(None, answer, corrected_batch).get_opcodes()
            for tag, answer_idx_1, answer_idx_2, corrected_inx_1, corrected_inx_2 in opcodes:
                if tag == "equal":
                    if incorrect_pos in range(answer_idx_1, answer_idx_2):
                        error_precision_correct_words_count += 1
                        lexical_precision_correct_words_count += answer_idx_2 - answer_idx_1 - 1
                        continue

                    lexical_precision_correct_words_count += answer_idx_2 - answer_idx_1
                else:
                    if incorrect_pos in range(answer_idx_1, answer_idx_2):
                        error_precision_incorrect_words_count += 1
                        lexical_precision_incorrect_words_count += answer_idx_2 - answer_idx_1 - 1
                        continue

                    lexical_precision_incorrect_words_count += answer_idx_2 - answer_idx_1

        error_precision = error_precision_correct_words_count / \
                          (error_precision_correct_words_count + error_precision_incorrect_words_count)
        lexical_precision = lexical_precision_correct_words_count / \
                            (lexical_precision_correct_words_count + lexical_precision_incorrect_words_count)

        return {self.WORD_PER_SECOND_KEY: total_count_words / elapsed_time,
                self.ERROR_PRECISION_KEY: error_precision,
                self.LEXICAL_PRECISION_KEY: lexical_precision,
                self.OVERALL_PRECISION_KEY: (error_precision + lexical_precision) / 2.0}

    def compute_missing_space_metric(self, original_data: List[List[str]], answer_data: List[List[str]],
                                     incorrect_word_pos_data: List[int], spellchecker_function) -> Dict[str, float]:
        """
            Compute metric for missing space mistake

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

        for i, corrected_batch in enumerate(result_batches):
            answer = answer_data[i]
            corrected_batch = " ".join(corrected_batch).split()
            if self._ignore_lemmatization:
                corrected_batch = [self._pre_processor.lemmatize(word) for word in corrected_batch]
            total_count_words += len(answer)
            incorrect_pos = incorrect_word_pos_data[i]

            opcodes = SequenceMatcher(None, answer, corrected_batch).get_opcodes()
            for tag, answer_idx_1, answer_idx_2, corrected_inx_1, corrected_inx_2 in opcodes:
                if tag == "equal":
                    if incorrect_pos in range(answer_idx_1, answer_idx_2) and \
                            incorrect_pos + 1 in range(answer_idx_1, answer_idx_2):
                        error_precision_correct_words_count += 2
                        lexical_precision_correct_words_count += answer_idx_2 - answer_idx_1 - 2
                        continue
                    if incorrect_pos in range(answer_idx_1, answer_idx_2) or \
                            incorrect_pos + 1 in range(answer_idx_1, answer_idx_2):
                        error_precision_correct_words_count += 1
                        lexical_precision_correct_words_count += answer_idx_2 - answer_idx_1 - 1
                        continue

                    lexical_precision_correct_words_count += answer_idx_2 - answer_idx_1
                else:
                    if incorrect_pos in range(answer_idx_1, answer_idx_2) and \
                            incorrect_pos + 1 in range(answer_idx_1, answer_idx_2):
                        error_precision_incorrect_words_count += 2
                        lexical_precision_incorrect_words_count += answer_idx_2 - answer_idx_1 - 2
                        continue
                    if incorrect_pos in range(answer_idx_1, answer_idx_2) or \
                            incorrect_pos + 1 in range(answer_idx_1, answer_idx_2):
                        error_precision_incorrect_words_count += 1
                        lexical_precision_incorrect_words_count += answer_idx_2 - answer_idx_1 - 1
                        continue

                    lexical_precision_incorrect_words_count += answer_idx_2 - answer_idx_1

        error_precision = error_precision_correct_words_count / \
                          (error_precision_correct_words_count + error_precision_incorrect_words_count)
        lexical_precision = lexical_precision_correct_words_count / \
                            (lexical_precision_correct_words_count + lexical_precision_incorrect_words_count)

        return {self.WORD_PER_SECOND_KEY: total_count_words / elapsed_time,
                self.ERROR_PRECISION_KEY: error_precision,
                self.LEXICAL_PRECISION_KEY: lexical_precision,
                self.OVERALL_PRECISION_KEY: (error_precision + lexical_precision) / 2.0}

    def compute_extra_space_metric(self, original_data: List[List[str]], answer_data: List[List[str]],
                                   incorrect_word_pos_data: List[int], spellchecker_function) -> Dict[str, float]:
        """
            Compute metric for missing space mistake

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

        for i, corrected_batch in enumerate(result_batches):
            answer = answer_data[i]
            corrected_batch = " ".join(corrected_batch).split()
            if self._ignore_lemmatization:
                corrected_batch = [self._pre_processor.lemmatize(word) for word in corrected_batch]
            total_count_words += len(answer)
            incorrect_pos = incorrect_word_pos_data[i]

            opcodes = SequenceMatcher(None, answer, corrected_batch).get_opcodes()
            for tag, answer_idx_1, answer_idx_2, corrected_inx_1, corrected_inx_2 in opcodes:
                if tag == "equal":
                    if incorrect_pos in range(answer_idx_1, answer_idx_2):
                        error_precision_correct_words_count += 1
                        lexical_precision_correct_words_count += answer_idx_2 - answer_idx_1 - 1
                        continue

                    lexical_precision_correct_words_count += answer_idx_2 - answer_idx_1
                else:
                    if incorrect_pos in range(answer_idx_1, answer_idx_2):
                        error_precision_incorrect_words_count += 1
                        lexical_precision_incorrect_words_count += answer_idx_2 - answer_idx_1 - 1
                        continue

                    lexical_precision_incorrect_words_count += answer_idx_2 - answer_idx_1

        error_precision = error_precision_correct_words_count / \
                          (error_precision_correct_words_count + error_precision_incorrect_words_count)
        lexical_precision = lexical_precision_correct_words_count / \
                            (lexical_precision_correct_words_count + lexical_precision_incorrect_words_count)

        return {self.WORD_PER_SECOND_KEY: total_count_words / elapsed_time,
                self.ERROR_PRECISION_KEY: error_precision,
                self.LEXICAL_PRECISION_KEY: lexical_precision,
                self.OVERALL_PRECISION_KEY: (error_precision + lexical_precision) / 2.0}
