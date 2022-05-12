from typing import final

import pandas as pd


@final
class MetricTestWithContext:
    def __init__(self, path_to_data):
        # Load data
        dataframe = pd.read_csv(path_to_data)
        dataframe.reset_index(drop=True, inplace=True)
        self.original_batch = [batch.split() for batch in dataframe.original]
        self.answer_batch = [batch.split() for batch in dataframe.answer]
        self.pos_incorrect_word = [int(pos) for pos in dataframe.pos_incorrect_word]

    def compute_all_metrics(self, spellchecker_function):
        result = spellchecker_function(self.original_batch)
        elapsed_time = result["elapsed"]
        result_butches = result["corrected_batch"]

        total_count_words = 0
        lexical_precision_correct_words_count = 0
        lexical_precision_incorrect_words_count = 0
        error_precision_correct_words_count = 0
        error_precision_incorrect_words_count = 0

        # Don't work with compound words!
        print("Incorrect words!")
        print(f"Batch id --- Corrected --- Answer")
        for i, corrected_batch in enumerate(result_butches):
            print(f"Batch {i}")
            print(f"Original {self.answer_batch[i]}")
            print(f"Corrected {corrected_batch}")
            for j, corrected_word in enumerate(corrected_batch):
                total_count_words += 1
                if self.pos_incorrect_word[i] != j:
                    if self.answer_batch[i][j] == corrected_word:
                        lexical_precision_correct_words_count += 1
                    else:
                        print(f"{i} --- {corrected_word} --- {self.answer_batch[i][j]} --- LEXICAL")
                        lexical_precision_incorrect_words_count += 1
                else:
                    if self.answer_batch[i][j] == corrected_word:
                        error_precision_correct_words_count += 1
                    else:
                        print(f"{i} --- {corrected_word} --- {self.answer_batch[i][j]} --- ERROR")
                        error_precision_incorrect_words_count += 1

        error_precision = error_precision_correct_words_count / \
                          (error_precision_correct_words_count + error_precision_incorrect_words_count)
        lexical_precision = lexical_precision_correct_words_count / \
                            (lexical_precision_correct_words_count + lexical_precision_incorrect_words_count)

        return {"words_per_second": total_count_words / elapsed_time,
                "error_precision": error_precision,
                "lexical_precision": lexical_precision,
                "overall_precision": (error_precision + lexical_precision) / 2.0}
