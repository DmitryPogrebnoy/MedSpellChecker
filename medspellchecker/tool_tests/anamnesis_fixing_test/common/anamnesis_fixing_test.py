import os

import pandas as pd
from sacremoses import MosesTokenizer


def perform_anamnesis_fixing_test(test_function, relative_path_to_save_result):
    path_to_test_dataset = os.path.join(os.path.dirname(__file__),
                                        "../../../../data/test/anamnesis_fixing_test/test_ru_med_prime_data.csv")
    df = pd.read_csv(path_to_test_dataset)
    anamnesis = df["data"].values[:100]
    tokenizer = MosesTokenizer(lang="ru")
    input_sequences = [tokenizer.tokenize(anamnesis) for anamnesis in anamnesis]
    result = test_function(input_sequences)
    answer_df = pd.DataFrame(result, columns=["data"])
    absolute_path_to_save_result = os.path.join(os.path.dirname(__file__),
                                                "../../../../data/test/anamnesis_fixing_test/after_fix/",
                                                relative_path_to_save_result)
    answer_df.to_csv(absolute_path_to_save_result)
    print(f"Result saved to {absolute_path_to_save_result}")
