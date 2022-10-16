import json

import requests

from common.metric_test_without_context import MetricTestWithoutContext
from other_spellcheckers.utils import ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT

YANDEX_SPELLER_URL = 'https://speller.yandex.net/services/spellservice.json/checkTexts'
MAX_CHARS_PER_REQUEST = 10_000
LANG_PARAM_NAME = "lang"
LANG_PARAM_VALUE = "ru"
FORMAT_PARAM_NAME = "format"
FORMAT_PARAM_VALUE = "plain"
OPTIONS_PARAM_NAME = "options"
OPTIONS_PARAM_VALUE = 518
TEXT_PARAM_NAME = "text"


def split_word_list_to_batches(word_list):
    if len(word_list) == 0:
        return []

    list_of_batches = []
    prev_word_counter = 0
    word_counter = 0
    current_char_length = 0

    while len(word_list) != word_counter:
        if len(word_list[word_counter]) >= MAX_CHARS_PER_REQUEST:
            print(f"Word with index {word_counter} is too big. Replaced with space.")
            word_list[word_counter] = " "
            current_char_length += 1

        if len(word_list[word_counter]) + current_char_length < MAX_CHARS_PER_REQUEST:
            current_char_length += len(word_list[word_counter])
            word_counter += 1
        else:
            list_of_batches.append(word_list[prev_word_counter:word_counter])
            prev_word_counter = word_counter
            current_char_length = 0

    list_of_batches.append(word_list[prev_word_counter:word_counter])

    return list_of_batches


def yandex_speller_tool_test(input_word_list):
    batches = split_word_list_to_batches(input_word_list)

    result = []
    timer = 0.0
    for batch in batches:
        params = {LANG_PARAM_NAME: LANG_PARAM_VALUE,
                  FORMAT_PARAM_NAME: FORMAT_PARAM_VALUE,
                  OPTIONS_PARAM_NAME: OPTIONS_PARAM_VALUE,
                  TEXT_PARAM_NAME: batch}
        response = requests.post(YANDEX_SPELLER_URL, data=params)
        timer += response.elapsed.total_seconds()
        response.raise_for_status()
        corrected_words = json.loads(response.text)

        for idx, word in enumerate(batch):
            if corrected_words[idx]:
                result.append(corrected_words[idx][0]['s'][0])
            else:
                result.append(word)

    return {"elapsed": timer, "corrected_word_list": result}


def perform_test():
    metric_test_without_context = MetricTestWithoutContext()
    return metric_test_without_context.compute_all_metrics(
        ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT, yandex_speller_tool_test, yandex_speller_tool_test)


if __name__ == '__main__':
    """
    Run test without context for Yandex Speller
    Link: https://yandex.ru/dev/speller/doc/dg/concepts/api-overview.html
    """
    test_result = perform_test()
    print(test_result)
