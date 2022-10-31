import json

import requests

from common.metric_test_with_context import MetricTestWithContext
from other_spellcheckers.utils import SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, \
    MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT

YANDEX_SPELLER_URL = 'https://speller.yandex.net/services/spellservice.json/checkTexts'
MAX_CHARS_PER_REQUEST = 10_000
LANG_PARAM_NAME = "lang"
LANG_PARAM_VALUE = "ru"
FORMAT_PARAM_NAME = "format"
FORMAT_PARAM_VALUE = "plain"
OPTIONS_PARAM_NAME = "options"
OPTIONS_PARAM_VALUE = 518
TEXT_PARAM_NAME = "text"


def split_sentence_list_to_batches(sentence_list):
    if len(sentence_list) == 0:
        return []

    list_of_batches = []
    prev_batches_counter = 0
    sentence_counter = 0
    current_char_length = 0

    while len(sentence_list) != sentence_counter:
        sentence_length = " ".join(sentence_list[sentence_counter])
        if len(sentence_length) >= MAX_CHARS_PER_REQUEST:
            print(f"Batch with index {sentence_counter} is too big. Replaced with space.")
            sentence_list[sentence_counter] = " "
            current_char_length += 1

        if len(sentence_length) + current_char_length < MAX_CHARS_PER_REQUEST:
            current_char_length += len(sentence_length) + 1
            sentence_counter += 1
        else:
            list_of_batches.append(sentence_list[prev_batches_counter:sentence_counter])
            prev_batches_counter = sentence_counter
            current_char_length = 0

    list_of_batches.append(sentence_list[prev_batches_counter:sentence_counter])

    return list_of_batches


def yandex_speller_tool_test(input_batches):
    batches = split_sentence_list_to_batches(input_batches)

    result = []
    timer = 0.0
    for batch in batches:
        params = {LANG_PARAM_NAME: LANG_PARAM_VALUE,
                  FORMAT_PARAM_NAME: FORMAT_PARAM_VALUE,
                  OPTIONS_PARAM_NAME: OPTIONS_PARAM_VALUE,
                  TEXT_PARAM_NAME: ["+".join(sentence) for sentence in batch]}
        response = requests.post(YANDEX_SPELLER_URL, data=params)
        timer += response.elapsed.total_seconds()
        response.raise_for_status()
        corrected_batches = json.loads(response.text)

        for idx_sentence, sentence in enumerate(batch):
            sentence_suggestion = corrected_batches[idx_sentence]
            suggestion_count = 0
            corrected_sentence = []
            for word in sentence:
                if len(sentence_suggestion) > suggestion_count and \
                        sentence_suggestion[suggestion_count]["word"] == word:
                    corrected_sentence.append(sentence_suggestion[suggestion_count]["s"][0])
                    suggestion_count += 1
                else:
                    corrected_sentence.append(word)
            result.append(corrected_sentence)

    return timer, result


def perform_test():
    metric_test_with_context = MetricTestWithContext()
    return metric_test_with_context.compute_all_metrics(
        SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        yandex_speller_tool_test)


if __name__ == '__main__':
    """
    Run test with context for Yandex Speller
    Link: https://yandex.ru/dev/speller/doc/dg/concepts/api-overview.html
    """
    test_result = perform_test()
    print(test_result)
