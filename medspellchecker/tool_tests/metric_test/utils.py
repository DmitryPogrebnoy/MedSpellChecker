import os


def build_path_relative_to_current_file(relative_path: str) -> str:
    return os.path.join(os.path.dirname(__file__), relative_path)


SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT = {
    "Wrong_character": build_path_relative_to_current_file(
        '../../../data/test/with_context/data_wrong_char_with_context.csv'),
    "Missing_character": build_path_relative_to_current_file(
        '../../../data/test/with_context/data_missing_char_with_context.csv'),
    "Extra_character": build_path_relative_to_current_file(
        '../../../data/test/with_context/data_extra_char_with_context.csv'),
    "Shuffled_character": build_path_relative_to_current_file(
        '../../../data/test/with_context/data_shuffled_char_with_context.csv')
}

MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT = (
    "Missing_separator", build_path_relative_to_current_file(
        '../../../data/test/with_context/data_missing_space_with_context.csv'))

EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT = (
    "Extra separator", build_path_relative_to_current_file(
        '../../../data/test/with_context/data_extra_space_with_context.csv'))

ERROR_TYPE_TO_DATA_PATH_WITHOUT_CONTEXT = {
    "Wrong_character": build_path_relative_to_current_file(
        '../../../data/test/without_context/data_wrong_char_without_context.csv'),
    "Missing_character": build_path_relative_to_current_file(
        '../../../data/test/without_context/data_missing_char_without_context.csv'),
    "Extra_character": build_path_relative_to_current_file(
        '../../../data/test/without_context/data_extra_char_without_context.csv'),
    "Shuffled_character": build_path_relative_to_current_file(
        '../../../data/test/without_context/data_shuffled_char_without_context.csv'),
    "Missing separator": build_path_relative_to_current_file(
        '../../../data/test/without_context/data_missing_space_without_context.csv'),
    "Extra separator": build_path_relative_to_current_file(
        '../../../data/test/without_context/data_extra_space_without_context.csv')
}
