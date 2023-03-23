from language_tool_python import LanguageTool
from tqdm import tqdm

from medspellchecker.tool.pre_post_processor import PreProcessor
from medspellchecker.tool_tests.anamnesis_fixing_test.common.anamnesis_fixing_test import perform_anamnesis_fixing_test


def language_tool_test(input_sentence):
    tool = LanguageTool('ru')
    pre_processor = PreProcessor()

    result = []
    timer = tqdm(input_sentence)
    for sentence in timer:
        corrected_sentence = []
        for word in sentence:
            if pre_processor.is_valid_token(word):
                fixed_word = tool.correct(word)
                corrected_sentence.append(fixed_word)
            else:
                corrected_sentence.append(word)
        result.append(" ".join(corrected_sentence))
    return result


if __name__ == '__main__':
    """
    Run test for Language Tool (language-tool-python package)
    
    Firstly you need to install Java SDK (sudo apt-get install openjdk-17-jdk)
    """
    perform_anamnesis_fixing_test(language_tool_test, "language_tool_fix.csv")
