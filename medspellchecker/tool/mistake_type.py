from enum import Enum
from typing import final


@final
class MistakeType(Enum):
    SINGLE_WORD_MISTAKE = 1,
    MISSING_SPACE_MISTAKE = 2,
    EXTRA_SPACE_MISTAKE = 3
