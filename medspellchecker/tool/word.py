from dataclasses import dataclass
from typing import final, List, Optional

from pymorphy2.analyzer import Parse
from pymorphy2.tagset import OpencorporaTag


@final
@dataclass
class Word:
    """Input word representation.

        Args:
            source_value: The original word.
            should_correct: Should this word be corrected?
            lemma_normal_form: The normal form of the word
            lemma_tag: Tag line of the original word form
            corrected_value: Final corrected word
    """
    position: int
    original_value: str
    should_correct: bool
    lemma_normal_form: Optional[str] = None
    lemma_tag: Optional[OpencorporaTag] = None
    lexeme: Optional[List[Parse]] = None
    corrected_value: Optional[str] = None

    def get_lemma_normal_or_original_form(self):
        if self.lemma_normal_form:
            return self.lemma_normal_form
        else:
            return self.original_value
