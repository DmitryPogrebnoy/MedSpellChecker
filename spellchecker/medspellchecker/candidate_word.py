from dataclasses import dataclass
from typing import final


@final
@dataclass
class CandidateWord:
    """Spelling suggestion.

        Args:
            value: The suggested word.
            distance: Edit distance from search word.
    """

    value: str
    distance: int
    # TODO: Add more features for ranking if needed
