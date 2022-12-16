from abc import ABC, abstractmethod
from enum import IntEnum

from editdistpy import damerau_osa, levenshtein


class EditDistanceAlgo(IntEnum):
    """Supported edit distance algorithms."""

    LEVENSHTEIN_FAST = 0  #: Fast Levenshtein algorithm.
    DAMERAU_OSA_FAST = 1  #: Fast Damerau optimal string alignment algorithm


class EditDistance:
    """Edit distance algorithms.

    Args:
        algorithm: The distance algorithm to use.

    Attributes:
        _algorithm (:class:`EditDistanceAlgo`): The edit distance algorithm to
            use.
        _distance_comparer (:class:`AbstractDistanceComparer`): An object to
            compute the relative distance between two strings. The concrete
            object will be chosen based on the value of :attr:`_algorithm`.

    Raises:
        ValueError: If `algorithm` specifies an invalid distance algorithm.
    """

    def __init__(self, algorithm: EditDistanceAlgo) -> None:
        self._distance_comparer: AbstractDistanceComparer
        self.algorithm = algorithm
        if algorithm == EditDistanceAlgo.LEVENSHTEIN_FAST:
            self._distance_comparer = LevenshteinFast()
        elif algorithm == EditDistanceAlgo.DAMERAU_OSA_FAST:
            self._distance_comparer = DamerauOsaFast()
        else:
            raise ValueError("Unknown distance algorithm!")

    def compare(self, string_1: str, string_2: str, max_distance: int) -> int:
        """Compares a string to the base string to determine the edit distance,
        using the previously selected algorithm.

        Args:
            string_1: Base string.
            string_2: The string to compare.
            max_distance: The maximum distance allowed.

        Returns:
            The edit distance (or -1 if `max_distance` exceeded).
        """
        return self._distance_comparer.distance(string_1, string_2, max_distance)


class AbstractDistanceComparer(ABC):
    """An interface to compute relative distance between two strings."""

    @abstractmethod
    def distance(self, string_1: str, string_2: str, max_distance: int) -> int:
        """Returns a measure of the distance between two strings.

        Args:
            string_1: One of the strings to compare.
            string_2: The other string to compare.
            max_distance: The maximum distance that is of interest.

        Returns:
            -1 if the distance is greater than the max_distance, 0 if the strings
                are equivalent, otherwise a positive number whose magnitude
                increases as difference between the strings increases.
        """


class LevenshteinFast(AbstractDistanceComparer):
    """Provides an interface for computing edit distance metric between two
    strings using the fast Levenshtein algorithm.
    """

    def distance(self, string_1: str, string_2: str, max_distance: int) -> int:
        """Computes the Levenshtein edit distance between two strings.

        Args:
            string_1: One of the strings to compare.
            string_2: The other string to compare.
            max_distance: The maximum distance that is of interest.

        Returns:
            -1 if the distance is greater than the max_distance, 0 if the strings
                are equivalent, otherwise a positive number whose magnitude
                increases as difference between the strings increases.
        """
        return levenshtein.distance(string_1, string_2, max_distance)


class DamerauOsaFast(AbstractDistanceComparer):
    """Provides an interface for computing edit distance metric between two
    strings using the fast Damerau-Levenshtein Optimal String Alignment (OSA)
    algorithm.
    """

    def distance(self, string_1: str, string_2: str, max_distance: int) -> int:
        """Computes the Damerau-Levenshtein optimal string alignment edit
        distance between two strings.

        Args:
            string_1: One of the strings to compare.
            string_2: The other string to compare.
            max_distance: The maximum distance that is of interest.

        Returns:
            -1 if the distance is greater than the max_distance, 0 if the strings
                are equivalent, otherwise a positive number whose magnitude
                increases as difference between the strings increases.
        """
        return damerau_osa.distance(string_1, string_2, max_distance)
