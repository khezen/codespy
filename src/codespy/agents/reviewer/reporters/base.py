"""Base reporter interface for review output."""

from abc import ABC, abstractmethod

from codespy.agents.reviewer.models import ReviewResult


class BaseReporter(ABC):
    """Abstract base class for review reporters."""

    @abstractmethod
    def report(self, result: ReviewResult) -> None:
        """Output the review result.

        Args:
            result: The review result to report.
        """
        pass