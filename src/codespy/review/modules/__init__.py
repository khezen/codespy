"""DSPy modules for code review."""

from codespy.review.modules.bugs import BugDetector
from codespy.review.modules.context import ContextAnalyzer
from codespy.review.modules.docs import DocumentationReviewer
from codespy.review.modules.security import SecurityAnalyzer

__all__ = [
    "SecurityAnalyzer",
    "BugDetector",
    "DocumentationReviewer",
    "ContextAnalyzer",
]
