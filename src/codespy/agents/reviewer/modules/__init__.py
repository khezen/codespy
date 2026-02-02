"""DSPy modules for code review."""

from codespy.agents.reviewer.modules.bugs import BugDetector
from codespy.agents.reviewer.modules.context import ContextAnalyzer
from codespy.agents.reviewer.modules.docs import DocumentationReviewer
from codespy.agents.reviewer.modules.security import SecurityAuditor

__all__ = [
    "SecurityAuditor",
    "BugDetector",
    "DocumentationReviewer",
    "ContextAnalyzer",
]