"""DSPy modules for code review."""

from codespy.agents.reviewer.modules.bug_detector import BugDetector
from codespy.agents.reviewer.modules.domain_expert import DomainExpert
from codespy.agents.reviewer.modules.doc_reviewer import DocumentationReviewer
from codespy.agents.reviewer.modules.scope_identifier import ScopeIdentifier
from codespy.agents.reviewer.modules.security_auditor import SecurityAuditor

__all__ = [
    "SecurityAuditor",
    "BugDetector",
    "DocumentationReviewer",
    "DomainExpert",
    "ScopeIdentifier",
]
