"""DSPy modules for code review."""

from codespy.agents.reviewer.modules.bug_detector import BugDetector
from codespy.agents.reviewer.modules.deduplicator import IssueDeduplicator
from codespy.agents.reviewer.modules.doc_reviewer import DocReviewer
from codespy.agents.reviewer.modules.scope_identifier import ScopeIdentifier
from codespy.agents.reviewer.modules.smell_detector import SmellDetector
from codespy.agents.reviewer.modules.supply_chain_auditor import SupplyChainAuditor

__all__ = [
    "BugDetector",
    "DocReviewer",
    "IssueDeduplicator",
    "ScopeIdentifier",
    "SmellDetector",
    "SupplyChainAuditor",
]