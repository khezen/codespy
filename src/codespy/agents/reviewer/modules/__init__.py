"""DSPy modules for code review."""

from codespy.agents.reviewer.modules.defect_detector import DefectDetector
from codespy.agents.reviewer.modules.deduplicator import IssueDeduplicator
from codespy.agents.reviewer.modules.domain_expert import DomainExpert
from codespy.agents.reviewer.modules.doc_reviewer import DocumentationReviewer
from codespy.agents.reviewer.modules.scope_identifier import ScopeIdentifier
from codespy.agents.reviewer.modules.supply_chain_auditor import SupplyChainAuditor

__all__ = [
    "SupplyChainAuditor",
    "DefectDetector",
    "DocumentationReviewer",
    "DomainExpert",
    "IssueDeduplicator",
    "ScopeIdentifier",
]