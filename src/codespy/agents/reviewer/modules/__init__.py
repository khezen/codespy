"""DSPy modules for code review."""

from codespy.agents.reviewer.modules.code_and_doc_reviewer import CodeAndDocReviewer
from codespy.agents.reviewer.modules.deduplicator import IssueDeduplicator
from codespy.agents.reviewer.modules.domain_expert import DomainExpert
from codespy.agents.reviewer.modules.scope_identifier import ScopeIdentifier
from codespy.agents.reviewer.modules.supply_chain_auditor import SupplyChainAuditor

__all__ = [
    "CodeAndDocReviewer",
    "DomainExpert",
    "IssueDeduplicator",
    "ScopeIdentifier",
    "SupplyChainAuditor",
]