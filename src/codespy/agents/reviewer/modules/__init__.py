"""DSPy modules for code review."""

from codespy.agents.reviewer.modules.auditor import Auditor
from codespy.agents.reviewer.modules.code_reviewer import CodeReviewer
from codespy.agents.reviewer.modules.doc_reviewer import DocReviewer
from codespy.agents.reviewer.modules.scope_resolver import ScopeResolver
from codespy.agents.reviewer.modules.summarizer import Summarizer
from codespy.agents.reviewer.modules.supply_chain_auditor import SupplyChainAuditor

__all__ = [
    "Auditor",
    "CodeReviewer",
    "DocReviewer",
    "ScopeResolver",
    "Summarizer",
    "SupplyChainAuditor",
]
