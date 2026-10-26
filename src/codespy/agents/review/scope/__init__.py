from codespy.agents.review.scope.agent import (
    AI_DIRS,
    AI_FILES,
    MANIFEST_FILES,
    MANIFEST_GLOBS,
    ScopeResolver,
    append_sparse_patterns,
    apply_sparse_checkout,
    build_sparse_patterns,
    derive_sparse_paths,
)
from codespy.agents.review.scope.models import PackageManifest, ScopeResult, ScopeType

__all__ = [
    "MANIFEST_FILES",
    "MANIFEST_GLOBS",
    "AI_DIRS",
    "AI_FILES",
    "ScopeResolver",
    "derive_sparse_paths",
    "build_sparse_patterns",
    "append_sparse_patterns",
    "apply_sparse_checkout",
    "PackageManifest",
    "ScopeResult",
    "ScopeType",
]
