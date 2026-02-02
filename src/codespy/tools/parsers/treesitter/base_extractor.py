"""Base extractor class with shared helper methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codespy.tools.parsers.treesitter.models import FunctionInfo

if TYPE_CHECKING:
    from tree_sitter import Node


class BaseExtractor(ABC):
    """Base class for language-specific function extractors."""

    @abstractmethod
    def extract_functions(
        self,
        node: Any,  # Node type from tree-sitter
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract function definitions from an AST node.

        Args:
            node: Root AST node
            file_path: Path to the source file
            source: Raw source bytes

        Returns:
            List of extracted function definitions
        """
        ...

    def _extract_generic_params(self, node: Any, source: bytes) -> list[str]:
        """Extract parameters from a generic function node.

        Args:
            node: Function AST node
            source: Raw source bytes

        Returns:
            List of parameter strings
        """
        params: list[str] = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for child in params_node.children:
                # Skip punctuation
                if child.type not in ("(", ")", ","):
                    param_text = source[child.start_byte:child.end_byte].decode()
                    params.append(param_text.strip())
        return params

    def _get_node_text(self, node: Any, source: bytes) -> str:
        """Get the text content of a node.

        Args:
            node: AST node
            source: Raw source bytes

        Returns:
            Decoded text content
        """
        return source[node.start_byte:node.end_byte].decode()