"""Bash/Shell function extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import FunctionInfo


class BashExtractor(BaseExtractor):
    """Extract function definitions from Bash/Shell source code."""

    def extract_functions(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract Bash function definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Any) -> None:
            if n.type == "function_definition":
                func_info = self._extract_function_info(n, file_path, source)
                if func_info:
                    functions.append(func_info)

            for child in n.children:
                visit(child)

        visit(node)
        return functions

    def _extract_function_info(
        self,
        func_node: Any,
        file_path: Path,
        source: bytes,
    ) -> FunctionInfo | None:
        """Extract function info from a function_definition node."""
        name_node = func_node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source)

        # Bash functions don't have typed parameters or return types
        # Parameters are accessed via $1, $2, etc.
        # We can try to extract parameter count from the body
        params = self._extract_bash_params(func_node, source)

        return FunctionInfo(
            name=name,
            file=str(file_path),
            line_start=func_node.start_point[0] + 1,
            line_end=func_node.end_point[0] + 1,
            parameters=params,
            return_type=None,  # Bash doesn't have explicit return types
            is_method=False,
        )

    def _extract_bash_params(self, node: Any, source: bytes) -> list[str]:
        """Extract Bash function parameters from the body.

        Bash uses positional parameters ($1, $2, etc.) so we look for
        references to determine how many parameters the function uses.
        """
        params: list[str] = []
        seen_params: set[int] = set()

        def find_param_refs(n: Any) -> None:
            if n.type == "special_variable_name":
                text = self._get_node_text(n, source)
                if text.startswith("$"):
                    try:
                        param_num = int(text[1:])
                        if param_num > 0:
                            seen_params.add(param_num)
                    except ValueError:
                        pass
            for child in n.children:
                find_param_refs(child)

        find_param_refs(node)

        # Build parameter list based on what we found
        if seen_params:
            max_param = max(seen_params)
            for i in range(1, max_param + 1):
                params.append(f"${i}")

        return params
