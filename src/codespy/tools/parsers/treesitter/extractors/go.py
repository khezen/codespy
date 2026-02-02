"""Go function extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import FunctionInfo


class GoExtractor(BaseExtractor):
    """Extract function definitions from Go source code."""

    def extract_functions(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract Go function definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Any) -> None:
            if n.type == "function_declaration":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = self._get_node_text(name_node, source)
                    params = self._extract_go_params(n, source)
                    return_type = self._extract_go_return_type(n, source)

                    functions.append(FunctionInfo(
                        name=name,
                        file=str(file_path),
                        line_start=n.start_point[0] + 1,
                        line_end=n.end_point[0] + 1,
                        parameters=params,
                        return_type=return_type,
                        is_method=False,
                    ))

            elif n.type == "method_declaration":
                name_node = n.child_by_field_name("name")
                receiver = n.child_by_field_name("receiver")
                if name_node:
                    name = self._get_node_text(name_node, source)
                    params = self._extract_go_params(n, source)
                    return_type = self._extract_go_return_type(n, source)
                    receiver_type = None
                    if receiver:
                        receiver_type = self._get_node_text(receiver, source)

                    functions.append(FunctionInfo(
                        name=name,
                        file=str(file_path),
                        line_start=n.start_point[0] + 1,
                        line_end=n.end_point[0] + 1,
                        parameters=params,
                        return_type=return_type,
                        is_method=True,
                        receiver_type=receiver_type,
                    ))

            for child in n.children:
                visit(child)

        visit(node)
        return functions

    def _extract_go_params(self, node: Any, source: bytes) -> list[str]:
        """Extract Go function parameters."""
        params: list[str] = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for child in params_node.children:
                if child.type == "parameter_declaration":
                    param_text = self._get_node_text(child, source)
                    params.append(param_text.strip())
        return params

    def _extract_go_return_type(self, node: Any, source: bytes) -> str | None:
        """Extract Go function return type."""
        result = node.child_by_field_name("result")
        if result:
            return self._get_node_text(result, source).strip()
        return None