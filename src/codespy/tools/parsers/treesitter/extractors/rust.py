"""Rust function extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import FunctionInfo


class RustExtractor(BaseExtractor):
    """Extract function definitions from Rust source code."""

    def extract_functions(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract Rust function definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Any, in_impl: bool = False) -> None:
            if n.type == "function_item":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = self._get_node_text(name_node, source)
                    params = self._extract_rust_params(n, source)
                    return_type = self._extract_rust_return_type(n, source)

                    functions.append(FunctionInfo(
                        name=name,
                        file=str(file_path),
                        line_start=n.start_point[0] + 1,
                        line_end=n.end_point[0] + 1,
                        parameters=params,
                        return_type=return_type,
                        is_method=in_impl,
                    ))

            elif n.type in ("impl_item", "trait_item"):
                for child in n.children:
                    visit(child, in_impl=True)
                return

            for child in n.children:
                visit(child, in_impl)

        visit(node)
        return functions

    def _extract_rust_params(self, node: Any, source: bytes) -> list[str]:
        """Extract Rust function parameters."""
        params: list[str] = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for child in params_node.children:
                if child.type == "parameter":
                    param_text = self._get_node_text(child, source)
                    params.append(param_text.strip())
                elif child.type == "self_parameter":
                    param_text = self._get_node_text(child, source)
                    params.append(param_text.strip())
        return params

    def _extract_rust_return_type(self, node: Any, source: bytes) -> str | None:
        """Extract Rust function return type."""
        return_type = node.child_by_field_name("return_type")
        if return_type:
            return self._get_node_text(return_type, source).strip()
        return None