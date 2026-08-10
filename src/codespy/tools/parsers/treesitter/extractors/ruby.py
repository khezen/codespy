"""Ruby function/method extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import FunctionInfo


class RubyExtractor(BaseExtractor):
    """Extract method definitions from Ruby source code."""

    def extract_functions(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract Ruby method definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Any, in_class: bool = False) -> None:
            if n.type == "method":
                func_info = self._extract_method_info(n, file_path, source, in_class)
                if func_info:
                    functions.append(func_info)

            elif n.type == "singleton_method":
                func_info = self._extract_singleton_method_info(n, file_path, source)
                if func_info:
                    functions.append(func_info)

            elif n.type == "lambda":
                # Anonymous lambda - skip (no name)
                pass

            elif n.type == "block":
                # Block passed to a method - skip
                pass

            for child in n.children:
                # Track if we're inside a class/module
                is_class = n.type in ("class", "module", "singleton_class")
                visit(child, in_class or is_class)

        visit(node)
        return functions

    def _extract_method_info(
        self,
        method_node: Any,
        file_path: Path,
        source: bytes,
        in_class: bool,
    ) -> FunctionInfo | None:
        """Extract method info from a method node."""
        name_node = method_node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source)
        params = self._extract_ruby_params(method_node, source)

        return FunctionInfo(
            name=name,
            file=str(file_path),
            line_start=method_node.start_point[0] + 1,
            line_end=method_node.end_point[0] + 1,
            parameters=params,
            return_type=None,  # Ruby is dynamically typed
            is_method=in_class,
        )

    def _extract_singleton_method_info(
        self,
        method_node: Any,
        file_path: Path,
        source: bytes,
    ) -> FunctionInfo | None:
        """Extract singleton method info (e.g., def self.method_name)."""
        name_node = method_node.child_by_field_name("name")
        if not name_node:
            return None

        name = "self." + self._get_node_text(name_node, source)
        params = self._extract_ruby_params(method_node, source)

        return FunctionInfo(
            name=name,
            file=str(file_path),
            line_start=method_node.start_point[0] + 1,
            line_end=method_node.end_point[0] + 1,
            parameters=params,
            return_type=None,
            is_method=True,
        )

    def _extract_ruby_params(self, node: Any, source: bytes) -> list[str]:
        """Extract Ruby method parameters."""
        params: list[str] = []
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return params

        for child in params_node.children:
            # Ruby parameters can be: identifier, optional_parameter, keyword_parameter, etc.
            if child.type in ("identifier", "simple_parameter"):
                param_name = self._get_node_text(child, source)
                if param_name and param_name not in ("(", ")", ",", "|", "&"):
                    params.append(param_name)
            elif child.type == "optional_parameter":
                name_node = child.child_by_field_name("name")
                if name_node:
                    param_name = self._get_node_text(name_node, source)
                    if param_name:
                        params.append(param_name)
            elif child.type == "block_parameter":
                # &block parameter
                name_node = child.child_by_field_name("name")
                if name_node:
                    param_name = "&" + self._get_node_text(name_node, source)
                    params.append(param_name)
            elif child.type == "keyword_parameter":
                name_node = child.child_by_field_name("name")
                if name_node:
                    param_name = self._get_node_text(name_node, source)
                    if param_name:
                        params.append(f"{param_name}:")

        return params
