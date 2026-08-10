"""PHP function extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import FunctionInfo


class PHPExtractor(BaseExtractor):
    """Extract function definitions from PHP source code."""

    def extract_functions(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract PHP function definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Any, in_class: bool = False) -> None:
            if n.type == "function_definition":
                func_info = self._extract_function_info(n, file_path, source)
                if func_info:
                    functions.append(func_info)

            elif n.type == "method_declaration":
                func_info = self._extract_method_info(n, file_path, source)
                if func_info:
                    functions.append(func_info)

            elif n.type == "anonymous_function":
                func_info = self._extract_anonymous_function_info(n, file_path, source)
                if func_info:
                    functions.append(func_info)

            elif n.type == "arrow_function":
                func_info = self._extract_arrow_function_info(n, file_path, source)
                if func_info:
                    functions.append(func_info)

            for child in n.children:
                # Track if we're inside a class
                is_class = n.type in ("class_declaration", "interface_declaration", "trait_declaration")
                visit(child, in_class or is_class)

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
        params = self._extract_php_params(func_node, source)
        return_type = self._extract_php_return_type(func_node, source)

        return FunctionInfo(
            name=name,
            file=str(file_path),
            line_start=func_node.start_point[0] + 1,
            line_end=func_node.end_point[0] + 1,
            parameters=params,
            return_type=return_type,
            is_method=False,  # Functions are not methods
        )

    def _extract_method_info(
        self,
        method_node: Any,
        file_path: Path,
        source: bytes,
    ) -> FunctionInfo | None:
        """Extract method info from a method_declaration node."""
        name_node = method_node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source)
        params = self._extract_php_params(method_node, source)
        return_type = self._extract_php_return_type(method_node, source)

        return FunctionInfo(
            name=name,
            file=str(file_path),
            line_start=method_node.start_point[0] + 1,
            line_end=method_node.end_point[0] + 1,
            parameters=params,
            return_type=return_type,
            is_method=True,
        )

    def _extract_anonymous_function_info(
        self,
        func_node: Any,
        file_path: Path,
        source: bytes,
    ) -> FunctionInfo | None:
        """Extract anonymous function (closure) info."""
        # Anonymous functions don't have a name, use placeholder
        params = self._extract_php_params(func_node, source)
        return_type = self._extract_php_return_type(func_node, source)

        return FunctionInfo(
            name="(anonymous)",
            file=str(file_path),
            line_start=func_node.start_point[0] + 1,
            line_end=func_node.end_point[0] + 1,
            parameters=params,
            return_type=return_type,
            is_method=False,
        )

    def _extract_arrow_function_info(
        self,
        func_node: Any,
        file_path: Path,
        source: bytes,
    ) -> FunctionInfo | None:
        """Extract arrow function (fn() => expr) info."""
        params = self._extract_php_params(func_node, source)
        return_type = self._extract_php_return_type(func_node, source)

        return FunctionInfo(
            name="(arrow)",
            file=str(file_path),
            line_start=func_node.start_point[0] + 1,
            line_end=func_node.end_point[0] + 1,
            parameters=params,
            return_type=return_type,
            is_method=False,
        )

    def _extract_php_params(self, node: Any, source: bytes) -> list[str]:
        """Extract PHP function parameters."""
        params: list[str] = []
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return params

        for child in params_node.children:
            if child.type == "parameter":
                # Extract parameter info
                param_text = self._get_node_text(child, source).strip()
                if param_text and param_text not in ("(", ")", ","):
                    # Extract just the variable name if possible
                    var_node = child.child_by_field_name("name")
                    if var_node:
                        params.append(self._get_node_text(var_node, source))
                    else:
                        params.append(param_text)

        return params

    def _extract_php_return_type(self, node: Any, source: bytes) -> str | None:
        """Extract PHP function return type."""
        type_node = node.child_by_field_name("return_type")
        if type_node:
            return self._get_node_text(type_node, source).strip()
        return None
