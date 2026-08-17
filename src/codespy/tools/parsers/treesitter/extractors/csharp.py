"""C# function/method extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import FunctionInfo


class CSharpExtractor(BaseExtractor):
    """Extract method definitions from C# source code."""

    def extract_functions(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract C# method definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Any, in_type: bool = False) -> None:
            if n.type == "method_declaration":
                func_info = self._extract_method_info(n, file_path, source)
                if func_info:
                    functions.append(func_info)

            elif n.type == "constructor_declaration":
                func_info = self._extract_constructor_info(n, file_path, source)
                if func_info:
                    functions.append(func_info)

            elif n.type == "local_function_statement":
                func_info = self._extract_local_function_info(n, file_path, source)
                if func_info:
                    functions.append(func_info)

            for child in n.children:
                # Track if we're inside a class/interface/struct
                is_type = n.type in ("class_declaration", "interface_declaration", "struct_declaration", "record_declaration")
                visit(child, in_type or is_type)

        visit(node)
        return functions

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
        params = self._extract_csharp_params(method_node, source)
        return_type = self._extract_csharp_return_type(method_node, source)

        return FunctionInfo(
            name=name,
            file=str(file_path),
            line_start=method_node.start_point[0] + 1,
            line_end=method_node.end_point[0] + 1,
            parameters=params,
            return_type=return_type,
            is_method=True,
        )

    def _extract_constructor_info(
        self,
        ctor_node: Any,
        file_path: Path,
        source: bytes,
    ) -> FunctionInfo | None:
        """Extract constructor info."""
        name_node = ctor_node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source)
        params = self._extract_csharp_params(ctor_node, source)

        return FunctionInfo(
            name=name,
            file=str(file_path),
            line_start=ctor_node.start_point[0] + 1,
            line_end=ctor_node.end_point[0] + 1,
            parameters=params,
            return_type=None,  # Constructors return the class type
            is_method=True,
        )

    def _extract_local_function_info(
        self,
        func_node: Any,
        file_path: Path,
        source: bytes,
    ) -> FunctionInfo | None:
        """Extract local function info (C# 7.0+)."""
        name_node = func_node.child_by_field_name("name")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source)
        params = self._extract_csharp_params(func_node, source)
        return_type = self._extract_csharp_return_type(func_node, source)

        return FunctionInfo(
            name=name,
            file=str(file_path),
            line_start=func_node.start_point[0] + 1,
            line_end=func_node.end_point[0] + 1,
            parameters=params,
            return_type=return_type,
            is_method=False,  # Local functions are not methods
        )

    def _extract_csharp_params(self, node: Any, source: bytes) -> list[str]:
        """Extract C# method parameters."""
        params: list[str] = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for child in params_node.children:
                if child.type == "parameter":
                    # Extract parameter info
                    param_text = self._get_node_text(child, source).strip()
                    if param_text and param_text not in ("(", ")", ","):
                        params.append(param_text)
        return params

    def _extract_csharp_return_type(self, node: Any, source: bytes) -> str | None:
        """Extract C# method return type."""
        type_node = node.child_by_field_name("type")
        if type_node:
            return self._get_node_text(type_node, source).strip()
        return None
