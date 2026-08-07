"""C/C++ function extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import FunctionInfo


class CppExtractor(BaseExtractor):
    """Extract function definitions from C/C++ source code."""

    def extract_functions(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract C/C++ function definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Any) -> None:
            if n.type == "function_definition":
                decl_node = n.child_by_field_name("declarator")
                if decl_node:
                    func_info = self._extract_function_info(decl_node, n, file_path, source)
                    if func_info:
                        functions.append(func_info)

            for child in n.children:
                visit(child)

        visit(node)
        return functions

    def _extract_function_info(
        self,
        declarator: Any,
        function_node: Any,
        file_path: Path,
        source: bytes,
    ) -> FunctionInfo | None:
        """Extract function info from a function declarator node."""
        # Get function name from declarator
        # The declarator can be: function_declarator, identifier, etc.
        name_node = None
        params = []
        return_type = None

        if declarator.type == "function_declarator":
            # Get name from the declarator
            name_part = declarator.child_by_field_name("declarator")
            if name_part:
                if name_part.type == "identifier":
                    name_node = name_part
                elif name_part.type == "field_identifier":
                    name_node = name_part
                elif name_part.type == "qualified_identifier":
                    # C++ class method: Class::method
                    # Get the last part
                    name_node = name_part

            # Extract parameters
            params_node = declarator.child_by_field_name("parameters")
            if params_node:
                params = self._extract_cpp_params(params_node, source)

        elif declarator.type == "identifier":
            name_node = declarator
            # Look for parameters in parent function_definition
            params_node = function_node.child_by_field_name("declarator")
            if params_node and params_node.type == "function_declarator":
                params_list = params_node.child_by_field_name("parameters")
                if params_list:
                    params = self._extract_cpp_params(params_list, source)

        elif declarator.type == "field_identifier":
            name_node = declarator

        if not name_node:
            return None

        name = self._get_node_text(name_node, source)

        # Try to extract return type from the function_definition
        type_node = function_node.child_by_field_name("type")
        if type_node:
            return_type = self._get_node_text(type_node, source).strip()
        else:
            # For constructors/destructors, return type is None
            pass

        return FunctionInfo(
            name=name,
            file=str(file_path),
            line_start=function_node.start_point[0] + 1,
            line_end=function_node.end_point[0] + 1,
            parameters=params,
            return_type=return_type if return_type else None,
            is_method="::" in name or self._is_in_class_context(function_node),
        )

    def _extract_cpp_params(self, params_node: Any, source: bytes) -> list[str]:
        """Extract C/C++ function parameters."""
        params: list[str] = []
        for child in params_node.children:
            if child.type in ("parameter_declaration", "parameter_list"):
                param_text = source[child.start_byte:child.end_byte].decode().strip()
                # Clean up the parameter text
                param_text = param_text.strip("()")
                if param_text and param_text != "void":
                    params.append(param_text)
        return params

    def _is_in_class_context(self, node: Any) -> bool:
        """Check if function is inside a class/struct context."""
        current = node
        while current:
            if current.type in ("class_specifier", "struct_specifier", "namespace_definition"):
                return True
            current = current.parent
        return False
