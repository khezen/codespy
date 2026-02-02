"""JavaScript/TypeScript function extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import FunctionInfo


class JavaScriptExtractor(BaseExtractor):
    """Extract function definitions from JavaScript/TypeScript source code."""

    def extract_functions(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract JavaScript/TypeScript function definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Any, in_class: bool = False) -> None:
            if n.type == "function_declaration":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = self._get_node_text(name_node, source)
                    params = self._extract_js_params(n, source)

                    functions.append(FunctionInfo(
                        name=name,
                        file=str(file_path),
                        line_start=n.start_point[0] + 1,
                        line_end=n.end_point[0] + 1,
                        parameters=params,
                        is_method=False,
                    ))

            elif n.type == "method_definition":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = self._get_node_text(name_node, source)
                    params = self._extract_js_params(n, source)

                    functions.append(FunctionInfo(
                        name=name,
                        file=str(file_path),
                        line_start=n.start_point[0] + 1,
                        line_end=n.end_point[0] + 1,
                        parameters=params,
                        is_method=True,
                    ))

            elif n.type == "lexical_declaration":
                # Handle: const foo = () => {} or const foo = function() {}
                for decl in n.children:
                    if decl.type == "variable_declarator":
                        name_node = decl.child_by_field_name("name")
                        value_node = decl.child_by_field_name("value")
                        if name_node and value_node:
                            if value_node.type in ("arrow_function", "function_expression"):
                                name = self._get_node_text(name_node, source)
                                params = self._extract_js_params(value_node, source)

                                functions.append(FunctionInfo(
                                    name=name,
                                    file=str(file_path),
                                    line_start=n.start_point[0] + 1,
                                    line_end=n.end_point[0] + 1,
                                    parameters=params,
                                    is_method=False,
                                ))

            elif n.type == "class_declaration":
                for child in n.children:
                    visit(child, in_class=True)
                return

            for child in n.children:
                visit(child, in_class)

        visit(node)
        return functions

    def _extract_js_params(self, node: Any, source: bytes) -> list[str]:
        """Extract JavaScript function parameters."""
        params: list[str] = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for child in params_node.children:
                if child.type in ("identifier", "required_parameter", "optional_parameter"):
                    param_text = self._get_node_text(child, source)
                    params.append(param_text.strip())
        return params