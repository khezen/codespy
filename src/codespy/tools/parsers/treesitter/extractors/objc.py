"""Objective-C function extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import FunctionInfo


class ObjCExtractor(BaseExtractor):
    """Extract function/method definitions from Objective-C source code."""

    def extract_functions(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract Objective-C function/method definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Any, in_class: bool = False) -> None:
            if n.type == "function_definition":
                # C-style function
                declarator = n.child_by_field_name("declarator")
                if declarator:
                    name = self._extract_c_declarator_name(declarator, source)
                    if name:
                        functions.append(FunctionInfo(
                            name=name,
                            file=str(file_path),
                            line_start=n.start_point[0] + 1,
                            line_end=n.end_point[0] + 1,
                            parameters=[],
                            is_method=False,
                        ))

            elif n.type == "method_definition":
                # ObjC method - extract selector
                selector = n.child_by_field_name("selector")
                if selector:
                    name = self._get_node_text(selector, source)
                    # Clean up method name (remove colons for simple name)
                    name = name.split(":")[0] if ":" in name else name

                    functions.append(FunctionInfo(
                        name=name,
                        file=str(file_path),
                        line_start=n.start_point[0] + 1,
                        line_end=n.end_point[0] + 1,
                        parameters=[],
                        is_method=True,
                    ))

            elif n.type in ("class_interface", "class_implementation", "category_interface",
                           "category_implementation"):
                for child in n.children:
                    visit(child, in_class=True)
                return

            for child in n.children:
                visit(child, in_class)

        visit(node)
        return functions

    def _extract_c_declarator_name(self, node: Any, source: bytes) -> str | None:
        """Extract function name from C-style declarator."""
        if node.type == "identifier":
            return self._get_node_text(node, source)
        elif node.type == "function_declarator":
            declarator = node.child_by_field_name("declarator")
            if declarator:
                return self._extract_c_declarator_name(declarator, source)
        for child in node.children:
            result = self._extract_c_declarator_name(child, source)
            if result:
                return result
        return None