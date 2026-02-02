"""Java function extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import FunctionInfo


class JavaExtractor(BaseExtractor):
    """Extract method definitions from Java source code."""

    def extract_functions(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract Java method definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Any, in_class: bool = False) -> None:
            if n.type == "method_declaration":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = self._get_node_text(name_node, source)
                    params = self._extract_generic_params(n, source)
                    return_type = self._extract_java_return_type(n, source)

                    functions.append(FunctionInfo(
                        name=name,
                        file=str(file_path),
                        line_start=n.start_point[0] + 1,
                        line_end=n.end_point[0] + 1,
                        parameters=params,
                        return_type=return_type,
                        is_method=True,
                    ))

            elif n.type == "constructor_declaration":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = self._get_node_text(name_node, source)
                    params = self._extract_generic_params(n, source)

                    functions.append(FunctionInfo(
                        name=name,
                        file=str(file_path),
                        line_start=n.start_point[0] + 1,
                        line_end=n.end_point[0] + 1,
                        parameters=params,
                        is_method=True,
                    ))

            elif n.type in ("class_declaration", "interface_declaration", "enum_declaration"):
                for child in n.children:
                    visit(child, in_class=True)
                return

            for child in n.children:
                visit(child, in_class)

        visit(node)
        return functions

    def _extract_java_return_type(self, node: Any, source: bytes) -> str | None:
        """Extract Java method return type."""
        type_node = node.child_by_field_name("type")
        if type_node:
            return self._get_node_text(type_node, source).strip()
        return None