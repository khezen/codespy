"""Tree-sitter parser for accurate AST-based code analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.extractors import (
    GoExtractor,
    JavaExtractor,
    JavaScriptExtractor,
    KotlinExtractor,
    ObjCExtractor,
    PythonExtractor,
    RustExtractor,
    SwiftExtractor,
)
from codespy.tools.parsers.treesitter.models import CallInfo, FunctionInfo

logger = logging.getLogger(__name__)

# Try to import tree-sitter and language grammars
try:
    import tree_sitter_go as ts_go
    import tree_sitter_java as ts_java
    import tree_sitter_javascript as ts_javascript
    import tree_sitter_kotlin as ts_kotlin
    import tree_sitter_objc as ts_objc
    import tree_sitter_python as ts_python
    import tree_sitter_rust as ts_rust
    import tree_sitter_swift as ts_swift
    import tree_sitter_typescript as ts_typescript
    from tree_sitter import Language, Node, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Parser = Any  # type: ignore[misc]
    Language = Any  # type: ignore[misc]
    Node = Any  # type: ignore[misc]
    logger.debug("Tree-sitter not available, will use ripgrep fallback")


class TreeSitterParser:
    """AST-based code parser using tree-sitter.

    Provides accurate code analysis including:
    - Function/method definitions
    - Function calls with scope awareness
    - Symbol extraction
    """

    # Language configurations: extension -> (language_name, function_node_types)
    LANGUAGE_MAP = {
        "go": ("go", ["func_declaration", "method_declaration"]),
        "py": ("python", ["function_definition", "async_function_definition"]),
        "js": ("javascript", ["function_declaration", "arrow_function", "method_definition"]),
        "jsx": ("javascript", ["function_declaration", "arrow_function", "method_definition"]),
        "ts": ("typescript", ["function_declaration", "arrow_function", "method_definition"]),
        "tsx": ("tsx", ["function_declaration", "arrow_function", "method_definition"]),
        "swift": ("swift", ["function_declaration"]),
        "java": ("java", ["method_declaration", "constructor_declaration"]),
        "kt": ("kotlin", ["function_declaration"]),
        "kts": ("kotlin", ["function_declaration"]),
        "m": ("objc", ["function_definition", "method_definition"]),
        "mm": ("objc", ["function_definition", "method_definition"]),
        "rs": ("rust", ["function_item"]),
    }

    def __init__(self, repo_path: Path) -> None:
        """Initialize the parser.

        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = repo_path
        self._parsers: dict[str, Parser] = {}
        self._languages: dict[str, Language] = {}
        self._extractors: dict[str, BaseExtractor] = {}

        if TREE_SITTER_AVAILABLE:
            self._init_languages()
            self._init_extractors()

    @property
    def available(self) -> bool:
        """Check if tree-sitter is available."""
        return TREE_SITTER_AVAILABLE and len(self._languages) > 0

    def _init_extractors(self) -> None:
        """Initialize language-specific extractors."""
        self._extractors = {
            "go": GoExtractor(),
            "python": PythonExtractor(),
            "javascript": JavaScriptExtractor(),
            "typescript": JavaScriptExtractor(),
            "tsx": JavaScriptExtractor(),
            "swift": SwiftExtractor(),
            "java": JavaExtractor(),
            "kotlin": KotlinExtractor(),
            "objc": ObjCExtractor(),
            "rust": RustExtractor(),
        }

    def _init_languages(self) -> None:
        """Initialize language parsers."""
        if not TREE_SITTER_AVAILABLE:
            return

        language_configs = [
            ("go", lambda: ts_go.language()),
            ("python", lambda: ts_python.language()),
            ("javascript", lambda: ts_javascript.language()),
            ("typescript", lambda: ts_typescript.language_typescript()),
            ("tsx", lambda: ts_typescript.language_tsx()),
            ("swift", lambda: ts_swift.language()),
            ("java", lambda: ts_java.language()),
            ("kotlin", lambda: ts_kotlin.language()),
            ("objc", lambda: ts_objc.language()),
            ("rust", lambda: ts_rust.language()),
        ]

        for lang_name, lang_func in language_configs:
            try:
                self._languages[lang_name] = Language(lang_func())
                parser = Parser(self._languages[lang_name])
                self._parsers[lang_name] = parser
            except Exception as e:
                logger.debug(f"Failed to initialize {lang_name} parser: {e}")

        logger.debug(f"Initialized tree-sitter for languages: {list(self._parsers.keys())}")

    def _get_parser(self, extension: str) -> Parser | None:
        """Get parser for file extension."""
        if extension not in self.LANGUAGE_MAP:
            return None
        lang_name = self.LANGUAGE_MAP[extension][0]
        return self._parsers.get(lang_name)

    def _get_language(self, extension: str) -> str | None:
        """Get language name for file extension."""
        if extension not in self.LANGUAGE_MAP:
            return None
        return self.LANGUAGE_MAP[extension][0]

    def parse_file(self, file_path: Path) -> Node | None:
        """Parse a file and return the AST root node.

        Args:
            file_path: Path to the file

        Returns:
            Root node of the AST, or None if parsing failed
        """
        if not self.available:
            return None

        extension = file_path.suffix.lstrip(".")
        parser = self._get_parser(extension)
        if not parser:
            return None

        try:
            content = file_path.read_bytes()
            tree = parser.parse(content)
            return tree.root_node
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return None

    def find_function_definitions(
        self,
        file_path: Path,
        content: str | None = None,
    ) -> list[FunctionInfo]:
        """Find all function/method definitions in a file.

        Args:
            file_path: Path to the file
            content: Optional file content (reads from file if not provided)

        Returns:
            List of function definitions
        """
        if not self.available:
            return []

        extension = file_path.suffix.lstrip(".")
        parser = self._get_parser(extension)
        language = self._get_language(extension)

        if not parser or not language:
            return []

        extractor = self._extractors.get(language)
        if not extractor:
            return []

        try:
            source = content.encode() if content else file_path.read_bytes()
            tree = parser.parse(source)
            return extractor.extract_functions(tree.root_node, file_path, source)
        except Exception as e:
            logger.debug(f"Failed to find functions in {file_path}: {e}")
            return []

    def find_function_calls(
        self,
        file_path: Path,
        function_name: str,
        content: str | None = None,
    ) -> list[CallInfo]:
        """Find all calls to a specific function in a file.

        Args:
            file_path: Path to the file
            function_name: Name of the function to find calls for
            content: Optional file content

        Returns:
            List of function calls
        """
        if not self.available:
            return []

        extension = file_path.suffix.lstrip(".")
        parser = self._get_parser(extension)
        language = self._get_language(extension)

        if not parser or not language:
            return []

        try:
            source = content.encode() if content else file_path.read_bytes()
            tree = parser.parse(source)
            calls: list[CallInfo] = []

            self._find_calls_recursive(
                tree.root_node,
                function_name,
                file_path,
                source,
                calls,
            )

            return calls
        except Exception as e:
            logger.debug(f"Failed to find calls in {file_path}: {e}")
            return []

    def find_all_calls_in_file(
        self,
        file_path: Path,
        content: str | None = None,
    ) -> list[CallInfo]:
        """Find all function calls in a file.

        Args:
            file_path: Path to the file
            content: Optional file content

        Returns:
            List of all function calls
        """
        if not self.available:
            return []

        extension = file_path.suffix.lstrip(".")
        parser = self._get_parser(extension)
        language = self._get_language(extension)

        if not parser or not language:
            return []

        try:
            source = content.encode() if content else file_path.read_bytes()
            tree = parser.parse(source)
            calls: list[CallInfo] = []

            self._find_all_calls_recursive(
                tree.root_node,
                file_path,
                source,
                calls,
            )

            return calls
        except Exception as e:
            logger.debug(f"Failed to find calls in {file_path}: {e}")
            return []

    def _find_calls_recursive(
        self,
        node: Any,
        function_name: str,
        file_path: Path,
        source: bytes,
        calls: list[CallInfo],
        current_function: str | None = None,
    ) -> None:
        """Recursively find calls to a specific function."""
        if node.type in ("function_declaration", "method_declaration",
                         "function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                current_function = source[name_node.start_byte:name_node.end_byte].decode()

        if node.type in ("call_expression", "call"):
            func_node = node.child_by_field_name("function")
            if func_node is None and len(node.children) > 0:
                func_node = node.children[0]

            if func_node:
                call_name = self._extract_call_name(func_node, source)
                if call_name == function_name:
                    args_node = node.child_by_field_name("arguments")
                    if args_node is None:
                        for child in node.children:
                            if child.type == "argument_list":
                                args_node = child
                                break

                    args_count = 0
                    if args_node:
                        args_count = sum(1 for c in args_node.children if c.type not in ("(", ")", ","))

                    line_num = node.start_point[0] + 1
                    lines = source.decode().split("\n")
                    line_content = lines[line_num - 1] if line_num <= len(lines) else ""

                    calls.append(CallInfo(
                        function_name=function_name,
                        file=str(file_path),
                        line_number=line_num,
                        line_content=line_content,
                        arguments_count=args_count,
                        caller_function=current_function,
                    ))

        for child in node.children:
            self._find_calls_recursive(
                child, function_name, file_path, source, calls, current_function
            )

    def _find_all_calls_recursive(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
        calls: list[CallInfo],
        current_function: str | None = None,
    ) -> None:
        """Recursively find all function calls."""
        if node.type in ("function_declaration", "method_declaration",
                         "function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                current_function = source[name_node.start_byte:name_node.end_byte].decode()

        if node.type in ("call_expression", "call"):
            func_node = node.child_by_field_name("function")
            if func_node is None and len(node.children) > 0:
                func_node = node.children[0]

            if func_node:
                call_name = self._extract_call_name(func_node, source)
                if call_name:
                    args_node = node.child_by_field_name("arguments")
                    if args_node is None:
                        for child in node.children:
                            if child.type == "argument_list":
                                args_node = child
                                break

                    args_count = 0
                    if args_node:
                        args_count = sum(1 for c in args_node.children if c.type not in ("(", ")", ","))

                    line_num = node.start_point[0] + 1
                    lines = source.decode().split("\n")
                    line_content = lines[line_num - 1] if line_num <= len(lines) else ""

                    calls.append(CallInfo(
                        function_name=call_name,
                        file=str(file_path),
                        line_number=line_num,
                        line_content=line_content,
                        arguments_count=args_count,
                        caller_function=current_function,
                    ))

        for child in node.children:
            self._find_all_calls_recursive(
                child, file_path, source, calls, current_function
            )

    def _extract_call_name(self, node: Any, source: bytes) -> str | None:
        """Extract the function name from a call expression."""
        if node.type == "identifier":
            return source[node.start_byte:node.end_byte].decode()
        elif node.type == "member_expression":
            prop = node.child_by_field_name("property")
            if prop:
                return source[prop.start_byte:prop.end_byte].decode()
        elif node.type == "selector_expression":
            field = node.child_by_field_name("field")
            if field:
                return source[field.start_byte:field.end_byte].decode()
        elif node.type == "attribute":
            attr = node.child_by_field_name("attribute")
            if attr:
                return source[attr.start_byte:attr.end_byte].decode()
        return None
