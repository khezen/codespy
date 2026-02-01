"""Tree-sitter integration for accurate AST-based code analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import tree-sitter and language grammars
try:
    import tree_sitter_go as ts_go
    import tree_sitter_javascript as ts_javascript
    import tree_sitter_python as ts_python
    import tree_sitter_typescript as ts_typescript
    from tree_sitter import Language, Node, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Define placeholder types for type hints
    Parser = Any  # type: ignore[misc]
    Language = Any  # type: ignore[misc]
    Node = Any  # type: ignore[misc]
    logger.debug("Tree-sitter not available, will use ripgrep fallback")


@dataclass
class FunctionInfo:
    """Information about a function/method definition."""

    name: str
    file: str
    line_start: int
    line_end: int
    parameters: list[str]
    return_type: str | None = None
    is_method: bool = False
    receiver_type: str | None = None  # For Go methods
    docstring: str | None = None


@dataclass
class CallInfo:
    """Information about a function call."""

    function_name: str
    file: str
    line_number: int
    line_content: str
    arguments_count: int
    caller_function: str | None = None  # Function containing this call


@dataclass
class SymbolInfo:
    """Information about a symbol (variable, type, etc.)."""

    name: str
    kind: str  # function, class, variable, type, etc.
    file: str
    line_number: int
    scope: str | None = None


class TreeSitterAnalyzer:
    """AST-based code analyzer using tree-sitter.

    Provides accurate code analysis including:
    - Function/method definitions
    - Function calls with scope awareness
    - Symbol extraction
    """

    # Language configurations
    LANGUAGE_MAP = {
        "go": ("go", ["func_declaration", "method_declaration"]),
        "py": ("python", ["function_definition", "async_function_definition"]),
        "js": ("javascript", ["function_declaration", "arrow_function", "method_definition"]),
        "jsx": ("javascript", ["function_declaration", "arrow_function", "method_definition"]),
        "ts": ("typescript", ["function_declaration", "arrow_function", "method_definition"]),
        "tsx": ("tsx", ["function_declaration", "arrow_function", "method_definition"]),
    }

    def __init__(self, repo_path: Path) -> None:
        """Initialize the analyzer.

        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = repo_path
        self._parsers: dict[str, Parser] = {}
        self._languages: dict[str, Language] = {}

        if TREE_SITTER_AVAILABLE:
            self._init_languages()

    @property
    def available(self) -> bool:
        """Check if tree-sitter is available."""
        return TREE_SITTER_AVAILABLE and len(self._languages) > 0

    def _init_languages(self) -> None:
        """Initialize language parsers."""
        if not TREE_SITTER_AVAILABLE:
            return

        try:
            # Initialize Go
            self._languages["go"] = Language(ts_go.language())
            parser = Parser(self._languages["go"])
            self._parsers["go"] = parser
        except Exception as e:
            logger.debug(f"Failed to initialize Go parser: {e}")

        try:
            # Initialize Python
            self._languages["python"] = Language(ts_python.language())
            parser = Parser(self._languages["python"])
            self._parsers["python"] = parser
        except Exception as e:
            logger.debug(f"Failed to initialize Python parser: {e}")

        try:
            # Initialize JavaScript
            self._languages["javascript"] = Language(ts_javascript.language())
            parser = Parser(self._languages["javascript"])
            self._parsers["javascript"] = parser
        except Exception as e:
            logger.debug(f"Failed to initialize JavaScript parser: {e}")

        try:
            # Initialize TypeScript
            self._languages["typescript"] = Language(ts_typescript.language_typescript())
            parser = Parser(self._languages["typescript"])
            self._parsers["typescript"] = parser
        except Exception as e:
            logger.debug(f"Failed to initialize TypeScript parser: {e}")

        try:
            # Initialize TSX
            self._languages["tsx"] = Language(ts_typescript.language_tsx())
            parser = Parser(self._languages["tsx"])
            self._parsers["tsx"] = parser
        except Exception as e:
            logger.debug(f"Failed to initialize TSX parser: {e}")

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

        try:
            source = content.encode() if content else file_path.read_bytes()

            tree = parser.parse(source)
            functions: list[FunctionInfo] = []

            if language == "go":
                functions.extend(self._extract_go_functions(tree.root_node, file_path, source))
            elif language == "python":
                functions.extend(self._extract_python_functions(tree.root_node, file_path, source))
            elif language in ("javascript", "typescript", "tsx"):
                functions.extend(self._extract_js_functions(tree.root_node, file_path, source))

            return functions

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

            # Find all call expressions
            self._find_calls_recursive(
                tree.root_node,
                function_name,
                file_path,
                source,
                calls,
                language,
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

            # Find all call expressions without filtering by name
            self._find_all_calls_recursive(
                tree.root_node,
                file_path,
                source,
                calls,
                language,
            )

            return calls

        except Exception as e:
            logger.debug(f"Failed to find calls in {file_path}: {e}")
            return []

    def _extract_go_functions(
        self,
        node: Node,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract Go function definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Node) -> None:
            if n.type == "function_declaration":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte:name_node.end_byte].decode()
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
                    name = source[name_node.start_byte:name_node.end_byte].decode()
                    params = self._extract_go_params(n, source)
                    return_type = self._extract_go_return_type(n, source)
                    receiver_type = None
                    if receiver:
                        receiver_type = source[receiver.start_byte:receiver.end_byte].decode()

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

    def _extract_go_params(self, node: Node, source: bytes) -> list[str]:
        """Extract Go function parameters."""
        params: list[str] = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for child in params_node.children:
                if child.type == "parameter_declaration":
                    param_text = source[child.start_byte:child.end_byte].decode()
                    params.append(param_text.strip())
        return params

    def _extract_go_return_type(self, node: Node, source: bytes) -> str | None:
        """Extract Go function return type."""
        result = node.child_by_field_name("result")
        if result:
            return source[result.start_byte:result.end_byte].decode().strip()
        return None

    def _extract_python_functions(
        self,
        node: Node,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract Python function definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Node, in_class: bool = False) -> None:
            if n.type in ("function_definition", "async_function_definition"):
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte:name_node.end_byte].decode()
                    params = self._extract_python_params(n, source)
                    return_type = self._extract_python_return_type(n, source)

                    functions.append(FunctionInfo(
                        name=name,
                        file=str(file_path),
                        line_start=n.start_point[0] + 1,
                        line_end=n.end_point[0] + 1,
                        parameters=params,
                        return_type=return_type,
                        is_method=in_class,
                    ))

            elif n.type == "class_definition":
                for child in n.children:
                    visit(child, in_class=True)
                return

            for child in n.children:
                visit(child, in_class)

        visit(node)
        return functions

    def _extract_python_params(self, node: Node, source: bytes) -> list[str]:
        """Extract Python function parameters."""
        params: list[str] = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for child in params_node.children:
                if child.type in ("identifier", "typed_parameter", "default_parameter"):
                    param_text = source[child.start_byte:child.end_byte].decode()
                    params.append(param_text.strip())
        return params

    def _extract_python_return_type(self, node: Node, source: bytes) -> str | None:
        """Extract Python function return type annotation."""
        return_type = node.child_by_field_name("return_type")
        if return_type:
            return source[return_type.start_byte:return_type.end_byte].decode().strip()
        return None

    def _extract_js_functions(
        self,
        node: Node,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract JavaScript/TypeScript function definitions."""
        functions: list[FunctionInfo] = []

        def visit(n: Node, in_class: bool = False) -> None:
            if n.type == "function_declaration":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte:name_node.end_byte].decode()
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
                    name = source[name_node.start_byte:name_node.end_byte].decode()
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
                                name = source[name_node.start_byte:name_node.end_byte].decode()
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

    def _extract_js_params(self, node: Node, source: bytes) -> list[str]:
        """Extract JavaScript function parameters."""
        params: list[str] = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for child in params_node.children:
                if child.type in ("identifier", "required_parameter", "optional_parameter"):
                    param_text = source[child.start_byte:child.end_byte].decode()
                    params.append(param_text.strip())
        return params

    def _find_calls_recursive(
        self,
        node: Node,
        function_name: str,
        file_path: Path,
        source: bytes,
        calls: list[CallInfo],
        language: str,
        current_function: str | None = None,
    ) -> None:
        """Recursively find calls to a specific function."""
        # Track current function context
        if node.type in ("function_declaration", "method_declaration",
                         "function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                current_function = source[name_node.start_byte:name_node.end_byte].decode()

        # Check for call expressions - Python uses "call", Go/JS use "call_expression"
        if node.type in ("call_expression", "call"):
            # Get function node - different access patterns for different languages
            func_node = node.child_by_field_name("function")
            if func_node is None and len(node.children) > 0:
                # Python: first child is the function being called
                func_node = node.children[0]

            if func_node:
                # Get the function name being called
                call_name = self._extract_call_name(func_node, source)
                if call_name == function_name:
                    # Get arguments - Python uses "arguments", Go/JS use "arguments"
                    args_node = node.child_by_field_name("arguments")
                    if args_node is None:
                        # Python: look for argument_list child
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
                child, function_name, file_path, source, calls, language, current_function
            )

    def _find_all_calls_recursive(
        self,
        node: Node,
        file_path: Path,
        source: bytes,
        calls: list[CallInfo],
        language: str,
        current_function: str | None = None,
    ) -> None:
        """Recursively find all function calls."""
        # Track current function context
        if node.type in ("function_declaration", "method_declaration",
                         "function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                current_function = source[name_node.start_byte:name_node.end_byte].decode()

        # Check for call expressions - Python uses "call", Go/JS use "call_expression"
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
                child, file_path, source, calls, language, current_function
            )

    def _extract_call_name(self, node: Node, source: bytes) -> str | None:
        """Extract the function name from a call expression."""
        if node.type == "identifier":
            return source[node.start_byte:node.end_byte].decode()
        elif node.type == "member_expression":
            # For calls like obj.method(), get just the method name
            prop = node.child_by_field_name("property")
            if prop:
                return source[prop.start_byte:prop.end_byte].decode()
        elif node.type == "selector_expression":
            # Go: pkg.Function() or obj.Method()
            field = node.child_by_field_name("field")
            if field:
                return source[field.start_byte:field.end_byte].decode()
        elif node.type == "attribute":
            # Python: obj.method()
            attr = node.child_by_field_name("attribute")
            if attr:
                return source[attr.start_byte:attr.end_byte].decode()
        return None
