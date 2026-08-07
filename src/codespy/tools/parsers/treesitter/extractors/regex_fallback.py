"""Regex-based function extractor for languages without tree-sitter grammars.

This module provides lightweight pattern matching as a fallback when tree-sitter
parsers are not available. It uses ripgrep for fast line-based searching combined
with heuristics to identify function definitions and extract signatures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import FunctionInfo


@dataclass
class LanguagePattern:
    """Pattern configuration for a language."""

    name: str
    extensions: set[str]
    # Pattern to match function definition line
    function_pattern: re.Pattern
    # Pattern to extract parameters from the definition line
    param_pattern: re.Pattern | None = None
    # Pattern to detect end of function (e.g., closing brace)
    end_pattern: re.Pattern | None = None
    # Comment characters to strip from signatures
    comment_prefix: str | None = None


class RegexFallbackExtractor(BaseExtractor):
    """Fallback extractor using regex patterns for unsupported languages.

    Supported languages:
    - C/C++ (.c, .cpp, .h, .hpp)
    - C# (.cs)
    - Ruby (.rb)
    - PHP (.php)
    - Shell/Bash (.sh, .bash)
    - SQL (.sql) - basic stored procedure detection
    """

    # Language patterns for function detection
    PATTERNS: dict[str, LanguagePattern] = {
        "c_cpp": LanguagePattern(
            name="C/C++",
            extensions={".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".hxx"},
            function_pattern=re.compile(
                r'^[\s]*'  # leading whitespace
                r'(?:static\s+|inline\s+|extern\s+|virtual\s+|explicit\s+|constexpr\s+|consteval\s+)*'  # modifiers
                r'(?:[\w:<>,\s\*&]+?\s+)?'  # return type with templates/pointers
                r'(\w+)'  # function name (capture group 1)
                r'\s*\([^)]*\)'  # parameters
                r'(?:\s*const)?'  # optional const
                r'(?:\s*->\s*[\w:<>,\s\*&]+)?'  # optional trailing return (C++)
                r'\s*[{;]',  # opening brace or semicolon
                re.MULTILINE,
            ),
            param_pattern=re.compile(r'\(([^)]*)\)'),
            end_pattern=re.compile(r'^[\s]*}'),
            comment_prefix="//",
        ),
        "csharp": LanguagePattern(
            name="C#",
            extensions={".cs"},
            function_pattern=re.compile(
                r'^[\s]*'  # leading whitespace
                r'(?:public\s+|private\s+|protected\s+|internal\s+|static\s+|virtual\s+|'
                r'override\s+|abstract\s+|sealed\s+|async\s+|unsafe\s+|extern\s+)*'  # modifiers
                r'(?:[\w<>,\s\[\]]+?\s+)'  # return type (including generic/Task types)
                r'(\w+)'  # function name (capture group 1)
                r'\s*\([^)]*\)'  # parameters
                r'(?:\s*where\s+\w+\s*:\s*[\w<>,\s]+)?'  # optional generic constraint
                r'\s*[{(]',  # opening brace or expression body
                re.MULTILINE,
            ),
            param_pattern=re.compile(r'\(([^)]*)\)'),
            end_pattern=re.compile(r'^[\s]*[}]'),
            comment_prefix="//",
        ),
        "ruby": LanguagePattern(
            name="Ruby",
            extensions={".rb", ".rbw", ".rake", ".gemspec"},
            function_pattern=re.compile(
                r'^[\s]*'  # leading whitespace
                r'(?:private\s+|protected\s+|public\s+)?'  # visibility modifiers
                r'def\s+'  # def keyword
                r'(?:self\.)?'  # optional self.
                r'(\w+[?!=]?)'  # function name with optional ? ! = (capture group 1)
                r'(?:\s*\([^)]*\))?'  # optional parentheses with params
                r'(?:\s+|$)',  # whitespace or end of line
                re.MULTILINE,
            ),
            # Ruby params are complex (block syntax, etc), keep simple
            param_pattern=re.compile(r'def\s+(?:self\.)?\w+\s*\(([^)]*)\)'),
            end_pattern=re.compile(r'^[\s]*end\s*$'),
            comment_prefix="#",
        ),
        "php": LanguagePattern(
            name="PHP",
            extensions={".php", ".php4", ".php5", ".phtml"},
            function_pattern=re.compile(
                r'^[\s]*'  # leading whitespace
                r'(?:public\s+|private\s+|protected\s+)?'  # visibility
                r'(?:static\s+)?'  # optional static
                r'(?:abstract\s+|final\s+)?'  # optional abstract/final
                r'function\s+'  # function keyword
                r'(&)?'  # optional reference return
                r'(\w+)'  # function name (capture group 2, 1 is &)
                r'\s*\([^)]*\)',  # parameters
                re.MULTILINE,
            ),
            param_pattern=re.compile(r'\(([^)]*)\)'),
            end_pattern=re.compile(r'^[\s]*}'),
            comment_prefix="//",  # Also supports # but // is more common
        ),
        "shell": LanguagePattern(
            name="Shell/Bash",
            extensions={".sh", ".bash", ".zsh", ".ksh", ".dash"},
            function_pattern=re.compile(
                r'^[\s]*'  # leading whitespace
                r'(?:function\s+)?'  # optional function keyword
                r'(\w+)'  # function name (capture group 1)
                r'\s*\(\s*\)'  # empty parentheses
                r'\s*\{',  # opening brace
                re.MULTILINE,
            ),
            # Shell functions typically don't declare params in signature
            param_pattern=None,
            end_pattern=re.compile(r'^[\s]*}'),
            comment_prefix="#",
        ),
        "sql": LanguagePattern(
            name="SQL",
            extensions={".sql"},
            function_pattern=re.compile(
                r'^[\s]*'  # leading whitespace
                r'(?:CREATE\s+OR\s+REPLACE\s+)?'  # optional create or replace
                r'(?:CREATE\s+)?'  # optional create
                r'(?:PROCEDURE|FUNCTION|TRIGGER|EVENT)\s+'  # object type
                r'(?:[\w.]+\s+)?'  # optional schema prefix
                r'(\w+)'  # name (capture group 1)
                r'\s*\(',  # opening paren for params
                re.MULTILINE | re.IGNORECASE,
            ),
            param_pattern=re.compile(r'\(([^)]*)\)'),
            end_pattern=re.compile(r'^[\s]*END\s*;?', re.IGNORECASE),
            comment_prefix="--",
        ),
    }

    def __init__(self) -> None:
        """Initialize the extractor."""
        # Build extension -> pattern mapping for fast lookup
        self._ext_to_pattern: dict[str, LanguagePattern] = {}
        for pattern in self.PATTERNS.values():
            for ext in pattern.extensions:
                self._ext_to_pattern[ext.lower()] = pattern

    def _get_pattern(self, file_path: Path) -> LanguagePattern | None:
        """Get pattern for file based on extension."""
        ext = file_path.suffix.lower()
        return self._ext_to_pattern.get(ext)

    def _strip_comments(self, line: str, comment_prefix: str | None) -> str:
        """Remove inline comments from a line."""
        if not comment_prefix:
            return line
        # Handle both // and # style comments
        if comment_prefix in line:
            return line.split(comment_prefix)[0].rstrip()
        return line

    def _extract_params(self, line: str, pattern: LanguagePattern) -> list[str]:
        """Extract parameter names from function signature."""
        if not pattern.param_pattern:
            return []

        match = pattern.param_pattern.search(line)
        if not match:
            return []

        params_str = match.group(1).strip()
        if not params_str:
            return []

        # Simple splitting - handles common cases
        # For complex cases (templates, function pointers), just return raw params
        params = []
        current_param = ""
        depth = 0

        for char in params_str:
            if char in "(<{":
                depth += 1
                current_param += char
            elif char in ")>}:
                depth -= 1
                current_param += char
            elif char == "," and depth == 0:
                # End of parameter
                param = current_param.strip()
                if param:
                    # Extract parameter name (last word before any =)
                    param_name = self._extract_param_name(param)
                    if param_name:
                        params.append(param_name)
                current_param = ""
            else:
                current_param += char

        # Handle last parameter
        if current_param.strip():
            param_name = self._extract_param_name(current_param.strip())
            if param_name:
                params.append(param_name)

        return params

    def _extract_param_name(self, param: str) -> str | None:
        """Extract parameter name from parameter declaration.

        Examples:
        - "int x" -> "x"
        - "const std::string& name" -> "name"
        - "int x = 5" -> "x"
        - "std::vector<int> items" -> "items"
        """
        # Remove default values
        if "=" in param:
            param = param.split("=")[0].strip()

        # Split by whitespace and take last part
        # Handle pointers/references by stripping * and &
        parts = param.split()
        if not parts:
            return None

        name_part = parts[-1]
        # Strip *, &, etc from the end
        name = name_part.rstrip("*&").strip()

        # Validate it's a reasonable identifier
        if re.match(r'^[a-zA-Z_]\w*$', name):
            return name
        return None

    def _find_function_end(
        self,
        lines: list[str],
        start_line: int,
        pattern: LanguagePattern,
    ) -> int:
        """Find the approximate end line of a function.

        Uses brace counting for C-style languages or end keyword detection.
        """
        if not pattern.end_pattern and not pattern.function_pattern:
            return start_line

        brace_depth = 0
        in_function = False

        for i, line in enumerate(lines[start_line - 1:], start=start_line):
            stripped = line.strip()

            if not in_function:
                # Look for opening brace to enter function
                if "{" in line:
                    brace_depth = line.count("{") - line.count("}")
                    in_function = True
                elif pattern.name in ("ruby", "sql") and stripped.startswith("def "):
                    in_function = True
                continue

            # In function body - track braces
            brace_depth += line.count("{") - line.count("}")

            # Check for end pattern
            if pattern.end_pattern and pattern.end_pattern.match(line):
                return i

            # For brace-based languages, depth reaching 0 means end
            if pattern.name not in ("ruby", "sql") and brace_depth <= 0:
                return i

            # Safety limit - don't search forever
            if i - start_line > 500:
                return start_line + 100

        return min(start_line + 50, len(lines))

    def extract_functions(
        self,
        root_node: Any,  # Not used, for compatibility with BaseExtractor
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Extract function definitions using regex patterns.

        Args:
            root_node: Not used (for compatibility)
            file_path: Path to the source file
            source: File content as bytes

        Returns:
            List of FunctionInfo objects
        """
        pattern = self._get_pattern(file_path)
        if not pattern:
            return []

        try:
            content = source.decode("utf-8", errors="ignore")
        except Exception:
            return []

        lines = content.split("\n")
        functions = []
        seen_lines: set[int] = set()  # Track to avoid duplicates

        for match in pattern.function_pattern.finditer(content):
            name = match.group(1)
            if not name:
                continue

            # Calculate line number
            line_start = content[: match.start()].count("\n") + 1

            # Skip if we've seen this line (can happen with overlapping patterns)
            if line_start in seen_lines:
                continue
            seen_lines.add(line_start)

            # Get the full line for parameter extraction
            line_idx = line_start - 1
            if line_idx >= len(lines):
                continue

            line = lines[line_idx]
            clean_line = self._strip_comments(line, pattern.comment_prefix)

            # Extract parameters
            params = self._extract_params(clean_line, pattern)

            # Find approximate end line
            line_end = self._find_function_end(lines, line_start, pattern)

            # Determine return type (heuristic)
            return_type = None
            if pattern.name in ("c_cpp", "csharp"):
                # Try to extract return type from before function name
                func_match = pattern.function_pattern.match(clean_line)
                if func_match:
                    prefix = clean_line[: func_match.start(1)].strip()
                    # Remove modifiers
                    for mod in ["static", "inline", "extern", "virtual", "explicit",
                                "constexpr", "consteval", "public", "private",
                                "protected", "internal", "async", "abstract",
                                "sealed", "unsafe", "override"]:
                        prefix = re.sub(rf"\b{mod}\b\s*", "", prefix)
                    return_type = prefix.strip() if prefix.strip() else None

            functions.append(
                FunctionInfo(
                    name=name,
                    file=str(file_path),
                    line_start=line_start,
                    line_end=line_end,
                    parameters=params,
                    return_type=return_type,
                    is_method=False,  # Could be refined
                    receiver_type=None,
                    docstring=None,
                )
            )

        return functions

    def extract_signatures(
        self,
        file_path: Path,
        source: bytes,
    ) -> dict[str, str]:
        """Extract function signatures as strings.

        Returns a mapping of function name -> signature string
        for use in hunks+metadata approach.

        Args:
            file_path: Path to the source file
            source: File content as bytes

        Returns:
            Dict of function name -> signature string
        """
        functions = self.extract_functions(None, file_path, source)
        return {
            f.name: self._format_signature(f) for f in functions
        }

    def _format_signature(self, func: FunctionInfo) -> str:
        """Format a FunctionInfo as a signature string."""
        params_str = ", ".join(func.parameters)
        if func.return_type:
            return f"{func.name}({params_str}) -> {func.return_type}"
        return f"{func.name}({params_str})"
