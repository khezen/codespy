"""Ripgrep + heuristics fallback extractor for function detection.

This module provides a generic fallback for extracting function information
when tree-sitter is not available. It uses ripgrep to find function definition
lines and interval intersection to determine which functions are affected by
changed lines.

Algorithm:
    1. Run ripgrep on the file with generic definition patterns
       → sorted list of (line_number, function_name)
    
    2. Derive implicit boundaries: each function spans from its definition
       line to the line before the next definition (or EOF)
    
    3. Intersect these boundaries with changed_line_ranges
       → return functions that contain at least one changed line

This approach handles all cases elegantly:
    - Body changes: Changed lines fall within a function's derived range
    - New function: The definition line IS a changed line, matches itself
    - Signature changes: Definition line is changed, matches itself
    - Multiple functions: Multiple intersections found
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.models import FunctionInfo

logger = logging.getLogger(__name__)


class RipgrepHeuristicsExtractor:
    """Generic fallback extractor using ripgrep + interval intersection.

    Instead of parsing the entire file with complex AST analysis, this extractor:
    1. Uses ripgrep to find all function definition lines in the file
    2. Derives implicit boundaries: each function spans to the next definition
    3. Intersects these ranges with the changed line ranges from the patch

    This is much faster than AST parsing and handles all edge cases:
    - Body changes: Lines fall within derived range
    - New functions: Definition line IS a changed line
    - Modified signatures: Definition line is changed
    """

    # Generic patterns that match function definitions across languages.
    # Ordered by specificity - more specific patterns first.
    DEFINITION_PATTERNS: list[tuple[str, re.Pattern]] = [
        # Keyword-based: def, func, fn, fun, function, sub, proc
        # Matches: Python, Ruby, Go, JavaScript, TypeScript, Rust, etc.
        (
            "keyword",
            re.compile(
                r"^[\s]*(?:"  # leading whitespace
                r"(?:pub|priv|protected|private|public|export|async|static|inline|const|final)\s+)*"  # modifiers
                r"(?:def|func|fn|fun|function|sub|proc)\s+"  # keyword
                r"(?:self\.)?"  # optional self. for Ruby
                r"(\w+)"  # function name (capture group 1)
            ),
        ),
        # C-style: return_type name(params) {
        # Matches: C, C++, C#, Java, PHP, etc.
        (
            "c_style",
            re.compile(
                r"^[\s]*"  # leading whitespace
                r"(?:[\w\*&<>,:\s]+\s+)"  # return type with modifiers/pointers
                r"(\w+)"  # function name (capture group 1)
                r"\s*\([^;]*$"  # opening paren, avoid forward declarations ending in ;
            ),
        ),
        # Shell/Bash: function name() { or name() {
        (
            "shell",
            re.compile(
                r"^[\s]*"  # leading whitespace
                r"(?:function\s+)?"  # optional 'function' keyword
                r"(\w+)"  # function name
                r"\s*\(\s*\)"  # empty parentheses
            ),
        ),
        # SQL: CREATE FUNCTION/PROCEDURE/TRIGGER name
        (
            "sql",
            re.compile(
                r"^[\s]*"  # leading whitespace
                r"(?:CREATE\s+(?:OR\s+REPLACE\s+)?)?"  # optional CREATE OR REPLACE
                r"(?:PROCEDURE|FUNCTION|TRIGGER)\s+"  # object type
                r"(?:[\w.]+\s+)?"  # optional schema prefix
                r"(\w+)",  # name
                re.IGNORECASE,
            ),
        ),
    ]

    # Map file extensions to preferred pattern order
    EXTENSION_PRIORITY: dict[str, list[str]] = {
        ".py": ["keyword"],  # Python uses 'def'
        ".rb": ["keyword"],  # Ruby uses 'def'
        ".go": ["keyword"],  # Go uses 'func'
        ".rs": ["keyword"],  # Rust uses 'fn'
        ".js": ["keyword", "c_style"],  # JavaScript can use both
        ".ts": ["keyword", "c_style"],  # TypeScript
        ".c": ["c_style"],  # C uses c_style
        ".cpp": ["c_style"],  # C++
        ".cc": ["c_style"],
        ".cxx": ["c_style"],
        ".h": ["c_style"],
        ".hpp": ["c_style"],
        ".cs": ["c_style"],  # C#
        ".java": ["c_style"],  # Java
        ".php": ["c_style", "keyword"],  # PHP uses both
        ".sh": ["shell", "keyword"],  # Shell
        ".bash": ["shell", "keyword"],
        ".zsh": ["shell", "keyword"],
        ".sql": ["sql"],  # SQL
    }

    def __init__(self, repo_path: Path) -> None:
        """Initialize the extractor.

        Args:
            repo_path: Path to the repository root (for context)
        """
        self.repo_path = repo_path
        self._rg_available = shutil.which("rg") is not None

    def _get_patterns_for_file(self, file_path: Path) -> list[tuple[str, re.Pattern]]:
        """Get definition patterns ordered by priority for the file extension."""
        ext = file_path.suffix.lower()
        priority_order = self.EXTENSION_PRIORITY.get(ext, [])

        # Build ordered list based on priority
        ordered = []
        for pattern_name in priority_order:
            for name, pattern in self.DEFINITION_PATTERNS:
                if name == pattern_name and (name, pattern) not in ordered:
                    ordered.append((name, pattern))

        # Add remaining patterns not in priority list
        for name, pattern in self.DEFINITION_PATTERNS:
            if (name, pattern) not in ordered:
                ordered.append((name, pattern))

        return ordered

    def _find_definitions(
        self,
        file_path: Path,
    ) -> list[tuple[int, str, str]]:
        """Find all function definition lines in a file using ripgrep.

        Args:
            file_path: Path to the source file

        Returns:
            List of (line_number, function_name, full_line) tuples, sorted by line_number
        """
        if not self._rg_available:
            logger.debug("ripgrep not available, skipping definition search")
            return []

        patterns = self._get_patterns_for_file(file_path)
        if not patterns:
            return []

        # Combine patterns with alternation for single ripgrep call
        # Extract just the pattern regexes
        pattern_regexes = [p[1].pattern for p in patterns]
        combined_pattern = "|".join(f"({p})" for p in pattern_regexes)

        definitions: list[tuple[int, str, str]] = []

        try:
            cmd = [
                "rg",
                "--line-number",
                "--no-heading",
                "--with-filename",
                "--color=never",
                "--multiline",  # Handle multi-line patterns
                combined_pattern,
                str(file_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.repo_path),
            )

            # ripgrep returns 1 when no matches found (not an error)
            if result.returncode > 1:
                logger.debug(f"ripgrep search failed: {result.stderr}")
                return []

            # Parse results: file:line:content
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                # Parse: filepath:line:content
                # Handle Windows/Unix path differences
                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue

                try:
                    line_num = int(parts[1])
                except ValueError:
                    continue

                content = parts[2]

                # Try each pattern to extract function name
                for pattern_name, pattern in patterns:
                    match = pattern.match(content)
                    if match:
                        func_name = match.group(1) if match.lastindex else None
                        if func_name:
                            definitions.append((line_num, func_name, content.strip()))
                            break  # Found a match, move to next line

        except subprocess.TimeoutExpired:
            logger.warning(f"ripgrep search timed out for {file_path}")
        except Exception as e:
            logger.debug(f"ripgrep search failed for {file_path}: {e}")

        # Sort by line number and remove duplicates
        seen = set()
        unique_defs = []
        for line_num, func_name, full_line in sorted(definitions):
            key = (line_num, func_name)
            if key not in seen:
                seen.add(key)
                unique_defs.append((line_num, func_name, full_line))

        return unique_defs

    def _derive_boundaries(
        self,
        definitions: list[tuple[int, str, str]],
        total_lines: int,
    ) -> list[tuple[int, int, str, str]]:
        """Derive function boundaries from definition lines.

        Each function spans from its definition line to the line before
        the next function's definition (or EOF).

        Args:
            definitions: List of (line_number, function_name, full_line)
            total_lines: Total number of lines in the file

        Returns:
            List of (start_line, end_line, function_name, signature_line)
        """
        boundaries = []
        for i, (line_num, func_name, full_line) in enumerate(definitions):
            # End is line before next definition, or EOF
            if i + 1 < len(definitions):
                end_line = definitions[i + 1][0] - 1
            else:
                end_line = total_lines

            boundaries.append((line_num, end_line, func_name, full_line))

        return boundaries

    def _count_file_lines(self, file_path: Path) -> int:
        """Count total lines in a file efficiently."""
        try:
            # Use wc -l for efficiency on large files
            result = subprocess.run(
                ["wc", "-l", str(file_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Parse: " 123 filename"
                parts = result.stdout.strip().split()
                if parts:
                    return int(parts[0])
        except Exception:
            pass

        # Fallback: read and count
        try:
            with open(file_path, "rb") as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.debug(f"Failed to count lines in {file_path}: {e}")
            return 0

    def _extract_parameters(self, signature_line: str) -> list[str]:
        """Extract parameter names from a function signature line.

        Best-effort extraction for common patterns.
        """
        params: list[str] = []

        # Look for parentheses with content
        match = re.search(r"\(([^)]*)\)", signature_line)
        if not match:
            return params

        params_str = match.group(1).strip()
        if not params_str:
            return params

        # Split by comma, handle simple cases
        for param in params_str.split(","):
            param = param.strip()
            if not param:
                continue

            # Extract parameter name (last word before = or end)
            # Examples: "int x", "const string& name", "x int"
            param = param.split("=")[0].strip()  # Remove default values
            words = param.split()
            if words:
                # Last word is usually the parameter name
                name = words[-1].rstrip("*&")
                if re.match(r"^[a-zA-Z_]\w*$", name):
                    params.append(name)

        return params

    def extract_functions(
        self,
        file_path: Path,
        changed_line_ranges: list[tuple[int, int]],
    ) -> list[FunctionInfo]:
        """Extract functions affected by the changed line ranges.

        Args:
            file_path: Path to the source file
            changed_line_ranges: List of (start, end) tuples for changed lines

        Returns:
            List of FunctionInfo for functions that overlap with changed lines
        """
        if not file_path.exists():
            logger.debug(f"File not found: {file_path}")
            return []

        if not changed_line_ranges:
            logger.debug("No changed line ranges provided")
            return []

        # Step 1: Find all function definitions
        definitions = self._find_definitions(file_path)
        if not definitions:
            logger.debug(f"No function definitions found in {file_path}")
            return []

        # Step 2: Get total lines and derive boundaries
        total_lines = self._count_file_lines(file_path)
        boundaries = self._derive_boundaries(definitions, total_lines)

        # Step 3: Intersect with changed ranges
        affected_functions: list[FunctionInfo] = []
        seen_names: set[str] = set()

        for func_start, func_end, func_name, signature_line in boundaries:
            # Check if this function overlaps with any changed range
            overlaps = False
            for change_start, change_end in changed_line_ranges:
                # Overlap condition: func_start <= change_end AND func_end >= change_start
                if func_start <= change_end and func_end >= change_start:
                    overlaps = True
                    break

            if overlaps and func_name not in seen_names:
                params = self._extract_parameters(signature_line)

                # Build return_type from signature (best effort)
                return_type = None
                # Try to extract return type from C-style signatures
                c_match = re.match(
                    r"^[\s]*([\w\*&<>,:\s]+)\s+\w+\s*\(", signature_line
                )
                if c_match:
                    potential = c_match.group(1).strip()
                    # Filter out modifiers
                    modifiers = {"static", "inline", "extern", "virtual", "const", "async", "public", "private", "protected"}
                    words = potential.split()
                    filtered = [w for w in words if w not in modifiers]
                    if filtered:
                        return_type = " ".join(filtered)

                affected_functions.append(
                    FunctionInfo(
                        name=func_name,
                        file=str(file_path),
                        line_start=func_start,
                        line_end=func_end,
                        parameters=params,
                        return_type=return_type,
                        is_method=False,  # Cannot determine from single line
                        receiver_type=None,
                        docstring=None,
                    )
                )
                seen_names.add(func_name)

        logger.debug(
            f"Found {len(affected_functions)} affected functions in {file_path}"
        )
        return affected_functions
