"""Ripgrep integration for fast code search and verification."""

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from ripgrep."""

    file: str
    line_number: int
    line_content: str
    match_text: str

    @property
    def location(self) -> str:
        """Get file:line format."""
        return f"{self.file}:{self.line_number}"


class RipgrepSearch:
    """Fast code search using ripgrep (rg).

    Falls back to grep if ripgrep is not available.
    """

    def __init__(self, repo_path: str) -> None:
        """Initialize with repository path.

        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = Path(repo_path)
        self._rg_available = shutil.which("rg") is not None
        self._grep_available = shutil.which("grep") is not None

    @property
    def available(self) -> bool:
        """Check if search is available."""
        return self._rg_available or self._grep_available

    def find_function_usages(
        self,
        function_name: str,
        file_patterns: list[str] | None = None,
        exclude_file: str | None = None,
    ) -> list[SearchResult]:
        """Find all usages of a function in the codebase.

        Args:
            function_name: Name of the function to search for
            file_patterns: Optional glob patterns to limit search (e.g., ["*.go", "*.py"])
            exclude_file: File to exclude from results (typically the definition file)

        Returns:
            List of search results showing where function is used
        """
        # Build pattern to match function calls/references
        # Match: functionName( or .functionName( or functionName, etc.
        pattern = rf"\b{re.escape(function_name)}\s*\("

        results = self._search(pattern, file_patterns)

        # Filter out the definition file if specified
        if exclude_file:
            results = [r for r in results if not r.file.endswith(exclude_file)]

        return results

    def find_type_usages(
        self,
        type_name: str,
        file_patterns: list[str] | None = None,
    ) -> list[SearchResult]:
        """Find all usages of a type in the codebase.

        Args:
            type_name: Name of the type to search for
            file_patterns: Optional glob patterns to limit search

        Returns:
            List of search results
        """
        # Match type name as a word (not part of another word)
        pattern = rf"\b{re.escape(type_name)}\b"
        return self._search(pattern, file_patterns)

    def find_imports_of(
        self,
        module_or_package: str,
        file_patterns: list[str] | None = None,
    ) -> list[SearchResult]:
        """Find all files that import a module or package.

        Args:
            module_or_package: Module or package name to search for
            file_patterns: Optional glob patterns

        Returns:
            List of search results showing import statements
        """
        # Match various import styles
        # Python: from X import, import X
        # Go: import "X", import X "path"
        # JS/TS: import * from "X", require("X")
        pattern = rf'(import|from|require).*["\']?{re.escape(module_or_package)}["\']?'
        return self._search(pattern, file_patterns)

    def find_callers(
        self,
        function_name: str,
        defining_file: str,
        language: str = "auto",
    ) -> list[SearchResult]:
        """Find all files that call a specific function.

        This is a higher-level method that uses language-appropriate patterns.

        Args:
            function_name: Name of the function
            defining_file: File where function is defined (will be excluded)
            language: Programming language ("go", "python", "typescript", "auto")

        Returns:
            List of search results showing callers
        """
        # Determine file patterns based on language
        patterns_map = {
            "go": ["*.go"],
            "python": ["*.py"],
            "typescript": ["*.ts", "*.tsx", "*.js", "*.jsx"],
            "javascript": ["*.js", "*.jsx", "*.ts", "*.tsx"],
            "rust": ["*.rs"],
        }

        if language == "auto":
            # Auto-detect from defining file extension
            ext = Path(defining_file).suffix.lstrip(".")
            lang_map = {
                "go": "go",
                "py": "python",
                "ts": "typescript",
                "tsx": "typescript",
                "js": "javascript",
                "jsx": "javascript",
                "rs": "rust",
            }
            language = lang_map.get(ext, "")

        file_patterns = patterns_map.get(language)
        return self.find_function_usages(function_name, file_patterns, defining_file)

    def search_literal(
        self,
        text: str,
        file_patterns: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for literal text (not regex).

        Args:
            text: Exact text to search for
            file_patterns: Optional glob patterns

        Returns:
            List of search results
        """
        return self._search(re.escape(text), file_patterns, fixed_string=True)

    def _search(
        self,
        pattern: str,
        file_patterns: list[str] | None = None,
        fixed_string: bool = False,
    ) -> list[SearchResult]:
        """Execute search using ripgrep or grep.

        Args:
            pattern: Regex pattern to search for
            file_patterns: Optional glob patterns to limit search
            fixed_string: If True, treat pattern as literal text

        Returns:
            List of search results
        """
        if self._rg_available:
            return self._search_with_rg(pattern, file_patterns, fixed_string)
        elif self._grep_available:
            return self._search_with_grep(pattern, file_patterns, fixed_string)
        else:
            logger.warning("Neither ripgrep nor grep available")
            return []

    def _search_with_rg(
        self,
        pattern: str,
        file_patterns: list[str] | None = None,
        fixed_string: bool = False,
    ) -> list[SearchResult]:
        """Search using ripgrep."""
        cmd = [
            "rg",
            "--line-number",  # Include line numbers
            "--no-heading",   # Don't group by file
            "--with-filename",  # Always show filename
            "--color=never",  # No color codes
        ]

        if fixed_string:
            cmd.append("--fixed-strings")

        # Add file type filters
        if file_patterns:
            for fp in file_patterns:
                cmd.extend(["--glob", fp])

        cmd.append(pattern)
        cmd.append(str(self.repo_path))

        return self._run_search(cmd, pattern)

    def _search_with_grep(
        self,
        pattern: str,
        file_patterns: list[str] | None = None,
        fixed_string: bool = False,
    ) -> list[SearchResult]:
        """Search using grep (fallback)."""
        cmd = [
            "grep",
            "-r",  # Recursive
            "-n",  # Line numbers
            "-H",  # Always show filename
        ]

        if fixed_string:
            cmd.append("-F")
        else:
            cmd.append("-E")  # Extended regex

        # Note: grep doesn't have nice glob filtering, skip for simplicity
        cmd.append(pattern)
        cmd.append(str(self.repo_path))

        return self._run_search(cmd, pattern)

    def _run_search(self, cmd: list[str], pattern: str) -> list[SearchResult]:
        """Run search command and parse results."""
        results: list[SearchResult] = []

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=str(self.repo_path),
            )

            # ripgrep returns 1 for no matches (not an error)
            if process.returncode > 1:
                logger.warning(f"Search command failed: {process.stderr}")
                return []

            # Parse output: file:line:content
            for line in process.stdout.strip().split("\n"):
                if not line:
                    continue

                # Handle Windows/Unix path differences
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_path = parts[0]
                    try:
                        line_num = int(parts[1])
                    except ValueError:
                        continue
                    content = parts[2] if len(parts) > 2 else ""

                    # Make path relative to repo
                    if file_path.startswith(str(self.repo_path)):
                        file_path = file_path[len(str(self.repo_path)):].lstrip("/\\")

                    # Extract the actual match
                    match = re.search(pattern, content)
                    match_text = match.group() if match else ""

                    results.append(SearchResult(
                        file=file_path,
                        line_number=line_num,
                        line_content=content.strip(),
                        match_text=match_text,
                    ))

            logger.debug(f"Search found {len(results)} results for pattern: {pattern}")
            return results

        except subprocess.TimeoutExpired:
            logger.error("Search timed out after 30 seconds")
            return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
