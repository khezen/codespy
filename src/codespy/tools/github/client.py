"""GitHub API client for fetching PR data."""

import logging
import re
from pathlib import Path

from git import Repo
from github import Auth, Github
from github.PullRequest import PullRequest as GHPullRequest

from codespy.tools.parsers.ripgrep import RipgrepSearch
from codespy.tools.parsers.treesitter import TreeSitterAnalyzer
from codespy.config import Settings, get_settings
from codespy.tools.github.models import CallerInfo, ChangedFile, FileStatus, PullRequest, ReviewContext

logger = logging.getLogger(__name__)


class GitHubClient:
    """Client for interacting with GitHub API."""

    PR_URL_PATTERN = re.compile(
        r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
    )

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the GitHub client.

        Args:
            settings: Application settings. Uses global settings if not provided.
        """
        self.settings = settings or get_settings()
        self._github: Github | None = None

    @property
    def github(self) -> Github:
        """Get or create GitHub client instance."""
        if self._github is None:
            if self.settings.github_token:
                auth = Auth.Token(self.settings.github_token)
                self._github = Github(auth=auth)
            else:
                self._github = Github()
        return self._github

    def parse_pr_url(self, url: str) -> tuple[str, str, int]:
        """Parse a GitHub PR URL into owner, repo, and PR number.

        Args:
            url: GitHub PR URL

        Returns:
            Tuple of (owner, repo, pr_number)

        Raises:
            ValueError: If URL is not a valid GitHub PR URL
        """
        match = self.PR_URL_PATTERN.match(url)
        if not match:
            raise ValueError(
                f"Invalid GitHub PR URL: {url}. "
                "Expected format: https://github.com/owner/repo/pull/123"
            )
        return match.group("owner"), match.group("repo"), int(match.group("number"))

    def _is_excluded_path(self, filepath: str) -> tuple[bool, str]:
        """Check if a file path matches any exclusion pattern.

        Args:
            filepath: The file path to check

        Returns:
            Tuple of (is_excluded, matched_pattern)
        """
        if self.settings.include_vendor:
            return False, ""

        for pattern in self.settings.exclude_patterns:
            # Check if pattern appears anywhere in the path
            if pattern in filepath:
                return True, pattern
            # Also check if path starts with pattern (for patterns without trailing slash)
            if filepath.startswith(pattern.rstrip("/")):
                return True, pattern

        return False, ""

    def fetch_pull_request(self, pr_url: str) -> PullRequest:
        """Fetch pull request data from GitHub.

        Args:
            pr_url: GitHub PR URL

        Returns:
            PullRequest model with all data
        """
        owner, repo_name, pr_number = self.parse_pr_url(pr_url)

        # Get repository and PR
        repo = self.github.get_repo(f"{owner}/{repo_name}")
        gh_pr: GHPullRequest = repo.get_pull(pr_number)

        # Build changed files list, filtering excluded paths
        changed_files: list[ChangedFile] = []
        excluded_count = 0
        excluded_patterns_matched: set[str] = set()

        for file in gh_pr.get_files():
            # Check if file should be excluded
            is_excluded, matched_pattern = self._is_excluded_path(file.filename)
            if is_excluded:
                excluded_count += 1
                excluded_patterns_matched.add(matched_pattern)
                continue

            status = FileStatus(file.status)

            # Get file content if available
            content = None
            previous_content = None

            if status != FileStatus.REMOVED:
                try:
                    content_file = repo.get_contents(file.filename, ref=gh_pr.head.sha)
                    if not isinstance(content_file, list):
                        content = content_file.decoded_content.decode("utf-8")
                except Exception:
                    pass  # File might be binary or too large

            if status in (FileStatus.MODIFIED, FileStatus.RENAMED):
                try:
                    prev_filename = file.previous_filename or file.filename
                    prev_content_file = repo.get_contents(prev_filename, ref=gh_pr.base.sha)
                    if not isinstance(prev_content_file, list):
                        previous_content = prev_content_file.decoded_content.decode("utf-8")
                except Exception:
                    pass

            changed_files.append(
                ChangedFile(
                    filename=file.filename,
                    status=status,
                    additions=file.additions,
                    deletions=file.deletions,
                    patch=file.patch,
                    previous_filename=file.previous_filename,
                    content=content,
                    previous_content=previous_content,
                )
            )

        # Log exclusion summary
        if excluded_count > 0:
            patterns_str = ", ".join(sorted(excluded_patterns_matched))
            logger.info(
                f"Excluded {excluded_count} files matching: {patterns_str} "
                "(use --include-vendor to include)"
            )

        return PullRequest(
            number=gh_pr.number,
            title=gh_pr.title,
            body=gh_pr.body,
            state=gh_pr.state,
            author=gh_pr.user.login,
            base_branch=gh_pr.base.ref,
            head_branch=gh_pr.head.ref,
            base_sha=gh_pr.base.sha,
            head_sha=gh_pr.head.sha,
            created_at=gh_pr.created_at,
            updated_at=gh_pr.updated_at,
            repo_owner=owner,
            repo_name=repo_name,
            changed_files=changed_files,
            labels=[label.name for label in gh_pr.labels],
            excluded_files_count=excluded_count,
        )

    def clone_repository(self, owner: str, repo_name: str, ref: str) -> Path:
        """Clone or update a repository for context analysis.

        Args:
            owner: Repository owner
            repo_name: Repository name
            ref: Git ref (branch, tag, or commit) to checkout

        Returns:
            Path to the cloned repository
        """
        cache_dir = self.settings.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        repo_dir = cache_dir / owner / repo_name

        if repo_dir.exists():
            # Update existing clone
            repo = Repo(repo_dir)
            repo.remotes.origin.fetch()
        else:
            # Fresh clone (shallow to save space)
            repo_url = f"https://github.com/{owner}/{repo_name}.git"
            if self.settings.github_token:
                repo_url = f"https://{self.settings.github_token}@github.com/{owner}/{repo_name}.git"

            repo_dir.mkdir(parents=True, exist_ok=True)
            repo = Repo.clone_from(
                repo_url,
                repo_dir,
                depth=1,
                no_single_branch=True,
            )

        # Checkout the specific ref
        repo.git.checkout(ref)

        return repo_dir

    def build_review_context(
        self,
        pr: PullRequest,
        include_repo_context: bool = True,
    ) -> ReviewContext:
        """Build review context for a pull request.

        Args:
            pr: The pull request to build context for
            include_repo_context: Whether to include related files from the repo

        Returns:
            ReviewContext with PR and related files
        """
        related_files: dict[str, str] = {}
        repo_structure: str | None = None
        callers: dict[str, list[CallerInfo]] = {}

        if include_repo_context:
            try:
                repo_dir = self.clone_repository(pr.repo_owner, pr.repo_name, pr.head_sha)

                # Get repository structure (top-level overview)
                repo_structure = self._get_repo_structure(repo_dir)

                # Find related files (imports/dependencies)
                for changed_file in pr.code_files:
                    file_path = repo_dir / changed_file.filename
                    if file_path.exists():
                        imports = self._find_imports(file_path, changed_file.extension)
                        for import_path in imports:
                            full_path = repo_dir / import_path
                            if full_path.exists() and import_path not in related_files:
                                try:
                                    content = full_path.read_text()
                                    # Limit size per file
                                    if len(content) < 10000:
                                        related_files[import_path] = content
                                except Exception:
                                    pass

                # Respect max context size
                self._trim_context(related_files, self.settings.max_context_size)

                # Find callers of changed functions using ripgrep
                callers = self._find_callers_for_pr(pr, repo_dir)

            except Exception as e:
                # If cloning fails, continue without repo context
                logger.debug(f"Failed to build full context: {e}")

        return ReviewContext(
            pull_request=pr,
            related_files=related_files,
            repository_structure=repo_structure,
            callers=callers,
        )

    def _find_callers_for_pr(
        self,
        pr: PullRequest,
        repo_dir: Path,
    ) -> dict[str, list[CallerInfo]]:
        """Find callers of functions changed in the PR.

        Uses tree-sitter for accurate AST-based analysis when available,
        falls back to ripgrep for text-based search.

        Args:
            pr: The pull request
            repo_dir: Path to the cloned repository

        Returns:
            Dictionary mapping filenames to lists of CallerInfo
        """
        callers: dict[str, list[CallerInfo]] = {}

        # Try tree-sitter first for accurate analysis
        ts_analyzer = TreeSitterAnalyzer(repo_dir)
        rg_search = RipgrepSearch(repo_dir)

        use_treesitter = ts_analyzer.available
        if use_treesitter:
            logger.debug("Using tree-sitter for caller analysis")
        else:
            logger.debug("Tree-sitter unavailable, using ripgrep fallback")

        for changed_file in pr.code_files:
            if not changed_file.patch:
                continue

            # Extract function names from the diff
            function_names = self._extract_changed_functions(
                changed_file.patch,
                changed_file.extension,
            )

            if not function_names:
                continue

            file_callers: list[CallerInfo] = []

            for func_name in function_names:
                if use_treesitter:
                    # Use tree-sitter for accurate AST-based search
                    callers_found = self._find_callers_treesitter(
                        ts_analyzer, repo_dir, func_name, changed_file.filename, changed_file.extension
                    )
                    file_callers.extend(callers_found)
                else:
                    # Fallback to ripgrep text search
                    file_pattern = f"*.{changed_file.extension}" if changed_file.extension else None
                    file_patterns = [file_pattern] if file_pattern else None
                    results = rg_search.find_function_usages(func_name, file_patterns)

                    for result in results:
                        # Skip if the result is in the same file (definition, not caller)
                        if result.file == changed_file.filename:
                            continue

                        file_callers.append(
                            CallerInfo(
                                file=result.file,
                                line_number=result.line_number,
                                line_content=result.line_content,
                                function_name=func_name,
                            )
                        )

            if file_callers:
                callers[changed_file.filename] = file_callers
                logger.debug(
                    f"Found {len(file_callers)} callers for functions in {changed_file.filename}"
                )

        return callers

    def _find_callers_treesitter(
        self,
        analyzer: TreeSitterAnalyzer,
        repo_dir: Path,
        function_name: str,
        source_file: str,
        extension: str,
    ) -> list[CallerInfo]:
        """Find callers using tree-sitter AST analysis.

        Args:
            analyzer: Tree-sitter analyzer instance
            repo_dir: Repository directory
            function_name: Function name to find calls for
            source_file: File where function is defined (to exclude)
            extension: File extension to search

        Returns:
            List of CallerInfo for each call found
        """
        callers: list[CallerInfo] = []

        # Find all files with matching extension
        pattern = f"**/*.{extension}"
        for file_path in repo_dir.glob(pattern):
            # Skip the source file
            rel_path = str(file_path.relative_to(repo_dir))
            if rel_path == source_file:
                continue

            # Skip vendor/node_modules directories
            if any(skip in str(file_path) for skip in ["vendor/", "node_modules/", "__pycache__/"]):
                continue

            # Find calls to this function in the file
            calls = analyzer.find_function_calls(file_path, function_name)

            for call in calls:
                callers.append(
                    CallerInfo(
                        file=rel_path,
                        line_number=call.line_number,
                        line_content=call.line_content,
                        function_name=function_name,
                    )
                )

        return callers

    def _extract_changed_functions(self, patch: str, extension: str) -> list[str]:
        """Extract function/method names that were modified in a diff.

        Args:
            patch: The diff patch
            extension: File extension

        Returns:
            List of function names that were changed
        """
        function_names: list[str] = []

        # Patterns for different languages
        if extension == "py":
            # Python: def function_name( or async def function_name(
            pattern = re.compile(r"^[+-]\s*(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
        elif extension == "go":
            # Go: func FunctionName( or func (receiver) MethodName(
            pattern = re.compile(r"^[+-]\s*func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", re.MULTILINE)
        elif extension in ("js", "ts", "jsx", "tsx"):
            # JavaScript/TypeScript: function name(, const name = (, name(, async name(
            pattern = re.compile(
                r"^[+-]\s*(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|"
                r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>|"
                r"(\w+)\s*\([^)]*\)\s*\{)",
                re.MULTILINE,
            )
        elif extension in ("java", "kt", "cs"):
            # Java/Kotlin/C#: public void methodName( or fun methodName(
            pattern = re.compile(
                r"^[+-]\s*(?:public|private|protected|internal|override|suspend|fun|static|\s)*"
                r"(?:\w+\s+)*(\w+)\s*\(",
                re.MULTILINE,
            )
        elif extension in ("c", "cpp", "h", "hpp"):
            # C/C++: type function_name( or Class::method_name(
            pattern = re.compile(
                r"^[+-]\s*(?:\w+\s+)+(?:\w+::)?(\w+)\s*\(",
                re.MULTILINE,
            )
        elif extension == "rs":
            # Rust: fn function_name( or pub fn function_name(
            pattern = re.compile(r"^[+-]\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*[<(]", re.MULTILINE)
        elif extension == "rb":
            # Ruby: def method_name
            pattern = re.compile(r"^[+-]\s*def\s+(\w+)", re.MULTILINE)
        elif extension == "swift":
            # Swift: func functionName( or static func, private func, etc.
            pattern = re.compile(
                r"^[+-]\s*(?:@\w+\s+)*(?:public|private|internal|fileprivate|open|static|class|override|\s)*"
                r"func\s+(\w+)\s*[<(]",
                re.MULTILINE,
            )
        elif extension in ("m", "mm"):
            # Objective-C: - (type)methodName or + (type)methodName
            pattern = re.compile(
                r"^[+-]\s*[-+]\s*\([^)]+\)\s*(\w+)",
                re.MULTILINE,
            )
        else:
            return []

        for match in pattern.finditer(patch):
            # Get the first non-None group
            for group in match.groups():
                if group:
                    # Skip common false positives
                    if group not in ("if", "for", "while", "switch", "catch", "return"):
                        function_names.append(group)
                    break

        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in function_names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)

        return unique_names[:20]  # Limit to 20 functions per file

    def _get_repo_structure(self, repo_dir: Path, max_depth: int = 2) -> str:
        """Get a tree-like overview of the repository structure.

        Args:
            repo_dir: Path to the repository
            max_depth: Maximum depth to traverse

        Returns:
            String representation of the repo structure
        """
        lines = []

        def walk(path: Path, prefix: str = "", depth: int = 0) -> None:
            if depth > max_depth:
                return

            # Skip hidden and common non-essential directories
            skip_dirs = {
                ".git",
                "node_modules",
                "__pycache__",
                ".venv",
                "venv",
                ".tox",
                "dist",
                "build",
                ".eggs",
            }

            try:
                entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
                dirs = [e for e in entries if e.is_dir() and e.name not in skip_dirs]
                files = [e for e in entries if e.is_file()]

                # Show directories
                for d in dirs[:10]:  # Limit directories shown
                    lines.append(f"{prefix}📁 {d.name}/")
                    walk(d, prefix + "  ", depth + 1)

                if len(dirs) > 10:
                    lines.append(f"{prefix}... and {len(dirs) - 10} more directories")

                # Show files at top level only
                if depth == 0:
                    for f in files[:5]:
                        lines.append(f"{prefix}📄 {f.name}")
                    if len(files) > 5:
                        lines.append(f"{prefix}... and {len(files) - 5} more files")

            except PermissionError:
                pass

        walk(repo_dir)
        return "\n".join(lines[:50])  # Limit total lines

    def _find_imports(self, file_path: Path, extension: str) -> list[str]:
        """Find import statements in a file and resolve to file paths.

        Args:
            file_path: Path to the source file
            extension: File extension

        Returns:
            List of relative file paths that are imported
        """
        imports: list[str] = []

        try:
            content = file_path.read_text()
        except Exception:
            return imports

        if extension == "py":
            imports.extend(self._parse_python_imports(content, file_path))
        elif extension in ("js", "ts", "jsx", "tsx"):
            imports.extend(self._parse_js_imports(content, file_path))
        elif extension == "go":
            imports.extend(self._parse_go_imports(content, file_path))

        return imports

    def _parse_python_imports(self, content: str, file_path: Path) -> list[str]:
        """Parse Python import statements."""
        imports: list[str] = []
        import_pattern = re.compile(
            r"^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE
        )

        for match in import_pattern.finditer(content):
            module = match.group(1) or match.group(2)
            if module:
                # Convert module path to file path
                parts = module.split(".")
                # Try to find the file relative to the source file
                for i in range(len(parts), 0, -1):
                    potential_path = "/".join(parts[:i]) + ".py"
                    imports.append(potential_path)
                    # Also try as package
                    imports.append("/".join(parts[:i]) + "/__init__.py")

        return imports[:10]  # Limit number of imports to check

    def _parse_js_imports(self, content: str, file_path: Path) -> list[str]:
        """Parse JavaScript/TypeScript import statements."""
        imports: list[str] = []
        # Match: import ... from '...' or require('...')
        import_pattern = re.compile(
            r"(?:import\s+.*?\s+from\s+['\"](.+?)['\"]|require\s*\(\s*['\"](.+?)['\"]\s*\))"
        )

        for match in import_pattern.finditer(content):
            path = match.group(1) or match.group(2)
            if path and path.startswith("."):
                # Relative import
                resolved = (file_path.parent / path).resolve()
                # Try different extensions
                for ext in ["", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"]:
                    imports.append(str(resolved) + ext)

        return imports[:10]

    def _parse_go_imports(self, content: str, file_path: Path) -> list[str]:
        """Parse Go import statements."""
        imports: list[str] = []
        # Match import blocks and single imports
        import_pattern = re.compile(r'import\s+(?:\(\s*([\s\S]*?)\s*\)|"(.+?)")')

        for match in import_pattern.finditer(content):
            if match.group(1):
                # Multi-line import block
                for line in match.group(1).split("\n"):
                    line = line.strip().strip('"')
                    if line and not line.startswith("//"):
                        imports.append(line)
            elif match.group(2):
                imports.append(match.group(2))

        return imports[:10]

    def _trim_context(self, related_files: dict[str, str], max_size: int) -> None:
        """Trim related files to fit within max context size.

        Modifies the dict in place, removing files if total size exceeds limit.
        """
        total_size = sum(len(content) for content in related_files.values())

        if total_size <= max_size:
            return

        # Sort by size (keep smaller files)
        sorted_files = sorted(related_files.items(), key=lambda x: len(x[1]))

        current_size = 0
        files_to_keep: set[str] = set()

        for filename, content in sorted_files:
            if current_size + len(content) <= max_size:
                files_to_keep.add(filename)
                current_size += len(content)
            else:
                break

        # Remove files that don't fit
        for filename in list(related_files.keys()):
            if filename not in files_to_keep:
                del related_files[filename]
