"""Tests for scope_resolver module."""

import tempfile
from pathlib import Path

import pytest

from codespy.agents.reviewer.models import ScopeType
from codespy.agents.reviewer.modules.scope_resolver import (
    ScopeResolver,
    derive_sparse_paths,
)
from codespy.tools.git.models import ChangedFile, FileStatus


class TestDeriveSparsePaths:
    """Test sparse path derivation."""

    def test_single_scope_indicator(self):
        """Test deriving sparse paths with single scope indicator."""
        changed_files = ["packages/auth/src/index.ts"]
        paths = derive_sparse_paths(changed_files)
        assert "packages/auth/" in paths
        assert "/*" in paths

    def test_multiple_scope_indicators(self):
        """Test deriving sparse paths with multiple indicators."""
        changed_files = [
            "mono/svc/api/cmd/main.go",
            "packages/auth/src/index.ts",
        ]
        paths = derive_sparse_paths(changed_files)
        assert "mono/svc/api/" in paths
        assert "packages/auth/" in paths
        assert "/*" in paths

    def test_no_scope_indicator_uses_depth_fallback(self):
        """Test fallback to depth-2 when no indicator found."""
        changed_files = ["backend/api/handler.go"]
        paths = derive_sparse_paths(changed_files)
        assert "backend/api/" in paths

    def test_root_level_file(self):
        """Test root-level files don't add extra paths."""
        changed_files = ["README.md", ".github/workflows/ci.yml"]
        paths = derive_sparse_paths(changed_files)
        # Should still have /* for root manifests
        assert "/*" in paths

    def test_deduplication(self):
        """Test that duplicate scope roots are deduplicated."""
        changed_files = [
            "packages/auth/src/index.ts",
            "packages/auth/src/utils.ts",
            "packages/auth/tests/auth.test.ts",
        ]
        paths = derive_sparse_paths(changed_files)
        # Should only have one packages/auth/
        assert paths.count("packages/auth/") == 1


class TestScopeResolver:
    """Test ScopeResolver class."""

    def test_discover_manifests(self):
        """Test manifest discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake repo structure
            repo_path = Path(tmpdir)
            (repo_path / "packages" / "auth").mkdir(parents=True)
            (repo_path / "packages" / "auth" / "package.json").touch()
            (repo_path / "services" / "api").mkdir(parents=True)
            (repo_path / "services" / "api" / "go.mod").touch()

            resolver = ScopeResolver()
            manifests = resolver._discover_manifests(repo_path, [])

            assert len(manifests) == 2
            assert Path("packages/auth") in manifests
            assert Path("services/api") in manifests

    def test_single_scope_repo(self):
        """Test single-scope repo with root manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "go.mod").touch()
            (repo_path / "main.go").touch()

            changed_files = [
                ChangedFile(filename="main.go", status=FileStatus.MODIFIED),
            ]

            resolver = ScopeResolver()
            scopes, orphans = resolver._resolve(repo_path, changed_files, "owner/repo")

            assert len(scopes) == 1
            assert scopes[0].subroot == "."
            assert len(orphans) == 0

    def test_mono_repo_with_packages(self):
        """Test mono-repo with packages/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "packages" / "auth" / "src").mkdir(parents=True)
            (repo_path / "packages" / "auth" / "package.json").touch()
            (repo_path / "packages" / "utils").mkdir(parents=True)
            (repo_path / "packages" / "utils" / "package.json").touch()

            changed_files = [
                ChangedFile(filename="packages/auth/src/index.ts", status=FileStatus.MODIFIED),
            ]

            resolver = ScopeResolver()
            scopes, orphans = resolver._resolve(repo_path, changed_files, "owner/repo")

            assert len(scopes) == 1
            assert scopes[0].subroot == "packages/auth"

    def test_orphan_file_no_manifest(self):
        """Test file with no manifest becomes orphan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            # No manifest files

            changed_files = [
                ChangedFile(filename="random/file.py", status=FileStatus.MODIFIED),
            ]

            resolver = ScopeResolver()
            scopes, orphans = resolver._resolve(repo_path, changed_files, "owner/repo")

            # Should have orphan since no manifest and no scope indicator
            assert len(orphans) == 1

    def test_lock_file_detection(self):
        """Test that lock file changes are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "package.json").touch()
            (repo_path / "package-lock.json").touch()

            changed_files = [
                ChangedFile(filename="package-lock.json", status=FileStatus.MODIFIED),
            ]

            resolver = ScopeResolver()
            scopes, orphans = resolver._resolve(repo_path, changed_files, "owner/repo")

            assert len(scopes) == 1
            assert scopes[0].package_manifest is not None
            assert scopes[0].package_manifest.dependencies_changed is True

    def test_scope_type_classification(self):
        """Test scope type classification from path indicators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "services" / "api").mkdir(parents=True)
            (repo_path / "services" / "api" / "go.mod").touch()
            (repo_path / "libs" / "utils").mkdir(parents=True)
            (repo_path / "libs" / "utils" / "go.mod").touch()
            (repo_path / "apps" / "web").mkdir(parents=True)
            (repo_path / "apps" / "web" / "package.json").touch()

            changed_files = [
                ChangedFile(filename="services/api/main.go", status=FileStatus.MODIFIED),
                ChangedFile(filename="libs/utils/helpers.go", status=FileStatus.MODIFIED),
                ChangedFile(filename="apps/web/index.ts", status=FileStatus.MODIFIED),
            ]

            resolver = ScopeResolver()
            scopes, orphans = resolver._resolve(repo_path, changed_files, "owner/repo")

            by_subroot = {s.subroot: s for s in scopes}
            assert by_subroot["services/api"].scope_type == ScopeType.SERVICE
            assert by_subroot["libs/utils"].scope_type == ScopeType.LIBRARY
            assert by_subroot["apps/web"].scope_type == ScopeType.APPLICATION


class TestManifestScoping:
    """Test scope detection based on manifests."""

    def test_manifest_creates_scope(self):
        """Test manifests create proper scopes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "go.mod").touch()

            changed_files = [
                ChangedFile(filename="main.go", status=FileStatus.MODIFIED),
            ]

            resolver = ScopeResolver()
            scopes, orphans = resolver._resolve(repo_path, changed_files, "owner/repo")

            # Candidate should exist with manifest
            assert scopes[0].package_manifest is not None

    def test_internal_dir_not_scope_indicator(self):
        """internal/ is not a scope indicator for sparse paths."""
        changed_files = ["backend/internal/cache/redis.go"]
        paths = derive_sparse_paths(changed_files)
        # Should use depth-2 fallback, not scope indicator
        assert "backend/internal/" in paths  # depth-2 prefix

    def test_indicator_suppressed_under_manifest(self):
        """Indicator scopes not created when parent manifest covers file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "services" / "api").mkdir(parents=True)
            (repo_path / "services" / "api" / "go.mod").touch()

            changed_files = [
                ChangedFile(filename="services/api/cmd/main.go", status=FileStatus.MODIFIED),
            ]

            resolver = ScopeResolver()
            scopes, orphans = resolver._resolve(repo_path, changed_files, "owner/repo")

            # Should be 1 scope (from manifest), not 2 (manifest + cmd/ indicator)
            assert len(scopes) == 1
            assert scopes[0].subroot == "services/api"
            assert len(orphans) == 0

    def test_root_manifest_suppresses_indicators_when_sole_manifest(self):
        """Root package.json as sole manifest suppresses indicator scopes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "package.json").touch()
            (repo_path / "scripts" / "deploy").mkdir(parents=True)

            changed_files = [
                ChangedFile(filename="scripts/deploy/prod.sh", status=FileStatus.MODIFIED),
            ]

            resolver = ScopeResolver()
            scopes, orphans = resolver._resolve(repo_path, changed_files, "owner/repo")

            # scripts/ indicator is suppressed when root is the only manifest (single-package repo)
            [s.subroot for s in scopes]
            assert len(scopes) == 1
            assert scopes[0].subroot == "."

    def test_root_manifest_suppresses_when_sole_manifest(self):
        """Single-package repo: root manifest suppresses all indicators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "pyproject.toml").touch()
            (repo_path / "src" / "pkg" / "tools" / "git").mkdir(parents=True)

            changed_files = [
                ChangedFile(filename="src/pkg/tools/git/client.py", status=FileStatus.MODIFIED),
            ]

            resolver = ScopeResolver()
            scopes, orphans = resolver._resolve(repo_path, changed_files, "owner/repo")

            assert len(scopes) == 1
            assert scopes[0].subroot == "."
            assert len(orphans) == 0

    def test_root_does_not_suppress_when_nested_manifests_exist(self):
        """Monorepo: root is container, indicators still fire for uncovered files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "package.json").touch()
            (repo_path / "packages" / "auth").mkdir(parents=True)
            (repo_path / "packages" / "auth" / "package.json").touch()
            (repo_path / "scripts" / "deploy").mkdir(parents=True)

            changed_files = [
                ChangedFile(filename="scripts/deploy/prod.sh", status=FileStatus.MODIFIED),
            ]

            resolver = ScopeResolver()
            scopes, orphans = resolver._resolve(repo_path, changed_files, "owner/repo")

            # scripts/deploy indicator should fire — root has nested manifests
            scope_subroots = [s.subroot for s in scopes]
            assert "scripts/deploy" in scope_subroots


class TestAssignFilesDeterminism:
    """Verify file assignment is deterministic for same-depth scopes."""

    def test_same_depth_scopes_deterministic(self):
        """Scopes at the same depth should assign files consistently."""
        from codespy.agents.reviewer.models import ScopeResult, ScopeType

        # Two scopes at depth 1 (one slash each)
        scope_a = ScopeResult(subroot="packages/alpha", scope_type=ScopeType.LIBRARY, reason="test")
        scope_b = ScopeResult(subroot="packages/beta", scope_type=ScopeType.LIBRARY, reason="test")
        file_alpha = ChangedFile(filename="packages/alpha/index.ts", status=FileStatus.MODIFIED)
        file_beta = ChangedFile(filename="packages/beta/index.ts", status=FileStatus.MODIFIED)

        resolver = ScopeResolver()
        # Pass scopes in both orders — assignment should be identical
        orphans_ab = resolver._assign_files([scope_a, scope_b], [file_alpha, file_beta])
        scope_a.changed_files.clear()
        scope_b.changed_files.clear()
        orphans_ba = resolver._assign_files([scope_b, scope_a], [file_alpha, file_beta])

        assert len(orphans_ab) == 0
        assert len(orphans_ba) == 0
        assert scope_a.changed_files == [file_alpha]
        assert scope_b.changed_files == [file_beta]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
