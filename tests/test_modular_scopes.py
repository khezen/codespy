"""Modular repo scoping (Swift packages, Gradle modules, web), concurrency cap, audit soft-fail."""

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codespy.agents.review.helpers import gather_limited
from codespy.agents.review.scope import ScopeResolver
from codespy.agents.review.scope.agent import ScopeBoundary
from codespy.agents.review.scope.manifest_parser import extract_package_name
from codespy.agents.review.scope.models import ScopeResult, ScopeType
from codespy.tools.git.models import ChangedFile, FileStatus


def _touch(root: Path, rel: str, content: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _changed(*paths: str) -> list[ChangedFile]:
    return [ChangedFile(filename=p, status=FileStatus.MODIFIED) for p in paths]


def _by_subroot(scopes: list[ScopeResult]) -> dict[str, list[str]]:
    return {s.subroot: sorted(f.filename for f in s.changed_files) for s in scopes}


@pytest.fixture
def ios_repo(tmp_path: Path) -> Path:
    """example-ios layout: root Gemfile + Xcode project + sibling Swift packages."""
    _touch(tmp_path, "Gemfile")
    _touch(tmp_path, "Example.xcodeproj/project.pbxproj")
    _touch(tmp_path, "Example.xcodeproj/project.xcworkspace/contents.xcworkspacedata")
    _touch(tmp_path, "Example.xcworkspace/contents.xcworkspacedata")
    for pkg in ("FeatureA", "FeatureB"):
        _touch(tmp_path, f"{pkg}/Package.swift", f'let package = Package(\n    name: "{pkg}",\n)')
    return tmp_path


class TestIOSLayout:
    def test_first_level_packages_get_own_scopes(self, ios_repo: Path) -> None:
        files = _changed(
            "FeatureA/Sources/x.swift",
            "FeatureB/Tests/y.swift",
            "Example/App.swift",
            "Example.xcodeproj/project.pbxproj",
        )
        scopes, orphans = ScopeResolver()._resolve(ios_repo, files, "o/r")

        assert not orphans
        assert _by_subroot(scopes) == {
            "FeatureA": ["FeatureA/Sources/x.swift"],
            "FeatureB": ["FeatureB/Tests/y.swift"],
            ".": ["Example.xcodeproj/project.pbxproj", "Example/App.swift"],
        }

    def test_root_manifest_is_xcode_not_gemfile(self, ios_repo: Path) -> None:
        scopes, _ = ScopeResolver()._resolve(ios_repo, _changed("Example/App.swift"), "o/r")
        root = {s.subroot: s for s in scopes}["."]
        assert root.package_manifest is not None
        assert root.package_manifest.package_manager == "xcode"
        assert root.package_manifest.manifest_path == "Example.xcodeproj/project.pbxproj"
        assert root.package_manifest.package_name == "Example"
        assert root.scope_type == ScopeType.APPLICATION

    def test_pbxproj_edit_is_not_a_dependency_change(self, ios_repo: Path) -> None:
        scopes, _ = ScopeResolver()._resolve(
            ios_repo, _changed("Example.xcodeproj/project.pbxproj"), "o/r"
        )
        assert scopes[0].package_manifest.dependencies_changed is False

    def test_swiftpm_lock_change_is_a_dependency_change(self, ios_repo: Path) -> None:
        lock = "Example.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved"
        _touch(ios_repo, lock, "{}")
        scopes, _ = ScopeResolver()._resolve(ios_repo, _changed(lock), "o/r")
        manifest = scopes[0].package_manifest
        assert manifest.lock_file_path == lock
        assert manifest.dependencies_changed is True

    def test_no_scope_inside_xcode_bundle(self, ios_repo: Path) -> None:
        _touch(ios_repo, "Example.xcodeproj/project.xcworkspace/Package.swift")
        files = _changed("Example.xcodeproj/project.xcworkspace/contents.xcworkspacedata")
        scopes, _ = ScopeResolver()._resolve(ios_repo, files, "o/r")
        assert [s.subroot for s in scopes] == ["."]


class TestAndroidLayout:
    def test_depth_one_and_depth_three_modules(self, tmp_path: Path) -> None:
        _touch(tmp_path, "build.gradle.kts")
        _touch(tmp_path, "settings.gradle.kts", 'rootProject.name = "Example"')
        _touch(
            tmp_path, "androidApp/build.gradle.kts", 'android {\n    namespace = "com.example.app"\n}'
        )
        _touch(tmp_path, "modules/feature/fea_x/build.gradle.kts", "plugins {}")
        files = _changed(
            "androidApp/src/main/kotlin/Main.kt",
            "modules/feature/fea_x/src/main/kotlin/X.kt",
            "gradle/libs.versions.toml",
        )
        scopes, orphans = ScopeResolver()._resolve(tmp_path, files, "o/r")

        assert not orphans
        by = {s.subroot: s for s in scopes}
        assert set(by) == {"androidApp", "modules/feature/fea_x", "."}
        assert by["androidApp"].package_manifest.package_name == "com.example.app"
        assert by["modules/feature/fea_x"].package_manifest.package_name is None
        assert by["."].package_manifest.package_name == "Example"


class TestManifestPriority:
    def test_independent_of_listdir_order(self, tmp_path: Path) -> None:
        _touch(tmp_path, "Gemfile")
        _touch(tmp_path, "X.xcodeproj/project.pbxproj")
        real_listdir = os.listdir
        resolver = ScopeResolver()
        results = set()
        for reverse in (False, True):
            def fake_listdir(path, _reverse=reverse):
                return sorted(real_listdir(path), reverse=_reverse)

            with patch("codespy.agents.review.scope.agent.os.listdir", side_effect=fake_listdir):
                manifests = resolver._discover_manifests(tmp_path, _changed("a.swift"), [])
            results.add(manifests[Path(".")])
        assert results == {("xcode", "X.xcodeproj")}

    def test_build_gradle_beats_settings_gradle(self, tmp_path: Path) -> None:
        _touch(tmp_path, "settings.gradle.kts")
        _touch(tmp_path, "build.gradle.kts")
        manifests = ScopeResolver()._discover_manifests(tmp_path, _changed("a.kt"), [])
        assert manifests[Path(".")] == ("gradle", "build.gradle.kts")

    def test_nx_project_json_requires_root_nx_json(self, tmp_path: Path) -> None:
        _touch(tmp_path, "apps/web/project.json", '{"name": "web"}')
        files = _changed("apps/web/src/main.ts")
        resolver = ScopeResolver()
        assert Path("apps/web") not in resolver._discover_manifests(tmp_path, files, [])

        _touch(tmp_path, "nx.json", "{}")
        manifests = resolver._discover_manifests(tmp_path, files, [])
        assert manifests[Path("apps/web")] == ("nx", "project.json")
        assert Path(".") not in manifests  # nx.json is a marker, not a manifest


class TestApplyBoundaries:
    def _det(self, tmp_path: Path, files: list[ChangedFile]) -> list[ScopeResult]:
        scopes, _ = ScopeResolver()._resolve(tmp_path, files, "o/r")
        return scopes

    def test_llm_boundary_kept_when_root_is_container(self, ios_repo: Path) -> None:
        files = _changed("FeatureA/Sources/x.swift", "Example/App.swift")
        det = self._det(ios_repo, files)
        boundaries = [
            ScopeBoundary(subroot="Example", scope_type=ScopeType.APPLICATION, reason="app target"),
            ScopeBoundary(subroot="FeatureA/Sources", scope_type=ScopeType.LIBRARY, reason="x"),
        ]
        final = ScopeResolver()._apply_boundaries(boundaries, files, det, "o/r")
        assert _by_subroot(final) == {
            "FeatureA": ["FeatureA/Sources/x.swift"],  # inside a non-root manifest: dropped
            "Example": ["Example/App.swift"],
        }

    def test_llm_boundary_dropped_when_root_is_sole_manifest(self, tmp_path: Path) -> None:
        _touch(tmp_path, "pyproject.toml")
        files = _changed("src/pkg/a.py")
        det = self._det(tmp_path, files)
        boundaries = [ScopeBoundary(subroot="src/pkg", scope_type=ScopeType.LIBRARY, reason="x")]
        final = ScopeResolver()._apply_boundaries(boundaries, files, det, "o/r")
        assert _by_subroot(final) == {".": ["src/pkg/a.py"]}


class TestManifestParser:
    def test_gradle_groovy_namespace(self, tmp_path: Path) -> None:
        _touch(tmp_path, "m/build.gradle", "android {\n    namespace 'com.example.m'\n}")
        assert extract_package_name("m/build.gradle", tmp_path) == "com.example.m"

    def test_deno_jsonc_comments_and_trailing_comma(self, tmp_path: Path) -> None:
        _touch(
            tmp_path,
            "deno.jsonc",
            '{\n  // comment with "quotes"\n  "name": "@scope/pkg", /* block */\n  "url": "https://x",\n}',
        )
        assert extract_package_name("deno.jsonc", tmp_path) == "@scope/pkg"

    def test_tuist_project_name(self, tmp_path: Path) -> None:
        _touch(tmp_path, "Project.swift", 'let project = Project(\n    name: "App",\n)')
        assert extract_package_name("Project.swift", tmp_path) == "App"

    def test_podspec_stem(self, tmp_path: Path) -> None:
        _touch(tmp_path, "Lib.podspec")
        assert extract_package_name("Lib.podspec", tmp_path) == "Lib"


class TestGatherLimited:
    def test_caps_in_flight_and_keeps_order(self) -> None:
        in_flight = 0
        peak = 0

        async def work(i: int) -> int:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            if i == 3:
                raise ValueError("boom")
            return i

        results = asyncio.run(gather_limited([work(i) for i in range(5)], 2))
        assert peak == 2
        assert results[:3] == [0, 1, 2] and results[4] == 4
        assert isinstance(results[3], ValueError)

    def test_invalid_limit_is_unbounded(self) -> None:
        async def work(i: int) -> int:
            return i

        assert asyncio.run(gather_limited([work(i) for i in range(3)], MagicMock())) == [0, 1, 2]


class TestRefinementFailureKeepsModules:
    def test_refusal_keeps_deterministic_scopes(self, ios_repo: Path) -> None:
        from codespy.agents.review.models import PRContext, ReviewContext

        files = _changed("FeatureA/Sources/x.swift", "FeatureB/Tests/y.swift", "Example/App.swift")
        pr = MagicMock(changed_files=files, repo_slug="o/r", repo_full_name="o/r")
        review_context = ReviewContext(
            pr_context=PRContext(
                repo_slug="github.com/o/r", pr_number=1, pr_title="t",
                pr_url="https://github.com/o/r/pull/1", summary="s",
            ),
            prefrontal_memory="",
        )
        review_context.metadata = MagicMock(pr=pr, repo_path=ios_repo, is_local=True, run_id="r")
        resolver = ScopeResolver()
        resolver._settings = MagicMock()
        resolver._settings.review.excluded_directories = []
        resolver._settings.is_signature_enabled.return_value = True
        resolver._settings.get_scope_skip_refinement.return_value = True

        async def noop(*a, **k):
            return None

        async def refuse(*a, **k):
            raise RuntimeError("model refused (content_filter)")

        with (
            patch.object(resolver, "_ensure_repo", side_effect=noop),
            patch.object(resolver, "_refine_scopes", side_effect=refuse),
        ):
            scopes = asyncio.run(resolver.aforward(review_context))

        assert _by_subroot(scopes) == {
            "FeatureA": ["FeatureA/Sources/x.swift"],
            "FeatureB": ["FeatureB/Tests/y.swift"],
            ".": ["Example/App.swift"],
        }


class TestAuditSoftFail:
    def test_refusal_does_not_abort_review(self) -> None:
        from codespy.agents.review.models import PRContext, ReviewContext

        review_context = ReviewContext(
            pr_context=PRContext(
                repo_slug="github.com/o/r",
                pr_number=1,
                pr_title="t",
                pr_url="https://github.com/o/r/pull/1",
                summary="s",
            ),
            prefrontal_memory="",
        )
        settings = MagicMock()
        settings.is_signature_enabled.return_value = True
        settings.memory.enabled = False

        async def refuse(agent, **kwargs):
            raise RuntimeError("model refused (content_filter)\n\nAdapter failed")

        mod = "codespy.agents.review.audit.agent"
        with (
            patch(f"{mod}.get_settings", return_value=settings),
            patch(f"{mod}.get_cost_tracker", return_value=MagicMock()),
            patch(f"{mod}.build_rlm_agent", return_value=MagicMock()),
            patch(f"{mod}.run_rlm", side_effect=refuse),
            patch(f"{mod}.SignatureContext"),
            patch(f"{mod}.get_prefrontal", return_value=None),
        ):
            from codespy.agents.review.audit.agent import Auditor

            assessment, recommendation = Auditor().forward(
                review_context=review_context, all_issues=[MagicMock()]
            )

        assert assessment == "Audit unavailable: model refused (content_filter)"
        assert recommendation == "NEEDS_DISCUSSION"
