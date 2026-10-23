"""Module markers (single-manifest modular monoliths) and workspace dependency detection."""

from pathlib import Path

import pytest

from codespy.agents.review.scope import ScopeResolver
from codespy.agents.review.scope.agent import ScopeBoundary, build_sparse_patterns
from codespy.agents.review.scope.manifest_parser import extract_package_name
from codespy.agents.review.scope.models import ScopeResult, ScopeType
from codespy.agents.review.scope.module_markers import (
    discover_module_markers,
    normalize_subroot,
)
from codespy.tools.git.models import ChangedFile, FileStatus


def _touch(root: Path, rel: str, content: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _changed(*paths: str) -> list[ChangedFile]:
    return [ChangedFile(filename=p, status=FileStatus.MODIFIED) for p in paths]


def _resolve(repo: Path, *paths: str) -> tuple[dict[str, ScopeResult], list[ChangedFile]]:
    scopes, orphans = ScopeResolver()._resolve(repo, _changed(*paths), "o/r")
    return {s.subroot: s for s in scopes}, orphans


def _files(scope: ScopeResult) -> list[str]:
    return sorted(f.filename for f in scope.changed_files)


# --- marker kinds -----------------------------------------------------------------------


class TestMarkerKinds:
    def test_packwerk(self, tmp_path: Path) -> None:
        _touch(tmp_path, "Gemfile")
        _touch(tmp_path, "packwerk.yml")
        _touch(tmp_path, "package.yml")
        _touch(tmp_path, "packs/billing/package.yml")
        scopes, _ = _resolve(tmp_path, "packs/billing/app/models/invoice.rb", "app/x.rb")
        assert _files(scopes["packs/billing"]) == ["packs/billing/app/models/invoice.rb"]
        assert scopes["packs/billing"].module_marker == "packwerk:packs/billing/package.yml"
        assert _files(scopes["."]) == ["app/x.rb"]

    def test_packwerk_requires_root_config(self, tmp_path: Path) -> None:
        _touch(tmp_path, "Gemfile")
        _touch(tmp_path, "packs/billing/package.yml")
        scopes, _ = _resolve(tmp_path, "packs/billing/app/x.rb")
        assert set(scopes) == {"."}

    def test_tach(self, tmp_path: Path) -> None:
        _touch(tmp_path, "pyproject.toml", '[project]\nname = "shop"\n')
        _touch(
            tmp_path,
            "tach.toml",
            'source_roots = ["src"]\n[[modules]]\npath = "shop.billing"\n'
            '[[modules]]\npath = "missing"\n',
        )
        _touch(tmp_path, "src/shop/billing/__init__.py")
        scopes, _ = _resolve(tmp_path, "src/shop/billing/api.py")
        assert "src/shop/billing" in scopes
        assert scopes["src/shop/billing"].module_marker == "tach:tach.toml"

    @pytest.mark.parametrize("source", ["importlinter", "pyproject"])
    def test_import_linter(self, tmp_path: Path, source: str) -> None:
        if source == "importlinter":
            _touch(tmp_path, "pyproject.toml", '[project]\nname = "shop"\n')
            _touch(
                tmp_path,
                ".importlinter",
                "[importlinter]\nroot_package = shop\n\n"
                "[importlinter:contract:layers]\nname = L\ntype = layers\n"
                "containers =\n    shop\nlayers =\n    api\n    (billing | auth)\n",
            )
        else:
            _touch(
                tmp_path,
                "pyproject.toml",
                '[project]\nname = "shop"\n[tool.importlinter]\nroot_package = "shop"\n'
                '[[tool.importlinter.contracts]]\nname = "i"\ntype = "independence"\n'
                'modules = ["shop.billing", "shop.auth"]\n',
            )
        _touch(tmp_path, "src/shop/billing/__init__.py")
        _touch(tmp_path, "src/shop/auth/__init__.py")
        markers = discover_module_markers(
            tmp_path, ["src/shop/billing/a.py"], {Path("."): ("pip", "pyproject.toml")}
        )
        assert "src/shop/billing" in markers
        assert "src/shop/auth" in markers

    def test_go_arch_lint(self, tmp_path: Path) -> None:
        _touch(tmp_path, "go.mod", "module x\n")
        _touch(
            tmp_path,
            ".go-arch-lint.yml",
            "version: 3\nworkdir: internal\ncomponents:\n"
            "  billing: { in: billing/** }\n  auth: { in: [auth/*.go] }\n",
        )
        _touch(tmp_path, "internal/billing/svc/a.go")
        _touch(tmp_path, "internal/auth/a.go")
        scopes, _ = _resolve(tmp_path, "internal/billing/svc/a.go", "internal/auth/a.go")
        assert {"internal/billing", "internal/auth"} <= set(scopes)

    def test_spring_modulith(self, tmp_path: Path) -> None:
        base = "src/main/java/com/shop"
        _touch(tmp_path, "pom.xml", "<project><artifactId>shop</artifactId></project>")
        _touch(tmp_path, f"{base}/ShopApplication.java", "@SpringBootApplication\nclass S {}")
        _touch(tmp_path, f"{base}/billing/Invoice.java")
        scopes, _ = _resolve(
            tmp_path, f"{base}/billing/Invoice.java", f"{base}/ShopApplication.java"
        )
        assert _files(scopes[f"{base}/billing"]) == [f"{base}/billing/Invoice.java"]
        assert _files(scopes["."]) == [f"{base}/ShopApplication.java"]

    def test_jpms(self, tmp_path: Path) -> None:
        _touch(tmp_path, "pom.xml", "<project><artifactId>p</artifactId></project>")
        _touch(tmp_path, "billing/src/main/java/module-info.java")
        scopes, _ = _resolve(tmp_path, "billing/src/main/java/b/A.java")
        assert scopes["billing"].module_marker == "jpms:billing/src/main/java/module-info.java"

    def test_go_cmd_single_module(self, tmp_path: Path) -> None:
        _touch(tmp_path, "go.mod", "module x\n")
        _touch(tmp_path, "cmd/api/main.go")
        _touch(tmp_path, "cmd/worker/main.go")
        scopes, _ = _resolve(
            tmp_path, "cmd/api/main.go", "cmd/worker/handler.go", "internal/db/db.go"
        )
        assert set(scopes) == {"cmd/api", "cmd/worker", "."}
        assert _files(scopes["."]) == ["internal/db/db.go"]

    def test_go_cmd_never_splits_nested_go_mod(self, tmp_path: Path) -> None:
        """example/mono layout: service with its own go.mod stays one scope."""
        _touch(tmp_path, "go.mod", "module mono\n")
        _touch(tmp_path, "svc/x/go.mod", "module mono/svc/x\n")
        _touch(tmp_path, "svc/x/cmd/srv/main.go")
        scopes, _ = _resolve(tmp_path, "svc/x/cmd/srv/main.go")
        assert set(scopes) == {"svc/x"}

    def test_ts_references(self, tmp_path: Path) -> None:
        _touch(tmp_path, "package.json", '{"name": "web"}')
        _touch(
            tmp_path,
            "tsconfig.json",
            '{\n  // project refs\n  "references": [\n    {"path": "./packages/auth"},\n'
            '    {"path": "packages/billing/tsconfig.build.json"},\n  ],\n}',
        )
        _touch(tmp_path, "packages/auth/src/a.ts")
        _touch(tmp_path, "packages/billing/tsconfig.build.json", "{}")
        scopes, _ = _resolve(tmp_path, "packages/auth/src/a.ts", "packages/billing/src/b.ts")
        assert {"packages/auth", "packages/billing"} <= set(scopes)

    def test_laravel(self, tmp_path: Path) -> None:
        _touch(tmp_path, "composer.json", '{"name": "acme/app"}')
        _touch(tmp_path, "Modules/Billing/module.json", "{}")
        _touch(tmp_path, "lib/Thing/module.json", "{}")
        scopes, _ = _resolve(tmp_path, "Modules/Billing/Http/C.php", "lib/Thing/T.php")
        assert scopes["Modules/Billing"].module_marker == "laravel:Modules/Billing/module.json"
        assert "lib/Thing" not in scopes  # module.json outside Modules/

    def test_deptrac(self, tmp_path: Path) -> None:
        _touch(tmp_path, "composer.json", '{"name": "acme/app"}')
        _touch(
            tmp_path,
            "deptrac.yaml",
            "deptrac:\n  layers:\n    - name: Billing\n      collectors:\n"
            "        - type: directory\n          value: src/Billing/.*\n",
        )
        _touch(tmp_path, "src/Billing/Invoice.php")
        scopes, _ = _resolve(tmp_path, "src/Billing/Invoice.php")
        assert "src/Billing" in scopes

    def test_sbt(self, tmp_path: Path) -> None:
        _touch(
            tmp_path,
            "build.sbt",
            'name := "shop"\nlazy val billing = project.in(file("billing"))\n'
            'lazy val core = project in file("core")\nlazy val root = project.in(file("."))\n',
        )
        _touch(tmp_path, "billing/src/main/scala/B.scala")
        scopes, _ = _resolve(tmp_path, "billing/src/main/scala/B.scala", "README.md")
        assert scopes["."].package_manifest.package_name == "shop"
        assert scopes["billing"].module_marker == "sbt:build.sbt"

    def test_mill(self, tmp_path: Path) -> None:
        _touch(tmp_path, "build.mill", "package build\nobject api extends ScalaModule {\n}\n")
        _touch(tmp_path, "api/src/A.scala")
        _touch(tmp_path, "core/package.mill", "package build.core\n")
        scopes, _ = _resolve(tmp_path, "api/src/A.scala", "core/src/C.scala")
        assert scopes["api"].module_marker == "mill:build.mill"
        assert scopes["core"].module_marker == "mill:core/package.mill"


# --- robustness -------------------------------------------------------------------------


class TestRobustness:
    @pytest.mark.parametrize(
        "name,content",
        [
            ("tach.toml", "[[modules]\npath = "),
            ("deptrac.yaml", "deptrac: [unclosed"),
            ("tsconfig.json", '{"references": [ {"path": '),
        ],
    )
    def test_malformed_config_ignored(self, tmp_path: Path, name: str, content: str) -> None:
        _touch(tmp_path, "package.json", '{"name": "x"}')
        _touch(tmp_path, name, content)
        scopes, _ = _resolve(tmp_path, "src/a.ts")
        assert set(scopes) == {"."}

    def test_path_safety(self, tmp_path: Path) -> None:
        (tmp_path / "ok").mkdir()
        assert normalize_subroot("../x", tmp_path) is None
        assert normalize_subroot("/etc", tmp_path) is None
        assert normalize_subroot("missing", tmp_path) is None
        assert normalize_subroot("./ok/", tmp_path) == "ok"

    def test_symlink_marker_outside_repo_rejected(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        outside = tmp_path / "outside"
        (outside / "pkg").mkdir(parents=True)
        _touch(outside, "package.yml")
        _touch(repo, "pyproject.toml", '[project]\nname = "shop"\n')
        _touch(repo, "packwerk.yml")
        _touch(repo, "tach.toml", '[[modules]]\npath = "evil"\n')
        (repo / "evil").symlink_to(outside, target_is_directory=True)
        (repo / "loop").symlink_to(repo, target_is_directory=True)
        assert normalize_subroot("evil", repo) is None
        assert normalize_subroot("loop", repo) is None
        markers = discover_module_markers(
            repo, ["evil/x.py", "evil/pkg/y.py"], {Path("."): ("pip", "pyproject.toml")}
        )
        assert markers == {}

    def test_marker_topic_is_path_based(self, tmp_path: Path) -> None:
        _touch(tmp_path, "Gemfile")
        _touch(tmp_path, "packwerk.yml")
        _touch(tmp_path, "packs/billing/package.yml")
        scopes, _ = _resolve(tmp_path, "packs/billing/x.rb")
        topic = scopes["packs/billing"].topic("o/r")
        assert topic.id.endswith("packs/billing")

    def test_manifest_wins_over_marker(self, tmp_path: Path) -> None:
        _touch(tmp_path, "Gemfile")
        _touch(tmp_path, "packwerk.yml")
        _touch(tmp_path, "packs/billing/package.yml")
        _touch(tmp_path, "packs/billing/billing.gemspec")
        scopes, _ = _resolve(tmp_path, "packs/billing/x.rb")
        assert scopes["packs/billing"].module_marker is None
        assert scopes["packs/billing"].package_manifest.package_name == "billing"

    def test_sparse_patterns_include_markers(self) -> None:
        patterns = build_sparse_patterns(["packs/billing/app/x.rb"])
        assert "/packwerk.yml" in patterns
        assert "/packs/billing/package.yml" in patterns
        assert "/packs/billing/*Application.java" in patterns
        assert "/packs/billing/main.go" in patterns


# --- _apply_boundaries ------------------------------------------------------------------


class TestApplyBoundaries:
    def test_marker_kept_and_inner_boundary_dropped(self, tmp_path: Path) -> None:
        _touch(tmp_path, "Gemfile")
        _touch(tmp_path, "packwerk.yml")
        _touch(tmp_path, "packs/billing/package.yml")
        files = _changed("packs/billing/app/a.rb", "lib/x.rb")
        det, orphans = ScopeResolver()._resolve(tmp_path, files, "o/r")
        boundaries = [
            ScopeBoundary(subroot=".", scope_type=ScopeType.APPLICATION, reason="r"),
            ScopeBoundary(subroot="packs/billing/app", scope_type=ScopeType.LIBRARY, reason="r"),
            ScopeBoundary(subroot="lib", scope_type=ScopeType.LIBRARY, reason="r"),
        ]
        final = {
            s.subroot: s for s in ScopeResolver()._apply_boundaries(boundaries, files, det, "o/r")
        }
        assert set(final) == {
            ".",
            "packs/billing",
        }  # inner dropped; lib dropped (sole root manifest)
        assert final["packs/billing"].module_marker == "packwerk:packs/billing/package.yml"
        assert _files(final["."]) == ["lib/x.rb"]


# --- new manifests ----------------------------------------------------------------------


class TestNewManifests:
    def test_cmake_gate(self, tmp_path: Path) -> None:
        _touch(tmp_path, "CMakeLists.txt", "project(engine)\nadd_subdirectory(lib)\n")
        _touch(tmp_path, "lib/CMakeLists.txt", "add_library(core a.c)\n")
        _touch(tmp_path, "tests/CMakeLists.txt", "add_subdirectory(unit)\n")
        scopes, _ = _resolve(tmp_path, "lib/a.c", "lib/CMakeLists.txt", "tests/t.c")
        assert scopes["."].package_manifest.package_name == "engine"
        assert scopes["lib"].package_manifest.package_manager == "cmake"
        assert scopes["lib"].package_manifest.dependencies_changed is False
        assert "tests" not in scopes

    def test_meson(self, tmp_path: Path) -> None:
        _touch(tmp_path, "meson.build", "project('engine', 'c')\n")
        scopes, _ = _resolve(tmp_path, "src/a.c")
        assert scopes["."].package_manifest.package_name == "engine"

    def test_scala_cli_gate(self, tmp_path: Path) -> None:
        _touch(tmp_path, "tool/project.scala", "//> using scala 3\n")
        _touch(tmp_path, "other/project.scala", "object X\n")
        scopes, _ = _resolve(tmp_path, "tool/project.scala", "other/Main.scala")
        assert scopes["tool"].package_manifest.package_manager == "scala-cli"
        assert scopes["tool"].package_manifest.dependencies_changed is True
        assert "other" not in scopes

    def test_gemspec_name(self, tmp_path: Path) -> None:
        _touch(tmp_path, "mygem.gemspec")
        assert extract_package_name("mygem.gemspec", tmp_path) == "mygem"


# --- dependency detection ---------------------------------------------------------------


class TestDependencyDetection:
    @pytest.fixture
    def cargo_ws(self, tmp_path: Path) -> Path:
        _touch(tmp_path, "Cargo.toml", '[workspace]\nmembers = ["crates/a"]\n')
        _touch(tmp_path, "Cargo.lock")
        _touch(tmp_path, "crates/a/Cargo.toml", '[package]\nname = "a"\n')
        return tmp_path

    def test_cargo_workspace_lock_flags_member(self, cargo_ws: Path) -> None:
        scopes, _ = _resolve(cargo_ws, "Cargo.lock", "crates/a/src/lib.rs")
        manifest = scopes["crates/a"].package_manifest
        assert manifest.lock_file_path == "Cargo.lock"
        assert manifest.dependencies_changed is True

    def test_cargo_member_src_edit_is_not_dep_change(self, cargo_ws: Path) -> None:
        scopes, _ = _resolve(cargo_ws, "crates/a/src/lib.rs")
        assert scopes["crates/a"].package_manifest.dependencies_changed is False

    def test_dart_workspace_lock(self, tmp_path: Path) -> None:
        _touch(tmp_path, "pubspec.yaml", "name: ws\nworkspace:\n  - packages/a\n")
        _touch(tmp_path, "pubspec.lock")
        _touch(tmp_path, "packages/a/pubspec.yaml", "name: a\nresolution: workspace\n")
        scopes, _ = _resolve(tmp_path, "pubspec.lock", "packages/a/lib/a.dart")
        assert scopes["packages/a"].package_manifest.dependencies_changed is True

    def test_npm_lock_variants_still_found(self, tmp_path: Path) -> None:
        _touch(tmp_path, "package.json", '{"name": "x"}')
        _touch(tmp_path, "yarn.lock")
        scopes, _ = _resolve(tmp_path, "yarn.lock")
        assert scopes["."].package_manifest.lock_file_path == "yarn.lock"

    def test_go_sum_not_inherited_from_ancestor(self, tmp_path: Path) -> None:
        _touch(tmp_path, "go.mod", "module m\n")
        _touch(tmp_path, "go.sum")
        _touch(tmp_path, "svc/x/go.mod", "module m/x\n")
        scopes, _ = _resolve(tmp_path, "go.sum", "svc/x/a.go")
        assert scopes["svc/x"].package_manifest.dependencies_changed is False

    def test_dotnet_central_packages(self, tmp_path: Path) -> None:
        _touch(tmp_path, "Directory.Packages.props", "<Project/>")
        _touch(tmp_path, "src/A/A.csproj", "<Project/>")
        _touch(tmp_path, "src/B/B.csproj", "<Project/>")
        scopes, _ = _resolve(tmp_path, "Directory.Packages.props", "src/A/a.cs", "src/B/b.cs")
        for sub in ("src/A", "src/B"):
            manifest = scopes[sub].package_manifest
            assert manifest.dependencies_changed is True
            assert manifest.lock_file_path == "Directory.Packages.props"

    def test_dotnet_root_prefers_sln(self, tmp_path: Path) -> None:
        _touch(tmp_path, "X.sln")
        _touch(tmp_path, "Directory.Build.props", "<Project/>")
        scopes, _ = _resolve(tmp_path, "README.md")
        assert scopes["."].package_manifest.manifest_path == "X.sln"

    def test_slnx_and_packages_config(self, tmp_path: Path) -> None:
        _touch(tmp_path, "X.slnx")
        _touch(tmp_path, "legacy/L.csproj", "<Project/>")
        _touch(tmp_path, "legacy/packages.config", "<packages/>")
        scopes, _ = _resolve(tmp_path, "legacy/packages.config", "README.md")
        assert scopes["."].package_manifest.manifest_path == "X.slnx"
        assert scopes["legacy"].package_manifest.dependencies_changed is True


@pytest.mark.parametrize(
    "subroot,lock,expected",
    [
        ("crates/a", "Cargo.lock", "."),
        ("crates/a", "crates/a/Cargo.lock", "crates/a"),
        ("ws/crates/a", "ws/Cargo.lock", "ws"),
        ("a", "ab/Cargo.lock", "."),
        (".", "Cargo.lock", "."),
    ],
)
def test_supply_chain_tools_root(subroot: str, lock: str, expected: str) -> None:
    from codespy.agents.review.supply_chain.agent import _common_dir

    assert _common_dir(subroot, lock) == expected
