"""SEO pre-filter: detect web scopes and web-relevant changes."""

from __future__ import annotations

import json
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from codespy.tools.git.models import ChangedFile

if TYPE_CHECKING:
    from codespy.agents.review.scope.models import ScopeResult

# Dependencies that indicate a JS framework
JS_FRAMEWORK_DEPS: set[str] = {
    "next",
    "nuxt",
    "@sveltejs/kit",
    "astro",
    "gatsby",
    "@remix-run/core",
    "@remix-run/node",
    "@remix-run/react",
    "@remix-run/server-runtime",
    "@angular/ssr",
    "@nguniversal/express-engine",
    "@nguniversal/common",
    "react-dom",
    "vue",
    "svelte",
    "@builder.io/qwik",
    "solid-start",
}

# Static site generator markers
SSG_MARKERS: list[tuple[str, str | None]] = [
    ("hugo.toml", r"baseURL\s*="),
    ("hugo.yaml", r"baseURL\s*="),
    ("hugo.json", r'"baseURL"'),
    ("config.toml", r"baseURL\s*="),
    ("config.yaml", r"baseURL\s*="),
    ("_config.yml", None),  # Jekyll
    ("Gemfile", r"jekyll"),
    (".eleventy.js", None),
    ("eleventy.config.mjs", None),
    ("eleventy.config.js", None),
    ("eleventy.config.ts", None),
]

# Server-side template patterns
SERVER_TEMPLATE_PATTERNS: list[tuple[str, str]] = [
    ("*.heex", "phoenix"),
    ("*.cshtml", "aspnet"),
    ("*.razor", "aspnet"),
    ("*.tmpl", "go"),
    ("*.gohtml", "go"),
    ("*.blade.php", "laravel"),
    ("*.twig", "symfony"),
    ("*.jinja", "jinja"),
    ("*.j2", "jinja"),
    ("*.erb", "rails"),
    ("*.liquid", "liquid"),
    ("*.njk", "nunjucks"),
    ("*.hbs", "handlebars"),
]

# Site config patterns (crawl and hosting files)
SITE_CONFIG_PATTERNS: list[str] = [
    "robots.txt",
    "robots.*",
    "sitemap*",
    "sitemap.xml*",
    "llms.txt",
    "llms-full.txt",
    "vercel.json",
    "netlify.toml",
    "_redirects",
    "_headers",
    ".htaccess",
    "nginx.conf",
    "**nginx**/*.conf",
    "**apache**/*.conf",
    "firebase.json",
    "wrangler.toml",
    "middleware.*",
    "next.config.*",
    "nuxt.config.*",
    "astro.config.*",
    "svelte.config.*",
    "gatsby-config.*",
    "gatsby-config.js",
    "gatsby-config.ts",
    "gatsby-config.mjs",
    "hugo.toml",
    "hugo.yaml",
    "hugo.json",
    "config.toml",
    "config.yaml",
    "_config.yml",
    ".eleventy.js",
    "eleventy.config.*",
    "package.json",
]

# Web-relevant file patterns
WEB_EXTENSIONS: set[str] = {
    ".html", ".htm", ".jsx", ".tsx", ".vue", ".svelte", ".astro",
    ".mdx", ".css", ".scss", ".sass", ".less",
    ".erb", ".blade.php", ".twig", ".jinja", ".j2",
    ".liquid", ".njk", ".hbs", ".heex", ".cshtml", ".tmpl",
}

# Framework config patterns
FRAMEWORK_CONFIG_PATTERNS: list[str] = [
    "next.config.*",
    "nuxt.config.*",
    "astro.config.*",
    "svelte.config.*",
    "gatsby-config.*",
    "hugo.toml",
    "hugo.yaml",
    "hugo.json",
    "config.toml",
    "config.yaml",
    "_config.yml",
    ".eleventy.js",
    "eleventy.config.*",
]

# Content directories
CONTENT_DIR_PATTERNS: list[str] = [
    "content/**",
    "posts/**",
    "blog/**",
    "pages/**",
    "app/**",
    "routes/**",
    "public/**",
    "static/**",
    "assets/**",
]

# Excluded patterns (test, CI, docs)
EXCLUDED_PATTERNS: list[str] = [
    "*test*",
    "*spec*",
    "__tests__/**",
    "**/*.test.*",
    "**/*.spec.*",
    "**/__mocks__/**",
    "**/mocks/**",
    "**/.github/**",
    "**/.gitlab/**",
    "**/ci/**",
    "**/docs/**",
    "**/doc/**",
    "**/documentation/**",
    "**/stories/**",
    "**/storybook/**",
    "**/fixtures/**",
    "**/examples/**",
    "**/demo/**",
    "**/demos/**",
    "**/playground/**",
    "**/tools/**",
    "**/scripts/**",
    "**/benchmark/**",
    "**/benchmarks/**",
]


def _matches_pattern(path: str, pattern: str) -> bool:
    """Check if path matches a glob pattern."""
    if "**" in pattern:
        # Simple ** handling
        parts = pattern.split("**")
        if len(parts) == 2:
            prefix, suffix = parts
            return path.startswith(prefix) and path.endswith(suffix)
        return fnmatch(path, pattern)
    return fnmatch(path, pattern)


def _is_excluded(path: str) -> bool:
    """Check if path matches an excluded pattern."""
    for pattern in EXCLUDED_PATTERNS:
        if _matches_pattern(path, pattern):
            return True
    return False


def _has_js_framework_deps(scope_root: Path) -> bool:
    """Check if scope has JS framework dependencies in package.json."""
    package_json = scope_root / "package.json"
    if not package_json.exists():
        return False
    try:
        content = package_json.read_text()
        data = json.loads(content)
        deps: set[str] = set()
        deps.update(data.get("dependencies", {}).keys())
        deps.update(data.get("devDependencies", {}).keys())
        return bool(deps & JS_FRAMEWORK_DEPS)
    except (json.JSONDecodeError, OSError):
        return False


def _has_ssg_marker(scope_root: Path) -> bool:
    """Check if scope has static site generator markers."""
    for filename, pattern in SSG_MARKERS:
        filepath = scope_root / filename
        if filepath.exists():
            if pattern is None:
                return True
            try:
                content = filepath.read_text()
                if re.search(pattern, content, re.IGNORECASE):
                    return True
            except OSError:
                pass
    return False


def _has_server_templates(scope_root: Path) -> bool:
    """Check if scope has server-side templates."""
    for pattern, _ in SERVER_TEMPLATE_PATTERNS:
        # Check in templates/views directories
        for templates_dir in ["templates", "views", "app/views", "resources/views"]:
            templates_path = scope_root / templates_dir
            if templates_path.exists():
                try:
                    for f in templates_path.rglob("*"):
                        if fnmatch(f.name, pattern):
                            return True
                except OSError:
                    pass
    return False


def _has_plain_html(scope_root: Path) -> bool:
    """Check if scope has plain HTML files."""
    for html_file in ["index.html", "index.htm"]:
        if (scope_root / html_file).exists():
            return True
    # Check for any .html files
    try:
        for f in scope_root.rglob("*.html"):
            if f.is_file() and not _is_excluded(str(f.relative_to(scope_root))):
                return True
    except OSError:
        pass
    return False


def is_web_scope(scope_root: Path, scope: "ScopeResult") -> tuple[bool, str]:
    """Determine if a scope looks like a web site/app.

    Returns (is_web, reason).
    """
    # Check changed files first - if crawl/hosting files changed, it's web
    for cf in scope.changed_files:
        if is_site_config_file(cf.filename):
            return True, f"site config file: {cf.filename}"

    # Check for JS frameworks
    if _has_js_framework_deps(scope_root):
        return True, "JS framework detected in package.json"

    # Check for SSG markers
    if _has_ssg_marker(scope_root):
        return True, "static site generator marker"

    # Check for server templates
    if _has_server_templates(scope_root):
        return True, "server-side templates detected"

    # Check for plain HTML
    if _has_plain_html(scope_root):
        return True, "plain HTML files detected"

    return False, "no web markers found"


def is_site_config_file(filename: str) -> bool:
    """Check if a file is a site config file (crawl or hosting config)."""
    for pattern in SITE_CONFIG_PATTERNS:
        if fnmatch(filename, pattern):
            return True
    return False


def has_web_relevant_changes(changed_files: list[ChangedFile]) -> tuple[bool, str]:
    """Check if changed files contain web-relevant content.

    Returns (has_relevant, reason).
    """
    for cf in changed_files:
        filename = cf.filename
        ext = Path(filename).suffix.lower()

        # Skip excluded patterns
        if _is_excluded(filename):
            continue

        # Check for site config files
        if is_site_config_file(filename):
            return True, f"site config: {filename}"

        # Check for framework config files
        for pattern in FRAMEWORK_CONFIG_PATTERNS:
            if fnmatch(filename, pattern):
                return True, f"framework config: {filename}"

        # Check for markup/components
        if ext in WEB_EXTENSIONS:
            return True, f"web component: {filename}"

        # Check for content directories
        for pattern in CONTENT_DIR_PATTERNS:
            if _matches_pattern(filename, pattern):
                return True, f"content file: {filename}"

        # Check for markdown in content dirs
        if ext in {".md", ".mdx"}:
            # Only count if under content-like dirs
            for dir_pattern in ["content/**", "posts/**", "blog/**", "pages/**"]:
                if _matches_pattern(filename, dir_pattern):
                    return True, f"content markdown: {filename}"

    return False, "no web-relevant changes"
