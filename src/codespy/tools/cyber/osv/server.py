"""MCP server for OSV (Open Source Vulnerabilities) security scanning operations."""

import logging
import os
from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP

from codespy.tools.cyber.osv.client import OSVClient

logger = logging.getLogger(__name__)
_caller_module = os.environ.get("MCP_CALLER_MODULE", "unknown")

mcp = FastMCP("osv")
_client: OSVClient | None = None


def _get_client() -> OSVClient:
    """Get the OSVClient instance, raising if not initialized."""
    if _client is None:
        raise RuntimeError("OSVClient not initialized")
    return _client


# @mcp.tool()
# def query_package(name: str, ecosystem: str, version: str) -> dict[str, Any]:
#     """Query vulnerabilities for a specific package version.

#     Args:
#         name: Package name (e.g., 'requests', 'lodash')
#         ecosystem: Package ecosystem (e.g., 'PyPI', 'npm', 'Go', 'Maven', 'RubyGems')
#         version: Package version to check

#     Returns:
#         Dict with vulnerabilities list, each containing id, summary, details, severity, etc.
#     """
#     logger.info(f"[OSV] {_caller_module} -> query_package: {ecosystem}/{name}@{version}")
#     vulns = _get_client().query_package(name, ecosystem, version)
#     return {"vulnerabilities": [v.model_dump() for v in vulns], "count": len(vulns)}


# @mcp.tool()
# def query_purl(purl: str, version: str | None = None) -> dict[str, Any]:
#     """Query vulnerabilities using a Package URL (purl).

#     Args:
#         purl: Package URL (e.g., 'pkg:pypi/requests', 'pkg:npm/lodash')
#         version: Optional version (if not included in purl)

#     Returns:
#         Dict with vulnerabilities list
#     """
#     logger.info(f"[OSV] {_caller_module} -> query_purl: {purl}@{version or 'latest'}")
#     vulns = _get_client().query_purl(purl, version)
#     return {"vulnerabilities": [v.model_dump() for v in vulns], "count": len(vulns)}


# @mcp.tool()
# def query_commit(commit_hash: str) -> dict[str, Any]:
#     """Query vulnerabilities for a git commit hash.

#     Args:
#         commit_hash: Git commit SHA hash

#     Returns:
#         Dict with vulnerabilities list affecting the commit
#     """
#     logger.info(f"[OSV] {_caller_module} -> query_commit: {commit_hash[:8]}")
#     vulns = _get_client().query_commit(commit_hash)
#     return {"vulnerabilities": [v.model_dump() for v in vulns], "count": len(vulns)}


# @mcp.tool()
# def get_vulnerability(osv_id: str) -> dict[str, Any]:
#     """Get full details of a specific vulnerability by its ID.

#     Args:
#         osv_id: OSV vulnerability ID (e.g., 'GHSA-xxxx-xxxx-xxxx', 'CVE-2021-xxxx')

#     Returns:
#         Dict with full vulnerability details including affected packages, severity, references
#     """
#     logger.info(f"[OSV] {_caller_module} -> get_vulnerability: {osv_id}")
#     vuln = _get_client().get_vulnerability(osv_id)
#     return vuln.model_dump()


# @lru_cache(maxsize=512)
# def _scan_package_cached(name: str, ecosystem: str, version: str) -> tuple:
#     """Cached version of scan_package."""
#     result = _get_client().scan_package(name, ecosystem, version)
#     return tuple(sorted(result.model_dump().items()))


# @mcp.tool()
# def scan_package(name: str, ecosystem: str, version: str) -> dict[str, Any]:
#     """Scan a single package for vulnerabilities.

#     Args:
#         name: Package name
#         ecosystem: Package ecosystem (PyPI, npm, Go, Maven, RubyGems, crates.io, etc.)
#         version: Package version

#     Returns:
#         Dict with package_name, ecosystem, version, vulnerabilities, is_vulnerable, count
#     """
#     logger.info(f"[OSV] {_caller_module} -> scan_package: {ecosystem}/{name}@{version}")
#     return dict(_scan_package_cached(name, ecosystem, version))


@mcp.tool()
def scan_dependencies(dependencies: list[dict[str, str]]) -> dict[str, Any]:
    """Scan multiple dependencies for vulnerabilities using batch querying.

    Args:
        dependencies: List of dependency dicts, each with 'name', 'ecosystem', 'version'
                     Example: [{"name": "requests", "ecosystem": "PyPI", "version": "2.25.0"}]

    Returns:
        Dict with scan summary: total_packages, vulnerable_packages, total_vulnerabilities,
        and detailed results for each package
    """
    logger.info(f"[OSV] {_caller_module} -> scan_dependencies: {len(dependencies)} packages")
    summary = _get_client().scan_dependencies(dependencies)
    return summary.model_dump()


# Individual scan tools are commented out to encourage batch scanning via scan_dependencies()
# which uses the OSV batch API and is more efficient for multiple packages.

# @mcp.tool()
# def scan_pypi_package(name: str, version: str) -> dict[str, Any]:
#     """Scan a PyPI (Python) package for vulnerabilities.

#     Args:
#         name: PyPI package name (e.g., 'requests', 'django', 'flask')
#         version: Package version

#     Returns:
#         Dict with scan result including vulnerabilities found
#     """
#     logger.info(f"[OSV] {_caller_module} -> scan_pypi_package: {name}@{version}")
#     return dict(_scan_package_cached(name, "PyPI", version))


# @mcp.tool()
# def scan_npm_package(name: str, version: str) -> dict[str, Any]:
#     """Scan an npm (JavaScript/Node.js) package for vulnerabilities.

#     Args:
#         name: npm package name (e.g., 'lodash', 'express', 'react')
#         version: Package version

#     Returns:
#         Dict with scan result including vulnerabilities found
#     """
#     logger.info(f"[OSV] {_caller_module} -> scan_npm_package: {name}@{version}")
#     return dict(_scan_package_cached(name, "npm", version))


# @mcp.tool()
# def scan_go_package(name: str, version: str) -> dict[str, Any]:
#     """Scan a Go module for vulnerabilities.

#     Args:
#         name: Go module name (e.g., 'github.com/gin-gonic/gin')
#         version: Module version

#     Returns:
#         Dict with scan result including vulnerabilities found
#     """
#     logger.info(f"[OSV] {_caller_module} -> scan_go_package: {name}@{version}")
#     return dict(_scan_package_cached(name, "Go", version))


# @lru_cache(maxsize=256)
# def _scan_maven_cached(group_id: str, artifact_id: str, version: str) -> tuple:
#     """Cached version of scan_maven_package."""
#     result = _get_client().scan_maven_package(group_id, artifact_id, version)
#     return tuple(sorted(result.model_dump().items()))


# @mcp.tool()
# def scan_maven_package(group_id: str, artifact_id: str, version: str) -> dict[str, Any]:
#     """Scan a Maven (Java) package for vulnerabilities.

#     Args:
#         group_id: Maven group ID (e.g., 'org.apache.logging.log4j')
#         artifact_id: Maven artifact ID (e.g., 'log4j-core')
#         version: Package version

#     Returns:
#         Dict with scan result including vulnerabilities found
#     """
#     logger.info(f"[OSV] {_caller_module} -> scan_maven_package: {group_id}:{artifact_id}@{version}")
#     return dict(_scan_maven_cached(group_id, artifact_id, version))


# @mcp.tool()
# def scan_rubygems_package(name: str, version: str) -> dict[str, Any]:
#     """Scan a RubyGems package for vulnerabilities.

#     Args:
#         name: Gem name (e.g., 'rails', 'nokogiri')
#         version: Gem version

#     Returns:
#         Dict with scan result including vulnerabilities found
#     """
#     logger.info(f"[OSV] {_caller_module} -> scan_rubygems_package: {name}@{version}")
#     return dict(_scan_package_cached(name, "RubyGems", version))


# @mcp.tool()
# def scan_cargo_package(name: str, version: str) -> dict[str, Any]:
#     """Scan a Cargo (Rust) crate for vulnerabilities.

#     Args:
#         name: Crate name (e.g., 'serde', 'tokio')
#         version: Crate version

#     Returns:
#         Dict with scan result including vulnerabilities found
#     """
#     logger.info(f"[OSV] {_caller_module} -> scan_cargo_package: {name}@{version}")
#     return dict(_scan_package_cached(name, "crates.io", version))


if __name__ == "__main__":
    # Suppress noisy MCP server "Processing request" logs
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)
    
    _client = OSVClient()
    mcp.run()
