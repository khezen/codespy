"""OSV (Open Source Vulnerabilities) API client."""

import logging
from typing import Any

import httpx

from codespy.tools.cyber.osv.models import (
    BatchQueryResponse,
    BatchQueryResult,
    PackageQuery,
    ScanResult,
    ScanSummary,
    Vulnerability,
    VulnerabilityQuery,
    VulnerabilityResponse,
)

logger = logging.getLogger(__name__)

# OSV API base URL
OSV_API_BASE_URL = "https://api.osv.dev"


class OSVClient:
    """Client for interacting with the OSV (Open Source Vulnerabilities) API.

    The OSV API is free and does not require authentication.
    Documentation: https://google.github.io/osv.dev/api/
    """

    def __init__(
        self,
        base_url: str = OSV_API_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the OSV client.

        Args:
            base_url: Base URL for the OSV API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the OSV API.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint (e.g., /v1/query)
            json_data: JSON body for POST requests

        Returns:
            Response JSON as dictionary

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url}{endpoint}"

        with httpx.Client(timeout=self.timeout) as client:
            if method.upper() == "GET":
                response = client.get(url)
            elif method.upper() == "POST":
                response = client.post(url, json=json_data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

    def query(self, query: VulnerabilityQuery) -> VulnerabilityResponse:
        """Query vulnerabilities for a package or commit.

        Args:
            query: Vulnerability query parameters

        Returns:
            VulnerabilityResponse with matching vulnerabilities
        """
        try:
            data = self._make_request("POST", "/v1/query", query.to_request_dict())
            return VulnerabilityResponse.model_validate(data)
        except httpx.HTTPStatusError as e:
            logger.error(f"OSV API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error querying OSV: {e}")
            raise

    def query_package(
        self,
        name: str,
        ecosystem: str,
        version: str,
    ) -> list[Vulnerability]:
        """Query vulnerabilities for a specific package version.

        Args:
            name: Package name
            ecosystem: Package ecosystem (e.g., PyPI, npm, Go)
            version: Package version

        Returns:
            List of vulnerabilities affecting the package
        """
        query = VulnerabilityQuery(
            package=PackageQuery(name=name, ecosystem=ecosystem),
            version=version,
        )

        all_vulns: list[Vulnerability] = []
        response = self.query(query)
        all_vulns.extend(response.vulns)

        # Handle pagination
        while response.next_page_token:
            query.page_token = response.next_page_token
            response = self.query(query)
            all_vulns.extend(response.vulns)

        return all_vulns

    def query_purl(
        self,
        purl: str,
        version: str | None = None,
    ) -> list[Vulnerability]:
        """Query vulnerabilities using a Package URL (purl).

        Args:
            purl: Package URL (e.g., pkg:pypi/requests)
            version: Optional version (if not included in purl)

        Returns:
            List of vulnerabilities affecting the package
        """
        query = VulnerabilityQuery(
            package=PackageQuery(purl=purl),
            version=version,
        )

        all_vulns: list[Vulnerability] = []
        response = self.query(query)
        all_vulns.extend(response.vulns)

        # Handle pagination
        while response.next_page_token:
            query.page_token = response.next_page_token
            response = self.query(query)
            all_vulns.extend(response.vulns)

        return all_vulns

    def query_commit(self, commit_hash: str) -> list[Vulnerability]:
        """Query vulnerabilities for a git commit hash.

        Args:
            commit_hash: Git commit SHA hash

        Returns:
            List of vulnerabilities affecting the commit
        """
        query = VulnerabilityQuery(commit=commit_hash)

        all_vulns: list[Vulnerability] = []
        response = self.query(query)
        all_vulns.extend(response.vulns)

        # Handle pagination
        while response.next_page_token:
            query.page_token = response.next_page_token
            response = self.query(query)
            all_vulns.extend(response.vulns)

        return all_vulns

    def query_batch(
        self,
        queries: list[dict[str, Any]],
    ) -> BatchQueryResponse:
        """Query vulnerabilities for multiple packages at once.

        This is more efficient than making individual queries.

        Args:
            queries: List of query dictionaries, each containing:
                - name: Package name
                - ecosystem: Package ecosystem
                - version: Package version
                OR
                - purl: Package URL
                - version: Optional version
                OR
                - commit: Git commit hash

        Returns:
            BatchQueryResponse with results for each query

        Example:
            queries = [
                {"name": "requests", "ecosystem": "PyPI", "version": "2.25.0"},
                {"name": "lodash", "ecosystem": "npm", "version": "4.17.20"},
                {"purl": "pkg:maven/org.apache.logging.log4j/log4j-core", "version": "2.14.0"},
            ]
            results = client.query_batch(queries)
        """
        # Convert queries to API format
        formatted_queries = []
        for q in queries:
            query_dict: dict[str, Any] = {}

            if "commit" in q:
                query_dict["commit"] = q["commit"]
            else:
                if "version" in q:
                    query_dict["version"] = q["version"]

                package_dict: dict[str, str] = {}
                if "name" in q:
                    package_dict["name"] = q["name"]
                if "ecosystem" in q:
                    package_dict["ecosystem"] = q["ecosystem"]
                if "purl" in q:
                    package_dict["purl"] = q["purl"]

                if package_dict:
                    query_dict["package"] = package_dict

            formatted_queries.append(query_dict)

        try:
            data = self._make_request(
                "POST",
                "/v1/querybatch",
                {"queries": formatted_queries},
            )
            return BatchQueryResponse.model_validate(data)
        except httpx.HTTPStatusError as e:
            logger.error(f"OSV batch API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error in batch query: {e}")
            raise

    def get_vulnerability(self, osv_id: str) -> Vulnerability:
        """Get full details of a specific vulnerability by its ID.

        Args:
            osv_id: OSV vulnerability ID (e.g., GHSA-xxxx-xxxx-xxxx, CVE-xxxx-xxxx)

        Returns:
            Vulnerability details
        """
        try:
            data = self._make_request("GET", f"/v1/vulns/{osv_id}")
            return Vulnerability.model_validate(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Vulnerability not found: {osv_id}")
            logger.error(f"OSV API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error fetching vulnerability {osv_id}: {e}")
            raise

    def scan_package(
        self,
        name: str,
        ecosystem: str,
        version: str,
    ) -> ScanResult:
        """Scan a single package for vulnerabilities.

        Args:
            name: Package name
            ecosystem: Package ecosystem
            version: Package version

        Returns:
            ScanResult with vulnerabilities found
        """
        try:
            vulns = self.query_package(name, ecosystem, version)
            return ScanResult(
                package_name=name,
                ecosystem=ecosystem,
                version=version,
                vulnerabilities=vulns,
            )
        except Exception as e:
            logger.error(f"Error scanning {ecosystem}/{name}@{version}: {e}")
            return ScanResult(
                package_name=name,
                ecosystem=ecosystem,
                version=version,
                error=str(e),
            )

    def scan_dependencies(
        self,
        dependencies: list[dict[str, str]],
    ) -> ScanSummary:
        """Scan multiple dependencies for vulnerabilities.

        Uses batch querying for efficiency.

        Args:
            dependencies: List of dependency dictionaries with:
                - name: Package name
                - ecosystem: Package ecosystem
                - version: Package version

        Returns:
            ScanSummary with all scan results

        Example:
            deps = [
                {"name": "requests", "ecosystem": "PyPI", "version": "2.25.0"},
                {"name": "django", "ecosystem": "PyPI", "version": "3.1.0"},
            ]
            summary = client.scan_dependencies(deps)
        """
        results: list[ScanResult] = []
        total_vulns = 0
        vulnerable_count = 0
        error_count = 0

        # Use batch query for efficiency
        try:
            batch_response = self.query_batch(dependencies)

            for i, (dep, query_result) in enumerate(
                zip(dependencies, batch_response.results, strict=False)
            ):
                result = ScanResult(
                    package_name=dep["name"],
                    ecosystem=dep["ecosystem"],
                    version=dep["version"],
                    vulnerabilities=query_result.vulns,
                )
                results.append(result)

                if result.is_vulnerable:
                    vulnerable_count += 1
                    total_vulns += result.vulnerability_count

        except Exception as e:
            logger.error(f"Batch query failed, falling back to individual queries: {e}")
            # Fallback to individual queries
            for dep in dependencies:
                result = self.scan_package(
                    name=dep["name"],
                    ecosystem=dep["ecosystem"],
                    version=dep["version"],
                )
                results.append(result)

                if result.error:
                    error_count += 1
                elif result.is_vulnerable:
                    vulnerable_count += 1
                    total_vulns += result.vulnerability_count

        return ScanSummary(
            results=results,
            total_packages=len(dependencies),
            vulnerable_packages=vulnerable_count,
            total_vulnerabilities=total_vulns,
            scan_errors=error_count,
        )

    def scan_pypi_package(self, name: str, version: str) -> ScanResult:
        """Convenience method to scan a PyPI package.

        Args:
            name: PyPI package name
            version: Package version

        Returns:
            ScanResult with vulnerabilities found
        """
        return self.scan_package(name, "PyPI", version)

    def scan_npm_package(self, name: str, version: str) -> ScanResult:
        """Convenience method to scan an npm package.

        Args:
            name: npm package name
            version: Package version

        Returns:
            ScanResult with vulnerabilities found
        """
        return self.scan_package(name, "npm", version)

    def scan_go_package(self, name: str, version: str) -> ScanResult:
        """Convenience method to scan a Go package.

        Args:
            name: Go module name
            version: Module version

        Returns:
            ScanResult with vulnerabilities found
        """
        return self.scan_package(name, "Go", version)

    def scan_maven_package(self, group_id: str, artifact_id: str, version: str) -> ScanResult:
        """Convenience method to scan a Maven package.

        Args:
            group_id: Maven group ID
            artifact_id: Maven artifact ID
            version: Package version

        Returns:
            ScanResult with vulnerabilities found
        """
        name = f"{group_id}:{artifact_id}"
        return self.scan_package(name, "Maven", version)

    def scan_rubygems_package(self, name: str, version: str) -> ScanResult:
        """Convenience method to scan a RubyGems package.

        Args:
            name: Gem name
            version: Gem version

        Returns:
            ScanResult with vulnerabilities found
        """
        return self.scan_package(name, "RubyGems", version)

    def scan_cargo_package(self, name: str, version: str) -> ScanResult:
        """Convenience method to scan a Cargo (Rust) package.

        Args:
            name: Crate name
            version: Crate version

        Returns:
            ScanResult with vulnerabilities found
        """
        return self.scan_package(name, "crates.io", version)