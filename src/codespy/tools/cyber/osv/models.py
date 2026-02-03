"""Models for OSV (Open Source Vulnerabilities) API."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Ecosystem(str, Enum):
    """Supported package ecosystems in OSV."""

    GO = "Go"
    NPM = "npm"
    PYPI = "PyPI"
    RUBYGEMS = "RubyGems"
    CRATES_IO = "crates.io"
    PACKAGIST = "Packagist"
    MAVEN = "Maven"
    NUGET = "NuGet"
    LINUX = "Linux"
    DEBIAN = "Debian"
    ALPINE = "Alpine"
    HEX = "Hex"
    PUB = "Pub"
    CONAN = "ConanCenter"
    HACKAGE = "Hackage"
    GIT = "GIT"
    OSS_FUZZ = "OSS-Fuzz"
    GITHUB_ACTIONS = "GitHub Actions"
    SWIFT = "SwiftURL"
    BITNAMI = "Bitnami"
    PHOTON_OS = "Photon OS"
    CRAN = "CRAN"
    BIOCONDUCTOR = "Bioconductor"
    ROCKY_LINUX = "Rocky Linux"
    ALMA_LINUX = "AlmaLinux"
    CHAINGUARD = "Chainguard"
    WOLFI = "Wolfi"


class SeverityType(str, Enum):
    """Severity scoring systems."""

    CVSS_V2 = "CVSS_V2"
    CVSS_V3 = "CVSS_V3"
    CVSS_V4 = "CVSS_V4"


class ReferenceType(str, Enum):
    """Types of references in vulnerability records."""

    ADVISORY = "ADVISORY"
    ARTICLE = "ARTICLE"
    DETECTION = "DETECTION"
    DISCUSSION = "DISCUSSION"
    REPORT = "REPORT"
    FIX = "FIX"
    GIT = "GIT"
    INTRODUCED = "INTRODUCED"
    PACKAGE = "PACKAGE"
    EVIDENCE = "EVIDENCE"
    WEB = "WEB"


class Package(BaseModel):
    """Package identifier."""

    name: str = Field(description="Package name")
    ecosystem: str = Field(description="Package ecosystem (e.g., PyPI, npm)")
    purl: str | None = Field(default=None, description="Package URL")


class Severity(BaseModel):
    """Severity score for a vulnerability."""

    type: SeverityType = Field(description="Severity scoring system")
    score: str = Field(description="Severity score value")


class Reference(BaseModel):
    """Reference URL for a vulnerability."""

    type: ReferenceType = Field(description="Type of reference")
    url: str = Field(description="Reference URL")


class RangeEvent(BaseModel):
    """Event in a version range (introduced or fixed)."""

    introduced: str | None = Field(default=None, description="Version where vulnerability was introduced")
    fixed: str | None = Field(default=None, description="Version where vulnerability was fixed")
    last_affected: str | None = Field(default=None, description="Last affected version")
    limit: str | None = Field(default=None, description="Upper limit version")


class Range(BaseModel):
    """Version range for affected packages."""

    type: str = Field(description="Range type (SEMVER, ECOSYSTEM, GIT)")
    repo: str | None = Field(default=None, description="Git repository URL (for GIT type)")
    events: list[RangeEvent] = Field(default_factory=list, description="Range events")


class AffectedPackage(BaseModel):
    """Package affected by a vulnerability."""

    package: Package = Field(description="Affected package")
    ranges: list[Range] = Field(default_factory=list, description="Affected version ranges")
    versions: list[str] = Field(default_factory=list, description="Specific affected versions")
    ecosystem_specific: dict | None = Field(default=None, description="Ecosystem-specific data")
    database_specific: dict | None = Field(default=None, description="Database-specific data")


class Credit(BaseModel):
    """Credit for vulnerability discovery or fix."""

    name: str = Field(description="Name of the credited party")
    contact: list[str] = Field(default_factory=list, description="Contact information")
    type: str | None = Field(default=None, description="Type of credit")


class Vulnerability(BaseModel):
    """OSV vulnerability record."""

    id: str = Field(description="OSV vulnerability ID")
    summary: str | None = Field(default=None, description="Short summary of the vulnerability")
    details: str | None = Field(default=None, description="Detailed description")
    aliases: list[str] = Field(default_factory=list, description="Alternative IDs (CVE, GHSA, etc.)")
    modified: datetime | None = Field(default=None, description="Last modification timestamp")
    published: datetime | None = Field(default=None, description="Publication timestamp")
    withdrawn: datetime | None = Field(default=None, description="Withdrawal timestamp if withdrawn")
    related: list[str] = Field(default_factory=list, description="Related vulnerability IDs")
    severity: list[Severity] = Field(default_factory=list, description="Severity scores")
    affected: list[AffectedPackage] = Field(default_factory=list, description="Affected packages")
    references: list[Reference] = Field(default_factory=list, description="Reference URLs")
    credits: list[Credit] = Field(default_factory=list, description="Credits")
    database_specific: dict | None = Field(default=None, description="Database-specific metadata")
    schema_version: str | None = Field(default=None, description="OSV schema version")

    def get_cvss_score(self) -> str | None:
        """Get the highest CVSS score if available."""
        for sev in self.severity:
            if sev.type in (SeverityType.CVSS_V4, SeverityType.CVSS_V3, SeverityType.CVSS_V2):
                return sev.score
        return None

    def get_cve_id(self) -> str | None:
        """Get CVE ID from aliases if present."""
        for alias in self.aliases:
            if alias.startswith("CVE-"):
                return alias
        return None

    def get_fixed_versions(self, package_name: str | None = None) -> list[str]:
        """Get fixed versions for the vulnerability.

        Args:
            package_name: Optional package name to filter by

        Returns:
            List of fixed versions
        """
        fixed_versions = []
        for affected in self.affected:
            if package_name and affected.package.name != package_name:
                continue
            for range_ in affected.ranges:
                for event in range_.events:
                    if event.fixed:
                        fixed_versions.append(event.fixed)
        return fixed_versions

    def to_markdown(self) -> str:
        """Convert vulnerability to markdown format."""
        lines = [f"### {self.id}"]

        if self.summary:
            lines.append(f"\n**Summary:** {self.summary}")

        cve = self.get_cve_id()
        if cve:
            lines.append(f"\n**CVE:** {cve}")

        cvss = self.get_cvss_score()
        if cvss:
            lines.append(f"\n**CVSS:** {cvss}")

        if self.details:
            lines.append(f"\n**Details:**\n{self.details[:500]}{'...' if len(self.details) > 500 else ''}")

        if self.affected:
            lines.append("\n**Affected Packages:**")
            for affected in self.affected[:5]:
                pkg = affected.package
                versions = ", ".join(affected.versions[:5]) if affected.versions else "See ranges"
                lines.append(f"- {pkg.ecosystem}/{pkg.name}: {versions}")

        if self.references:
            lines.append("\n**References:**")
            for ref in self.references[:3]:
                lines.append(f"- [{ref.type}]({ref.url})")

        return "\n".join(lines)


# Query Models
class PackageQuery(BaseModel):
    """Package specification for queries."""

    name: str | None = Field(default=None, description="Package name")
    ecosystem: str | None = Field(default=None, description="Package ecosystem")
    purl: str | None = Field(default=None, description="Package URL (alternative to name+ecosystem)")


class VulnerabilityQuery(BaseModel):
    """Query parameters for vulnerability lookup."""

    commit: str | None = Field(default=None, description="Git commit hash to query")
    version: str | None = Field(default=None, description="Package version to query")
    package: PackageQuery | None = Field(default=None, description="Package to query")
    page_token: str | None = Field(default=None, description="Pagination token")

    def to_request_dict(self) -> dict:
        """Convert to API request dictionary."""
        result = {}
        if self.commit:
            result["commit"] = self.commit
        if self.version:
            result["version"] = self.version
        if self.package:
            pkg_dict = {}
            if self.package.name:
                pkg_dict["name"] = self.package.name
            if self.package.ecosystem:
                pkg_dict["ecosystem"] = self.package.ecosystem
            if self.package.purl:
                pkg_dict["purl"] = self.package.purl
            if pkg_dict:
                result["package"] = pkg_dict
        if self.page_token:
            result["page_token"] = self.page_token
        return result


# Response Models
class VulnerabilityResponse(BaseModel):
    """Response from vulnerability query."""

    vulns: list[Vulnerability] = Field(default_factory=list, description="List of vulnerabilities")
    next_page_token: str | None = Field(default=None, description="Token for next page of results")


class BatchQueryResult(BaseModel):
    """Result for a single query in a batch request."""

    vulns: list[Vulnerability] = Field(default_factory=list, description="Vulnerabilities for this query")
    next_page_token: str | None = Field(default=None, description="Pagination token for this query")


class BatchQueryResponse(BaseModel):
    """Response from batch vulnerability query."""

    results: list[BatchQueryResult] = Field(default_factory=list, description="Results for each query")


class ScanResult(BaseModel):
    """Result of scanning a package for vulnerabilities."""

    package_name: str = Field(description="Package name that was scanned")
    ecosystem: str = Field(description="Package ecosystem")
    version: str = Field(description="Version that was scanned")
    vulnerabilities: list[Vulnerability] = Field(
        default_factory=list, description="Vulnerabilities found"
    )
    error: str | None = Field(default=None, description="Error message if scan failed")

    @property
    def is_vulnerable(self) -> bool:
        """Check if the package has vulnerabilities."""
        return len(self.vulnerabilities) > 0

    @property
    def vulnerability_count(self) -> int:
        """Get the number of vulnerabilities found."""
        return len(self.vulnerabilities)

    def to_markdown(self) -> str:
        """Convert scan result to markdown format."""
        lines = [f"## {self.ecosystem}/{self.package_name}@{self.version}"]

        if self.error:
            lines.append(f"\n**Error:** {self.error}")
            return "\n".join(lines)

        if not self.is_vulnerable:
            lines.append("\n✅ No vulnerabilities found")
            return "\n".join(lines)

        lines.append(f"\n⚠️ **{self.vulnerability_count} vulnerabilities found**")

        for vuln in self.vulnerabilities:
            lines.append(f"\n{vuln.to_markdown()}")

        return "\n".join(lines)


class ScanSummary(BaseModel):
    """Summary of a vulnerability scan across multiple packages."""

    results: list[ScanResult] = Field(default_factory=list, description="Individual scan results")
    total_packages: int = Field(default=0, description="Total packages scanned")
    vulnerable_packages: int = Field(default=0, description="Packages with vulnerabilities")
    total_vulnerabilities: int = Field(default=0, description="Total vulnerabilities found")
    scan_errors: int = Field(default=0, description="Number of scan errors")

    def to_markdown(self) -> str:
        """Convert scan summary to markdown format."""
        lines = [
            "# OSV Vulnerability Scan Summary",
            "",
            f"- **Packages Scanned:** {self.total_packages}",
            f"- **Vulnerable Packages:** {self.vulnerable_packages}",
            f"- **Total Vulnerabilities:** {self.total_vulnerabilities}",
        ]

        if self.scan_errors > 0:
            lines.append(f"- **Scan Errors:** {self.scan_errors}")

        if self.results:
            lines.append("\n---\n")
            for result in self.results:
                lines.append(result.to_markdown())
                lines.append("\n---\n")

        return "\n".join(lines)