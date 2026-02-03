"""OSV (Open Source Vulnerabilities) API integration for codespy."""

from codespy.tools.cyber.osv.client import OSVClient
from codespy.tools.cyber.osv.server import mcp as osv_mcp
from codespy.tools.cyber.osv.models import (
    AffectedPackage,
    BatchQueryResponse,
    BatchQueryResult,
    Credit,
    Ecosystem,
    Package,
    PackageQuery,
    Range,
    RangeEvent,
    Reference,
    ReferenceType,
    ScanResult,
    ScanSummary,
    Severity,
    SeverityType,
    Vulnerability,
    VulnerabilityQuery,
    VulnerabilityResponse,
)

__all__ = [
    # Client
    "OSVClient",
    # MCP Server
    "osv_mcp",
    # Core Models
    "Vulnerability",
    "AffectedPackage",
    "Package",
    "Severity",
    "Reference",
    "Range",
    "RangeEvent",
    "Credit",
    # Query Models
    "PackageQuery",
    "VulnerabilityQuery",
    # Response Models
    "VulnerabilityResponse",
    "BatchQueryResponse",
    "BatchQueryResult",
    # Scan Models
    "ScanResult",
    "ScanSummary",
    # Enums
    "Ecosystem",
    "SeverityType",
    "ReferenceType",
]