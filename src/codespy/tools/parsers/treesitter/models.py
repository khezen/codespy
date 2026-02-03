"""Data models for tree-sitter code analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionInfo:
    """Information about a function/method definition."""

    name: str
    file: str
    line_start: int
    line_end: int
    parameters: list[str]
    return_type: str | None = None
    is_method: bool = False
    receiver_type: str | None = None  # For Go methods
    docstring: str | None = None


@dataclass
class CallInfo:
    """Information about a function call."""

    function_name: str
    file: str
    line_number: int
    line_content: str
    arguments_count: int
    caller_function: str | None = None  # Function containing this call


@dataclass
class SymbolInfo:
    """Information about a symbol (variable, type, etc.)."""

    name: str
    kind: str  # function, class, variable, type, etc.
    file: str
    line_number: int
    scope: str | None = None


# ============================================================================
# Terraform/HCL Models
# ============================================================================


@dataclass
class TerraformResourceInfo:
    """Information about a Terraform resource block."""

    resource_type: str  # e.g., "aws_instance", "google_compute_instance"
    resource_name: str  # e.g., "web", "main"
    file: str
    line_start: int
    line_end: int
    provider: str | None = None  # e.g., "aws", "google", "azurerm"
    attributes: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class TerraformDataSourceInfo:
    """Information about a Terraform data source block."""

    data_type: str  # e.g., "aws_ami", "google_compute_image"
    data_name: str
    file: str
    line_start: int
    line_end: int
    provider: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerraformVariableInfo:
    """Information about a Terraform variable declaration."""

    name: str
    file: str
    line_start: int
    line_end: int
    var_type: str | None = None  # e.g., "string", "list(string)", "map(any)"
    default: Any = None
    description: str | None = None
    sensitive: bool = False
    validation: str | None = None


@dataclass
class TerraformOutputInfo:
    """Information about a Terraform output declaration."""

    name: str
    file: str
    line_start: int
    line_end: int
    value_expression: str | None = None
    description: str | None = None
    sensitive: bool = False


@dataclass
class TerraformModuleCallInfo:
    """Information about a Terraform module call."""

    name: str  # Module alias
    source: str  # Module source path/registry
    file: str
    line_start: int
    line_end: int
    version: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerraformProviderInfo:
    """Information about a Terraform provider configuration."""

    name: str  # e.g., "aws", "google"
    file: str
    line_start: int
    line_end: int
    alias: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerraformLocalInfo:
    """Information about a Terraform locals block entry."""

    name: str
    file: str
    line_number: int
    value_expression: str | None = None


@dataclass
class TerraformBlockInfo:
    """Generic container for all Terraform block types in a file."""

    file: str
    resources: list[TerraformResourceInfo] = field(default_factory=list)
    data_sources: list[TerraformDataSourceInfo] = field(default_factory=list)
    variables: list[TerraformVariableInfo] = field(default_factory=list)
    outputs: list[TerraformOutputInfo] = field(default_factory=list)
    modules: list[TerraformModuleCallInfo] = field(default_factory=list)
    providers: list[TerraformProviderInfo] = field(default_factory=list)
    locals: list[TerraformLocalInfo] = field(default_factory=list)
