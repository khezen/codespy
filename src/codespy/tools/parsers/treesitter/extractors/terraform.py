"""Terraform/HCL extractor for infrastructure-as-code analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codespy.tools.parsers.treesitter.base_extractor import BaseExtractor
from codespy.tools.parsers.treesitter.models import (
    FunctionInfo,
    TerraformBlockInfo,
    TerraformDataSourceInfo,
    TerraformLocalInfo,
    TerraformModuleCallInfo,
    TerraformOutputInfo,
    TerraformProviderInfo,
    TerraformResourceInfo,
    TerraformVariableInfo,
)


class TerraformExtractor(BaseExtractor):
    """Extract Terraform/HCL block definitions from source code.

    Supports extracting:
    - Resources (resource "type" "name" { })
    - Data sources (data "type" "name" { })
    - Variables (variable "name" { })
    - Outputs (output "name" { })
    - Modules (module "name" { })
    - Providers (provider "name" { })
    - Locals (locals { })
    """

    # Provider prefixes for common cloud providers
    PROVIDER_PREFIXES = {
        "aws_": "aws",
        "azurerm_": "azurerm",
        "google_": "google",
        "kubernetes_": "kubernetes",
        "helm_": "helm",
        "null_": "null",
        "random_": "random",
        "local_": "local",
        "tls_": "tls",
        "vault_": "vault",
        "datadog_": "datadog",
        "newrelic_": "newrelic",
        "github_": "github",
        "gitlab_": "gitlab",
        "cloudflare_": "cloudflare",
        "digitalocean_": "digitalocean",
        "oci_": "oci",
        "alicloud_": "alicloud",
    }

    def extract_functions(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[FunctionInfo]:
        """Terraform doesn't have functions - return empty list.

        This method exists for API compatibility with BaseExtractor.
        Use extract_terraform_blocks() for Terraform-specific extraction.
        """
        return []

    def extract_terraform_blocks(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> TerraformBlockInfo:
        """Extract all Terraform blocks from an HCL file.

        Args:
            node: Root AST node
            file_path: Path to the source file
            source: Raw source bytes

        Returns:
            TerraformBlockInfo containing all extracted blocks
        """
        result = TerraformBlockInfo(file=str(file_path))

        def visit(n: Any) -> None:
            if n.type == "block":
                self._parse_block(n, source, file_path, result)
            for child in n.children:
                visit(child)

        visit(node)
        return result

    def _parse_block(
        self,
        node: Any,
        source: bytes,
        file_path: Path,
        result: TerraformBlockInfo,
    ) -> None:
        """Parse a top-level HCL block."""
        # Get block type (first identifier)
        block_type = self._get_block_type(node, source)
        if not block_type:
            return

        if block_type == "resource":
            resource = self._parse_resource_block(node, source, file_path)
            if resource:
                result.resources.append(resource)

        elif block_type == "data":
            data_source = self._parse_data_block(node, source, file_path)
            if data_source:
                result.data_sources.append(data_source)

        elif block_type == "variable":
            variable = self._parse_variable_block(node, source, file_path)
            if variable:
                result.variables.append(variable)

        elif block_type == "output":
            output = self._parse_output_block(node, source, file_path)
            if output:
                result.outputs.append(output)

        elif block_type == "module":
            module = self._parse_module_block(node, source, file_path)
            if module:
                result.modules.append(module)

        elif block_type == "provider":
            provider = self._parse_provider_block(node, source, file_path)
            if provider:
                result.providers.append(provider)

        elif block_type == "locals":
            locals_list = self._parse_locals_block(node, source, file_path)
            result.locals.extend(locals_list)

    def _get_block_type(self, node: Any, source: bytes) -> str | None:
        """Get the type of an HCL block (resource, variable, etc.)."""
        for child in node.children:
            if child.type == "identifier":
                return self._get_node_text(child, source)
        return None

    def _get_block_labels(self, node: Any, source: bytes) -> list[str]:
        """Get the labels of an HCL block (e.g., "aws_instance", "web")."""
        labels: list[str] = []
        skip_first = True  # Skip block type identifier
        for child in node.children:
            if child.type == "identifier":
                if skip_first:
                    skip_first = False
                    continue
                labels.append(self._get_node_text(child, source))
            elif child.type == "string_lit":
                # Remove quotes from string literals
                text = self._get_node_text(child, source)
                labels.append(text.strip('"\''))
        return labels

    def _get_block_body(self, node: Any) -> Any | None:
        """Get the body node of an HCL block."""
        for child in node.children:
            if child.type == "block_body":
                return child
        return None

    def _extract_attributes(self, body_node: Any, source: bytes) -> dict[str, Any]:
        """Extract attributes from a block body."""
        attributes: dict[str, Any] = {}
        if not body_node:
            return attributes

        for child in body_node.children:
            if child.type == "attribute":
                key_node = child.child_by_field_name("key")
                value_node = child.child_by_field_name("value")
                if key_node:
                    key = self._get_node_text(key_node, source)
                    if value_node:
                        value = self._get_node_text(value_node, source)
                        # Try to parse simple values
                        attributes[key] = self._parse_attribute_value(value)
                    else:
                        attributes[key] = None
        return attributes

    def _parse_attribute_value(self, value: str) -> Any:
        """Parse an attribute value to its appropriate Python type."""
        value = value.strip()
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        # Try to parse as number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        # Return as expression string
        return value

    def _infer_provider(self, resource_type: str) -> str | None:
        """Infer the provider from a resource type."""
        for prefix, provider in self.PROVIDER_PREFIXES.items():
            if resource_type.startswith(prefix):
                return provider
        # Fallback: use first part before underscore
        if "_" in resource_type:
            return resource_type.split("_")[0]
        return None

    def _parse_resource_block(
        self,
        node: Any,
        source: bytes,
        file_path: Path,
    ) -> TerraformResourceInfo | None:
        """Parse a resource block."""
        labels = self._get_block_labels(node, source)
        if len(labels) < 2:
            return None

        resource_type = labels[0]
        resource_name = labels[1]
        body = self._get_block_body(node)
        attributes = self._extract_attributes(body, source)

        # Extract depends_on if present
        depends_on: list[str] = []
        if "depends_on" in attributes:
            dep_value = attributes.pop("depends_on")
            if isinstance(dep_value, str):
                # Parse list syntax [resource.a, resource.b]
                dep_value = dep_value.strip("[]")
                depends_on = [d.strip() for d in dep_value.split(",") if d.strip()]

        return TerraformResourceInfo(
            resource_type=resource_type,
            resource_name=resource_name,
            file=str(file_path),
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            provider=self._infer_provider(resource_type),
            attributes=attributes,
            depends_on=depends_on,
        )

    def _parse_data_block(
        self,
        node: Any,
        source: bytes,
        file_path: Path,
    ) -> TerraformDataSourceInfo | None:
        """Parse a data source block."""
        labels = self._get_block_labels(node, source)
        if len(labels) < 2:
            return None

        data_type = labels[0]
        data_name = labels[1]
        body = self._get_block_body(node)
        attributes = self._extract_attributes(body, source)

        return TerraformDataSourceInfo(
            data_type=data_type,
            data_name=data_name,
            file=str(file_path),
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            provider=self._infer_provider(data_type),
            attributes=attributes,
        )

    def _parse_variable_block(
        self,
        node: Any,
        source: bytes,
        file_path: Path,
    ) -> TerraformVariableInfo | None:
        """Parse a variable block."""
        labels = self._get_block_labels(node, source)
        if len(labels) < 1:
            return None

        var_name = labels[0]
        body = self._get_block_body(node)
        attributes = self._extract_attributes(body, source)

        return TerraformVariableInfo(
            name=var_name,
            file=str(file_path),
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            var_type=attributes.get("type"),
            default=attributes.get("default"),
            description=attributes.get("description"),
            sensitive=attributes.get("sensitive", False),
            validation=attributes.get("validation"),
        )

    def _parse_output_block(
        self,
        node: Any,
        source: bytes,
        file_path: Path,
    ) -> TerraformOutputInfo | None:
        """Parse an output block."""
        labels = self._get_block_labels(node, source)
        if len(labels) < 1:
            return None

        output_name = labels[0]
        body = self._get_block_body(node)
        attributes = self._extract_attributes(body, source)

        return TerraformOutputInfo(
            name=output_name,
            file=str(file_path),
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            value_expression=attributes.get("value"),
            description=attributes.get("description"),
            sensitive=attributes.get("sensitive", False),
        )

    def _parse_module_block(
        self,
        node: Any,
        source: bytes,
        file_path: Path,
    ) -> TerraformModuleCallInfo | None:
        """Parse a module call block."""
        labels = self._get_block_labels(node, source)
        if len(labels) < 1:
            return None

        module_name = labels[0]
        body = self._get_block_body(node)
        attributes = self._extract_attributes(body, source)

        # Extract special attributes
        source_path = attributes.pop("source", "")
        version = attributes.pop("version", None)

        return TerraformModuleCallInfo(
            name=module_name,
            source=source_path,
            file=str(file_path),
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            version=version,
            inputs=attributes,  # Remaining attributes are module inputs
        )

    def _parse_provider_block(
        self,
        node: Any,
        source: bytes,
        file_path: Path,
    ) -> TerraformProviderInfo | None:
        """Parse a provider configuration block."""
        labels = self._get_block_labels(node, source)
        if len(labels) < 1:
            return None

        provider_name = labels[0]
        body = self._get_block_body(node)
        attributes = self._extract_attributes(body, source)

        # Extract alias if present
        alias = attributes.pop("alias", None)

        return TerraformProviderInfo(
            name=provider_name,
            file=str(file_path),
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            alias=alias,
            attributes=attributes,
        )

    def _parse_locals_block(
        self,
        node: Any,
        source: bytes,
        file_path: Path,
    ) -> list[TerraformLocalInfo]:
        """Parse a locals block."""
        locals_list: list[TerraformLocalInfo] = []
        body = self._get_block_body(node)
        if not body:
            return locals_list

        for child in body.children:
            if child.type == "attribute":
                key_node = child.child_by_field_name("key")
                value_node = child.child_by_field_name("value")
                if key_node:
                    key = self._get_node_text(key_node, source)
                    value_expr = None
                    if value_node:
                        value_expr = self._get_node_text(value_node, source)
                    locals_list.append(TerraformLocalInfo(
                        name=key,
                        file=str(file_path),
                        line_number=child.start_point[0] + 1,
                        value_expression=value_expr,
                    ))

        return locals_list

    def extract_resources(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[TerraformResourceInfo]:
        """Extract only resource blocks from an HCL file."""
        return self.extract_terraform_blocks(node, file_path, source).resources

    def extract_variables(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[TerraformVariableInfo]:
        """Extract only variable blocks from an HCL file."""
        return self.extract_terraform_blocks(node, file_path, source).variables

    def extract_outputs(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[TerraformOutputInfo]:
        """Extract only output blocks from an HCL file."""
        return self.extract_terraform_blocks(node, file_path, source).outputs

    def extract_modules(
        self,
        node: Any,
        file_path: Path,
        source: bytes,
    ) -> list[TerraformModuleCallInfo]:
        """Extract only module call blocks from an HCL file."""
        return self.extract_terraform_blocks(node, file_path, source).modules

    def find_resource_references(
        self,
        node: Any,
        source: bytes,
    ) -> list[str]:
        """Find all resource references in expressions.

        Looks for patterns like:
        - aws_instance.web.id
        - data.aws_ami.ubuntu.id
        - module.vpc.vpc_id
        - var.instance_type
        - local.common_tags

        Returns:
            List of reference strings found
        """
        references: list[str] = []

        def visit(n: Any) -> None:
            # Look for identifier chains that might be references
            if n.type in ("get_attr", "index_expr", "attr_expr"):
                ref = self._get_node_text(n, source)
                # Filter for likely Terraform references
                if any(ref.startswith(prefix) for prefix in
                       ("aws_", "azurerm_", "google_", "data.", "module.",
                        "var.", "local.", "kubernetes_", "helm_")):
                    references.append(ref)

            for child in n.children:
                visit(child)

        visit(node)
        return references
