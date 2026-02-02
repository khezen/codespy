"""Data models for tree-sitter code analysis."""

from __future__ import annotations

from dataclasses import dataclass


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