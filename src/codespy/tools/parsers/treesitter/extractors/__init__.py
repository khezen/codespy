"""Language-specific extractors for tree-sitter parsing."""

from codespy.tools.parsers.treesitter.extractors.go import GoExtractor
from codespy.tools.parsers.treesitter.extractors.java import JavaExtractor
from codespy.tools.parsers.treesitter.extractors.javascript import JavaScriptExtractor
from codespy.tools.parsers.treesitter.extractors.kotlin import KotlinExtractor
from codespy.tools.parsers.treesitter.extractors.objc import ObjCExtractor
from codespy.tools.parsers.treesitter.extractors.python import PythonExtractor
from codespy.tools.parsers.treesitter.extractors.rust import RustExtractor
from codespy.tools.parsers.treesitter.extractors.swift import SwiftExtractor
from codespy.tools.parsers.treesitter.extractors.terraform import TerraformExtractor

__all__ = [
    "GoExtractor",
    "JavaExtractor",
    "JavaScriptExtractor",
    "KotlinExtractor",
    "ObjCExtractor",
    "PythonExtractor",
    "RustExtractor",
    "SwiftExtractor",
    "TerraformExtractor",
]
