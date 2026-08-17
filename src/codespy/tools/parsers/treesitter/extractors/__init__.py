"""Language-specific extractors for tree-sitter parsing."""

from codespy.tools.parsers.treesitter.extractors.bash import BashExtractor
from codespy.tools.parsers.treesitter.extractors.cpp import CppExtractor
from codespy.tools.parsers.treesitter.extractors.csharp import CSharpExtractor
from codespy.tools.parsers.treesitter.extractors.go import GoExtractor
from codespy.tools.parsers.treesitter.extractors.java import JavaExtractor
from codespy.tools.parsers.treesitter.extractors.javascript import JavaScriptExtractor
from codespy.tools.parsers.treesitter.extractors.kotlin import KotlinExtractor
from codespy.tools.parsers.treesitter.extractors.objc import ObjCExtractor
from codespy.tools.parsers.treesitter.extractors.php import PHPExtractor
from codespy.tools.parsers.treesitter.extractors.python import PythonExtractor
from codespy.tools.parsers.treesitter.extractors.ripgrep_fallback import RipgrepHeuristicsExtractor
from codespy.tools.parsers.treesitter.extractors.ruby import RubyExtractor
from codespy.tools.parsers.treesitter.extractors.rust import RustExtractor
from codespy.tools.parsers.treesitter.extractors.swift import SwiftExtractor
from codespy.tools.parsers.treesitter.extractors.terraform import TerraformExtractor

__all__ = [
    "BashExtractor",
    "CppExtractor",
    "CSharpExtractor",
    "GoExtractor",
    "JavaExtractor",
    "JavaScriptExtractor",
    "KotlinExtractor",
    "ObjCExtractor",
    "PHPExtractor",
    "PythonExtractor",
    "RipgrepHeuristicsExtractor",
    "RubyExtractor",
    "RustExtractor",
    "SwiftExtractor",
    "TerraformExtractor",
]
