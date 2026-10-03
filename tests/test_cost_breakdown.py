"""Tests for cost breakdown functionality: Review/Memory split tables."""

import pytest

from codespy.config_memory import MEMORY_UNIT_PREFIX
from codespy.workflows.review.models import ReviewResult, SignatureStatsResult


class TestSignatureStatsResultIsMemory:
    """Tests for SignatureStatsResult.is_memory property."""

    def test_is_memory_true_for_memory_prefixed_names(self):
        """Names starting with MEMORY_UNIT_PREFIX are memory signatures."""
        assert SignatureStatsResult(name=f"{MEMORY_UNIT_PREFIX}retain").is_memory is True
        assert SignatureStatsResult(name=f"{MEMORY_UNIT_PREFIX}cartographer").is_memory is True
        assert SignatureStatsResult(name=f"{MEMORY_UNIT_PREFIX}prefrontal").is_memory is True

    def test_is_memory_false_for_review_names(self):
        """Names not starting with MEMORY_UNIT_PREFIX are review signatures."""
        assert SignatureStatsResult(name="code_review").is_memory is False
        assert SignatureStatsResult(name="scope").is_memory is False
        assert SignatureStatsResult(name="summary").is_memory is False
        assert SignatureStatsResult(name="doc").is_memory is False
        assert SignatureStatsResult(name="supply_chain").is_memory is False
        assert SignatureStatsResult(name="audit").is_memory is False


class TestReviewResultCostBreakdownMarkdownLines:
    """Tests for ReviewResult.cost_breakdown_markdown_lines."""

    @staticmethod
    def _make_review_stats(name: str, cost: float, calls: int = 1) -> SignatureStatsResult:
        """Create a SignatureStatsResult with minimal fields set."""
        return SignatureStatsResult(
            name=name,
            cost=cost,
            tokens=100 * calls,
            call_count=calls,
            duration_seconds=1.5 * calls,
            input_tokens=50 * calls,
            output_tokens=50 * calls,
            input_cost=cost * 0.6,
            output_cost=cost * 0.4,
        )

    def test_empty_signature_stats_returns_empty_list(self):
        """Empty signature_stats should produce no breakdown."""
        result = ReviewResult(
            pr_number=1,
            pr_title="Test",
            pr_url="http://example.com/1",
            repo="test/repo",
            model_used="gpt-4",
            signature_stats=[],
        )
        assert result.cost_breakdown_markdown_lines() == []

    def test_mixed_stats_render_two_tables_review_first(self):
        """Mixed stats render Review table first, then Memory."""
        stats = [
            self._make_review_stats("scope", 0.5),
            self._make_review_stats("code_review", 1.0),
            self._make_review_stats(f"{MEMORY_UNIT_PREFIX}retain", 0.3),
            self._make_review_stats(f"{MEMORY_UNIT_PREFIX}prefrontal", 0.2),
        ]
        result = ReviewResult(
            pr_number=1,
            pr_title="Test",
            pr_url="http://example.com/1",
            repo="test/repo",
            model_used="gpt-4",
            signature_stats=stats,
        )
        lines = result.cost_breakdown_markdown_lines()
        text = "\n".join(lines)

        # Both tables should be present
        assert "### Review" in text
        assert "### Memory" in text

        # Review should come before Memory
        assert text.index("### Review") < text.index("### Memory")

        # Each signature should appear in its correct table
        # Review signatures
        assert "| scope |" in text
        assert "| code_review |" in text
        # Memory signatures
        assert f"| {MEMORY_UNIT_PREFIX}retain |" in text
        assert f"| {MEMORY_UNIT_PREFIX}prefrontal |" in text

    def test_rows_sorted_by_cost_descending(self):
        """Rows within each table should be sorted by cost descending."""
        stats = [
            self._make_review_stats("scope", 0.1),
            self._make_review_stats("code_review", 0.5),
            self._make_review_stats("summary", 0.3),
        ]
        result = ReviewResult(
            pr_number=1,
            pr_title="Test",
            pr_url="http://example.com/1",
            repo="test/repo",
            model_used="gpt-4",
            signature_stats=stats,
        )
        lines = result.cost_breakdown_markdown_lines()

        # Find the review section lines
        review_start = next(i for i, line in enumerate(lines) if "### Review" in line)
        # Find the first subtotal after review section
        subtotal_idx = next(
            i for i in range(review_start, len(lines)) if "**Subtotal:**" in lines[i]
        )

        # Extract table body (between header and subtotal)
        # Skip header lines (heading, blank, header, separator)
        table_lines = lines[review_start + 4 : subtotal_idx]
        # Filter out empty lines
        table_lines = [l for l in table_lines if l and l.startswith("|")]

        # Check order: code_review (0.5) should come before summary (0.3) before scope (0.1)
        names = [line.split("|")[1].strip() for line in table_lines]
        assert names == ["code_review", "summary", "scope"]

    def test_subtotals_calculated_correctly(self):
        """Subtotals should sum costs and calls for each group."""
        stats = [
            self._make_review_stats("scope", 0.5, calls=2),
            self._make_review_stats("code_review", 1.0, calls=3),
            self._make_review_stats(f"{MEMORY_UNIT_PREFIX}retain", 0.3, calls=1),
            self._make_review_stats(f"{MEMORY_UNIT_PREFIX}prefrontal", 0.2, calls=2),
        ]
        result = ReviewResult(
            pr_number=1,
            pr_title="Test",
            pr_url="http://example.com/1",
            repo="test/repo",
            model_used="gpt-4",
            signature_stats=stats,
        )
        lines = result.cost_breakdown_markdown_lines()

        # Find subtotal lines
        subtotal_lines = [l for l in lines if "**Subtotal:**" in l]
        assert len(subtotal_lines) == 2

        # Review subtotal: 1.5 cost, 5 calls
        review_subtotal = subtotal_lines[0]
        assert "1.5000" in review_subtotal
        assert "5" in review_subtotal  # LLM Calls: 5

        # Memory subtotal: 0.5 cost, 3 calls
        memory_subtotal = subtotal_lines[1]
        assert "0.5000" in memory_subtotal
        assert "3" in memory_subtotal  # LLM Calls: 3

    def test_no_cache_references_in_output(self):
        """Neither to_markdown nor cost_breakdown_markdown_lines should contain 'Cache'."""
        stats = [
            self._make_review_stats("scope", 0.5),
            self._make_review_stats(f"{MEMORY_UNIT_PREFIX}retain", 0.3),
        ]
        result = ReviewResult(
            pr_number=1,
            pr_title="Test",
            pr_url="http://example.com/1",
            repo="test/repo",
            model_used="gpt-4",
            signature_stats=stats,
        )

        # Check cost_breakdown_markdown_lines
        lines = result.cost_breakdown_markdown_lines()
        text = "\n".join(lines)
        assert "Cache" not in text
        assert "cache_read" not in text.lower()
        assert "cache_write" not in text.lower()

        # Check full to_markdown output
        full_md = result.to_markdown()
        assert "Cache" not in full_md
        assert "cache_read" not in full_md.lower()
        assert "cache_write" not in full_md.lower()


class TestGitReporterCostBreakdown:
    """Tests for GitReporter cost breakdown integration.

    These tests use cost_breakdown_markdown_lines() which is what
    GitReporter._build_review_body uses internally for the cost tables.
    """

    @staticmethod
    def _make_review_stats(name: str, cost: float, calls: int = 1) -> SignatureStatsResult:
        return SignatureStatsResult(
            name=name,
            cost=cost,
            tokens=100 * calls,
            call_count=calls,
            duration_seconds=1.5 * calls,
            input_tokens=50 * calls,
            output_tokens=50 * calls,
            input_cost=cost * 0.6,
            output_cost=cost * 0.4,
        )

    def test_build_review_body_no_cache_references(self):
        """Cost breakdown should not contain 'Cache' references."""
        stats = [
            self._make_review_stats("scope", 0.5),
            self._make_review_stats(f"{MEMORY_UNIT_PREFIX}retain", 0.3),
        ]
        result = ReviewResult(
            pr_number=1,
            pr_title="Test",
            pr_url="http://example.com/1",
            repo="test/repo",
            model_used="gpt-4",
            signature_stats=stats,
            total_cost=0.8,
            llm_calls=2,
        )

        lines = result.cost_breakdown_markdown_lines("####")
        body = "\n".join(lines)

        assert "Cache" not in body
        assert "cache_read" not in body.lower()
        assert "cache_write" not in body.lower()
        # Should use #### heading level for cost tables
        assert "#### Review" in body or "### Review" in body

    def test_only_review_stats_no_memory_heading(self):
        """When only review stats exist, no Memory heading should appear."""
        stats = [
            self._make_review_stats("scope", 0.5),
            self._make_review_stats("code_review", 1.0),
        ]
        result = ReviewResult(
            pr_number=1,
            pr_title="Test",
            pr_url="http://example.com/1",
            repo="test/repo",
            model_used="gpt-4",
            signature_stats=stats,
            total_cost=1.5,
            llm_calls=2,
        )
        lines = result.cost_breakdown_markdown_lines()
        text = "\n".join(lines)

        assert "### Review" in text or "#### Review" in text  # Review heading should be present
        assert "### Memory" not in text and "#### Memory" not in text  # Memory heading should NOT appear

    def test_only_memory_stats_no_review_heading(self):
        """When only memory stats exist, no Review heading should appear."""
        stats = [
            self._make_review_stats(f"{MEMORY_UNIT_PREFIX}retain", 0.3),
            self._make_review_stats(f"{MEMORY_UNIT_PREFIX}prefrontal", 0.2),
        ]
        result = ReviewResult(
            pr_number=1,
            pr_title="Test",
            pr_url="http://example.com/1",
            repo="test/repo",
            model_used="gpt-4",
            signature_stats=stats,
            total_cost=0.5,
            llm_calls=2,
        )
        lines = result.cost_breakdown_markdown_lines()
        text = "\n".join(lines)

        assert "### Memory" in text or "#### Memory" in text
        assert "### Review" not in text and "#### Review" not in text


class TestReviewResultNoCacheFields:
    """Tests that cache fields are not present in models."""

    def test_signature_stats_result_no_cache_fields(self):
        """SignatureStatsResult should not have cache fields."""
        stats = SignatureStatsResult(name="test", cost=0.5)
        d = stats.model_dump()
        assert "cache_read_tokens" not in d
        assert "cache_write_tokens" not in d

    def test_cost_table_columns(self):
        """Cost table should have correct columns (no Cache columns)."""
        stats = [
            SignatureStatsResult(
                name="test",
                cost=0.5,
                tokens=100,
                call_count=2,
                input_tokens=50,
                output_tokens=50,
                input_cost=0.3,
                output_cost=0.2,
            ),
        ]
        result = ReviewResult(
            pr_number=1,
            pr_title="Test",
            pr_url="http://example.com/1",
            repo="test/repo",
            model_used="gpt-4",
            signature_stats=stats,
        )
        lines = result.cost_breakdown_markdown_lines()
        # Find the header line
        header_line = next(l for l in lines if "| Signature |" in l)
        # Should have In Tokens, Out Tokens, In Cost, Out Cost, Calls, Duration
        assert "Signature" in header_line
        assert "In Tokens" in header_line
        assert "Out Tokens" in header_line
        assert "In Cost" in header_line
        assert "Out Cost" in header_line
        assert "Calls" in header_line
        assert "Duration" in header_line
        # Should NOT have cache columns
        assert "Cache Read" not in header_line
        assert "Cache Write" not in header_line
