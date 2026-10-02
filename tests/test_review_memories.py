"""Tests for the collapsible memories section of the review output."""

from codespy.workflows.review.models import (
    TRUNCATED_MARKER,
    MemorySection,
    RecalledMemory,
    ReviewResult,
)
from codespy.workflows.review.reporters.git import GitReporter


def _result(
    *memories: tuple[str, list[tuple[str, str]]], recommendation: str | None = "approve"
) -> ReviewResult:
    """Build a ReviewResult with memories from (task, [(section_title, section_body), ...]) tuples."""
    recalled = []
    for task, sections in memories:
        recalled.append(
            RecalledMemory(
                task=task,
                sections=[MemorySection(title=t, text=b) for t, b in sections],
            )
        )
    return ReviewResult(
        pr_number=1,
        pr_title="t",
        pr_url="https://github.com/o/r/pull/1",
        repo="o/r",
        model_used="m",
        recommendation=recommendation,
        memories=recalled,
    )


def _git_body(result: ReviewResult) -> str:
    reporter = GitReporter.__new__(GitReporter)
    return reporter._build_review_body(result, [])


def test_no_memories_no_section():
    assert "memories</summary>" not in _result().to_markdown()
    assert "memories</summary>" not in _git_body(_result())


def test_single_memory_nested_structure():
    """Single memory still gets nested <details>: outer > recall > sections."""
    md = _result(("review", [("Around this work", "fact A"), ("Decisions", "fact B")])).to_markdown()
    # Should have: outer memories + 1 recall + 2 section details = 4 <details>
    assert md.count("<details>") == 4
    assert md.count("</details>") == 4
    # No ### task heading - task is in <summary>
    assert "### review" not in md
    assert "<summary>review</summary>" in md
    # Sections are in summaries
    assert "<summary>Around this work</summary>" in md
    assert "<summary>Decisions</summary>" in md


def test_memories_before_summary():
    """Memories section must appear before ## Summary in to_markdown output."""
    md = _result(("review", [("Around this work", "fact A")])).to_markdown()
    assert md.index("<summary>memories</summary>") < md.index("## Summary")


def test_git_body_memories_before_summary():
    """Memories section must appear before 📋 Summary in git body."""
    body = _git_body(_result(("review", [("Around this work", "fact A")])))
    assert "<summary>🧠 memories</summary>" in body
    assert body.index("<summary>🧠 memories</summary>") < body.index("<summary>📋 Summary</summary>")


def test_multiple_memories_ordered():
    """Multiple recalls appear in order with nested sections."""
    md = _result(
        ("scope", [("Around this work", "scope context")]),
        ("review", [("Decisions", "review decisions")]),
    ).to_markdown()
    # Should have: outer + 2 recalls + 2 sections = 5 <details>
    assert md.count("<details>") == 5
    assert md.count("</details>") == 5
    # Recalls are ordered
    assert md.index("<summary>scope</summary>") < md.index("<summary>review</summary>")
    # Outer section closes last
    assert md.rstrip().endswith("</details>")


def test_section_with_subheadings_stays_in_one_details():
    """A section body containing '## ' must render as exactly one <details> block."""
    body_with_subheadings = "## Service Structure\nContent\n## Another subsection\nMore content"
    md = _result(
        ("review", [("Around this work", body_with_subheadings), ("Decisions", "fact")])
    ).to_markdown()
    # outer (1) + recall (1) + 2 sections (2) = 4 <details>
    assert md.count("<details>") == 4
    assert md.count("</details>") == 4


def test_empty_sections_not_rendered():
    """Empty memories list produces no memories section."""
    result = ReviewResult(
        pr_number=1,
        pr_title="t",
        pr_url="https://github.com/o/r/pull/1",
        repo="o/r",
        model_used="m",
        recommendation="approve",
        memories=[RecalledMemory(task="review", sections=[])],
    )
    assert result.memories_markdown_lines() == []


def test_git_body_truncates_to_budget():
    """Large memories are truncated to fit within MAX_BODY_CHARS."""
    big = "x" * 50_000
    body = _git_body(_result(("scope", [("Around this work", big)]), ("review", [("Decisions", big)])))
    assert len(body) <= GitReporter.MAX_BODY_CHARS
    assert TRUNCATED_MARKER in body
    assert body.count("<details>") == body.count("</details>")


def test_budget_too_small_drops_section():
    """When budget is too small, memories section is dropped entirely."""
    result = _result(("review", [("Around this work", "abc")]))
    assert result.memories_markdown_lines(max_chars=10) == []


def test_balanced_tags_when_truncated():
    """When truncating, all <details> tags remain balanced."""
    big = "x" * 100_000
    result = _result(("review", [("Around this work", big), ("Decisions", big)]))
    lines = result.memories_markdown_lines(max_chars=5000)
    text = "\n".join(lines)
    assert text.count("<details>") == text.count("</details>")
    assert TRUNCATED_MARKER in text
