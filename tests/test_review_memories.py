"""Tests for the collapsible memories section of the review output."""

from codespy.workflows.review.models import (
    TRUNCATED_MARKER,
    MemorySection,
    RecalledMemory,
    ReviewResult,
)
from codespy.workflows.review.reporters.git import GitReporter


def _result(
    *memories: tuple[str, list[tuple[str, str]]] | tuple[str, list[tuple[str, str]], bool],
    recommendation: str | None = "approve",
) -> ReviewResult:
    """Build a ReviewResult with memories from (task, [(section_title, section_body), ...]) tuples.

    Optional third element in tuple is nested flag (default True).
    """
    recalled = []
    for item in memories:
        task = item[0]
        sections = item[1]
        nested = item[2] if len(item) > 2 else True
        recalled.append(
            RecalledMemory(
                task=task,
                sections=[MemorySection(title=t, text=b) for t, b in sections],
                nested=nested,
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
    md = _result(("scope", [("Around this work", "fact A"), ("Decisions", "fact B")])).to_markdown()
    # Nested: outer (1) + recall (1) + 2 sections (2) = 4 <details>
    # With blockquote: each <details> has one <blockquote>
    assert md.count("<details>") == 4
    assert md.count("</details>") == 4
    # Each details+blockquote pair should have balanced tags
    assert md.count("<blockquote>") == md.count("</blockquote>")
    # No ### task heading - task is in <summary>
    assert "### scope" not in md
    assert "<summary>scope</summary>" in md
    # Sections are in summaries
    assert "<summary>Around this work</summary>" in md
    assert "<summary>Decisions</summary>" in md


def test_flat_memory_no_recall_wrapper():
    """Flat (nested=False) memory has no <summary>task</summary> wrapper."""
    md = _result(("review", [("Around this work", "fact A"), ("Decisions", "fact B")], False)).to_markdown()
    # Flat: outer (1) + 2 sections (2) = 3 <details> (no recall wrapper)
    assert md.count("<details>") == 3
    assert md.count("</details>") == 3
    # No <summary>review</summary> since it's flat
    assert "<summary>review</summary>" not in md
    # Sections are in summaries
    assert "<summary>Around this work</summary>" in md
    assert "<summary>Decisions</summary>" in md


def test_memories_before_summary():
    """Memories section must appear before ## Summary in to_markdown output."""
    md = _result(("scope", [("Around this work", "fact A")])).to_markdown()
    assert md.index("<summary>memories</summary>") < md.index("## Summary")


def test_git_body_memories_before_summary():
    """Memories section must appear before 📋 Summary in git body."""
    body = _git_body(_result(("scope", [("Around this work", "fact A")])))
    assert "<summary>🧠 memories</summary>" in body
    assert body.index("<summary>🧠 memories</summary>") < body.index("<summary>📋 Summary</summary>")


def test_multiple_memories_flat_before_scope():
    """Multiple recalls: flat first, then scope, with correct structure."""
    md = _result(
        ("review", [("Decisions", "review decisions")], False),  # flat
        ("scope", [("Around this work", "scope context")]),  # nested
    ).to_markdown()
    # Flat + nested: outer (1) + 1 section (flat) + 1 recall (scope) + 1 section (nested) = 4 <details>
    assert md.count("<details>") == 4
    assert md.count("</details>") == 4
    # Balanced blockquote tags
    assert md.count("<blockquote>") == md.count("</blockquote>")
    # Flat sections appear before scope
    assert md.index("<summary>Decisions</summary>") < md.index("<summary>scope</summary>")
    # Flat sections don't have <summary>review</summary>
    assert "<summary>review</summary>" not in md
    # Outer section closes last
    assert md.rstrip().endswith("</details>")


def test_section_with_subheadings_stays_in_one_details():
    """A section body containing '## ' must render as exactly one <details> block."""
    body_with_subheadings = "## Service Structure\nContent\n## Another subsection\nMore content"
    md = _result(
        ("scope", [("Around this work", body_with_subheadings), ("Decisions", "fact")])
    ).to_markdown()
    # nested: outer (1) + recall (1) + 2 sections (2) = 4 <details>
    assert md.count("<details>") == 4
    assert md.count("</details>") == 4
    # Balanced blockquote tags
    assert md.count("<blockquote>") == md.count("</blockquote>")


def test_empty_sections_not_rendered():
    """Empty memories list produces no memories section."""
    result = ReviewResult(
        pr_number=1,
        pr_title="t",
        pr_url="https://github.com/o/r/pull/1",
        repo="o/r",
        model_used="m",
        recommendation="approve",
        memories=[RecalledMemory(task="scope", sections=[])],
    )
    assert result.memories_markdown_lines() == []


def test_git_body_truncates_to_budget():
    """Large memories are truncated to fit within MAX_BODY_CHARS."""
    big = "x" * 50_000
    body = _git_body(_result(("scope", [("Around this work", big)]), ("review", [("Decisions", big)], False)))
    assert len(body) <= GitReporter.MAX_BODY_CHARS
    assert TRUNCATED_MARKER in body
    assert body.count("<details>") == body.count("</details>")
    assert body.count("<blockquote>") == body.count("</blockquote>")


def test_budget_too_small_drops_section():
    """When budget is too small, memories section is dropped entirely."""
    result = _result(("scope", [("Around this work", "abc")]))
    assert result.memories_markdown_lines(max_chars=10) == []


def test_balanced_tags_when_truncated():
    """When truncating, all <details> and <blockquote> tags remain balanced."""
    big = "x" * 100_000
    result = _result(("review", [("Around this work", big), ("Decisions", big)], False))
    lines = result.memories_markdown_lines(max_chars=5000)
    text = "\n".join(lines)
    assert text.count("<details>") == text.count("</details>")
    assert text.count("<blockquote>") == text.count("</blockquote>")
    assert TRUNCATED_MARKER in text


def test_flat_structure_tag_counts():
    """Flat single recall with 2 sections -> 3 details + 3 blockquotes."""
    md = _result(("review", [("Briefings", "b1"), ("Around this work", "b2")], False)).to_markdown()
    # Flat: outer + 2 sections = 3 <details>
    assert md.count("<details>") == 3
    assert md.count("</details>") == 3
    # Each details has one blockquote wrapper
    assert md.count("<blockquote>") == 3
    assert md.count("</blockquote>") == 3


def test_flat_plus_scope_structure_tag_counts():
    """Flat + scope (1 section each) -> 4 details + 4 blockquotes."""
    md = _result(
        ("review", [("Briefings", "b1")], False),
        ("scope", [("Around this work", "s1")]),
    ).to_markdown()
    # Flat + nested: outer + 1 flat section + scope recall + 1 nested section = 4 <details>
    assert md.count("<details>") == 4
    assert md.count("</details>") == 4
    # Each details has one blockquote wrapper
    assert md.count("<blockquote>") == 4
    assert md.count("</blockquote>") == 4


def test_order_flat_before_scope():
    """Flat sections must appear before <summary>scope</summary>."""
    md = _result(
        ("review", [("Briefings", "b1"), ("Decisions", "d1")], False),
        ("scope", [("Around this work", "s1")]),
    ).to_markdown()
    # Find positions of flat section and scope
    briefings_pos = md.index("<summary>Briefings</summary>")
    scope_pos = md.index("<summary>scope</summary>")
    assert briefings_pos < scope_pos
