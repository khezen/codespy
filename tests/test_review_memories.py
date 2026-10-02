"""Tests for the collapsible memories section of the review output."""

from codespy.workflows.review.models import TRUNCATED_MARKER, RecalledMemory, ReviewResult
from codespy.workflows.review.reporters.git import GitReporter


def _result(*memories: tuple[str, str], recommendation: str | None = "approve") -> ReviewResult:
    return ReviewResult(
        pr_number=1,
        pr_title="t",
        pr_url="https://github.com/o/r/pull/1",
        repo="o/r",
        model_used="m",
        recommendation=recommendation,
        memories=[RecalledMemory(task=t, text=x) for t, x in memories],
    )


def _git_body(result: ReviewResult) -> str:
    reporter = GitReporter.__new__(GitReporter)
    return reporter._build_review_body(result, [])


def test_no_memories_no_section():
    assert "memories</summary>" not in _result().to_markdown()
    assert "memories</summary>" not in _git_body(_result())


def test_single_memory_uses_heading_not_nested():
    md = _result(("review", "## Context\nfact A")).to_markdown()
    section = md[md.index("<summary>memories</summary>"):]
    assert "### review" in section
    assert section.count("<details>") == 0  # only the outer one, before the summary
    assert md.count("<details>") == 1
    assert md.count("</details>") == 1


def test_multiple_memories_nested_and_ordered():
    md = _result(("scope", "scope text"), ("review", "review text")).to_markdown()
    assert md.index("## Recommendation") < md.index("<summary>memories</summary>")
    assert md.count("<details>") == 3
    assert md.count("</details>") == 3
    assert "### scope" not in md
    assert md.index("<summary>scope</summary>") < md.index("<summary>review</summary>")
    assert md.index("scope text") < md.index("<summary>review</summary>")
    # Outer section closes last
    assert md.rstrip().endswith("</details>")
    assert md.rindex("</details>") > md.index("review text")


def test_git_body_memories_last():
    body = _git_body(_result(("scope", "s"), ("review", "r")))
    assert "<summary>🧠 memories</summary>" in body
    assert body.index("<summary>💡 Recommendation</summary>") < body.index(
        "<summary>🧠 memories</summary>"
    )
    assert body.count("<details>") == body.count("</details>")


def test_git_body_truncates_to_budget():
    big = "x" * 50_000
    body = _git_body(_result(("scope", big), ("review", big)))
    assert len(body) <= GitReporter.MAX_BODY_CHARS
    assert TRUNCATED_MARKER in body
    assert body.count("<details>") == body.count("</details>")


def test_budget_too_small_drops_section():
    assert _result(("review", "abc")).memories_markdown_lines(max_chars=10) == []
