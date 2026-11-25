"""Coverage of Prefrontal memory by the root RLM: sections, SUBMIT gate, prompt, wiring."""

import logging
import re
from pathlib import Path

import dspy  # type: ignore[import-untyped]
import pytest
from dspy.primitives.code_interpreter import FinalOutput  # type: ignore[import-untyped]
from dspy.primitives.repl_types import REPLEntry, REPLHistory  # type: ignore[import-untyped]
from dspy.utils.dummies import DummyLM  # type: ignore[import-untyped]

from codespy.agents.context_safe import COVERAGE_REJECTION_PREFIX, CodespyRLM
from codespy.agents.memory.prefrontal import prefrontal as pf_mod
from codespy.agents.memory.prefrontal.format import (
    FacetResult,
    format_context,
    format_context_sections,
    is_section_title,
)
from codespy.agents.memory.prefrontal.prefrontal import (
    MEMORY_BLOCK_HEADER,
    MEMORY_READ_BLOCK_HEADER,
    PREFRONTAL_FIELD,
    with_prefrontal_memory,
)
from codespy.agents.memory.prefrontal.query import (
    FACET_CONTEXT,
    FACET_DECISIONS,
    FACET_PATTERNS,
    FACET_SEEN_BEFORE,
)
from codespy.agents.review.code_review.agent import (
    CodeReviewSignature,
    coverage_check,
    with_output_cap,
)
from codespy.agents.rlm_tools import (
    combine_checks,
    log_rlm_outcome,
    memory_check,
    memory_sections,
    memory_slices,
    unseen_memory_sections,
)
from codespy.tools.git.models import ChangedFile, FileStatus


def _bullets(prefix: str, n: int) -> str:
    return "\n".join(
        f"- [observation] {prefix} fact number {i} about module_{prefix}" for i in range(n)
    )


DECISIONS = _bullets("decision", 10)
PATTERNS = _bullets("pattern", 10)
# Reflect answers and briefings carry their own markdown headings.
CONTEXT = (
    "# Review: payments\n### Summary\nThe payments service retries twice.\n## Details\n"
    + _bullets("context", 5)
)
MEMORY = (
    f"## Briefing: svc\nUse structured logging.\n## Notes\nNever log secrets.\n\n"
    f"## Around this work\n{CONTEXT}\n\n"
    f"## Decisions\n{DECISIONS}\n\n"
    f"## Recurring patterns\n{PATTERNS}"
)
TITLES = ["Briefing: svc", "Around this work", "Decisions", "Recurring patterns"]


def _history(*outputs: str, code: str = "print(x)") -> REPLHistory:
    h = REPLHistory(max_output_chars=10_000)
    for out in outputs:
        h = h.append(reasoning="r", code=code, output=out)
    return h


def _section(title: str, text: str = MEMORY) -> tuple[int, int]:
    for t, start, end, _ in memory_sections(text):
        if t == title:
            return start, end
    raise KeyError(title)


# --------------------------------------------------------------------------
# memory_sections
# --------------------------------------------------------------------------


class TestMemorySections:
    def test_titles_and_offsets(self):
        sections = memory_sections(MEMORY)
        assert [s[0] for s in sections] == TITLES
        for title, start, end, body in sections:
            chunk = MEMORY[start:end]
            assert chunk.startswith(f"## {title}\n")
            assert chunk.endswith(body)
            assert chunk == chunk.rstrip()
        assert (
            MEMORY[_section("Decisions")[0] : _section("Decisions")[1]]
            == f"## Decisions\n{DECISIONS}"
        )
        assert sections[-1][2] == len(MEMORY)

    def test_no_header_is_one_section(self):
        assert memory_sections("just some text\nmore") == [
            ("memory", 0, len("just some text\nmore"), "just some text\nmore")
        ]

    def test_unknown_header_only_is_one_section(self):
        text = "## Random\nbody"
        assert [s[0] for s in memory_sections(text)] == ["memory"]

    @pytest.mark.parametrize("text", ["", None, "  \n\n "])
    def test_empty(self, text):
        assert memory_sections(text) == []

    def test_nested_headings_stay_in_section(self):
        sections = {t: body for t, _, _, body in memory_sections(MEMORY)}
        assert "## Notes" in sections["Briefing: svc"]
        assert "# Review: payments" in sections["Around this work"]
        assert "### Summary" in sections["Around this work"]
        assert "## Details" in sections["Around this work"]

    def test_known_titles(self):
        assert is_section_title("Decisions")
        assert is_section_title("Briefing: x")
        assert is_section_title("Other repositories in acme — verify before relying on it")
        assert not is_section_title("Details")

    def test_round_trip_with_format_context(self):
        def facts(p):
            return [
                {"id": f"{p}{i}", "text": f"{p} fact {i}", "fact_type": "observation"}
                for i in range(3)
            ]

        results = {
            FACET_CONTEXT: FacetResult(FACET_CONTEXT, text="# Review: x\n## Sub\n### Summary\nok"),
            FACET_SEEN_BEFORE: FacetResult(FACET_SEEN_BEFORE, facts=facts("seen")),
            FACET_DECISIONS: FacetResult(FACET_DECISIONS, facts=facts("dec")),
            FACET_PATTERNS: FacetResult(FACET_PATTERNS, facts=facts("pat")),
        }
        briefings = [{"name": "svc", "content": "# Conventions\n## Logging\nUse slog."}]
        remote = [
            {"id": "r1", "text": "remote fact", "tags": ["repo:acme/other"], "fact_type": "world"}
        ]
        args = (briefings, results, remote, "acme/repo")
        expected = [t for t, _ in format_context_sections(*args)]
        assert len(expected) == 6
        assert [t for t, *_ in memory_sections(format_context(*args))] == expected


# --------------------------------------------------------------------------
# unseen_memory_sections
# --------------------------------------------------------------------------


class TestUnseenMemorySections:
    def test_full_print_all_seen(self):
        assert unseen_memory_sections(MEMORY, _history(MEMORY), 10_000) == []

    def test_nothing_printed_all_unseen(self):
        assert [t for t, *_ in unseen_memory_sections(MEMORY, _history(), 10_000)] == TITLES

    def test_cut_by_output_cap_stays_unseen(self):
        filler = "x" * 6_000
        output = filler + "\n" + DECISIONS + "\n" + filler
        assert "characters omitted" in REPLEntry.format_output(output, 10_000)
        unseen = [t for t, *_ in unseen_memory_sections(MEMORY, _history(output), 10_000)]
        assert "Decisions" in unseen

    def test_slice_marks_only_its_sections(self):
        a, b = _section("Decisions")
        unseen = [t for t, *_ in unseen_memory_sections(MEMORY, _history(MEMORY[a:b]), 10_000)]
        assert unseen == ["Briefing: svc", "Around this work", "Recurring patterns"]

    def test_lines_pasted_in_code_are_seen(self):
        unseen = unseen_memory_sections(MEMORY, _history("ok", code=PATTERNS), 10_000)
        assert "Recurring patterns" not in [t for t, *_ in unseen]

    def test_offsets_returned(self):
        unseen = unseen_memory_sections(MEMORY, _history(), 10_000)
        assert unseen[2] == ("Decisions", *_section("Decisions"))

    def test_empty_memory(self):
        assert unseen_memory_sections("", _history(), 10_000) == []


# --------------------------------------------------------------------------
# memory_check / memory_slices / combine_checks
# --------------------------------------------------------------------------


_SLICE = re.compile(r"prefrontal_memory\[(\d+):(\d+)\]")


class TestMemoryCheck:
    def test_none_for_empty_memory(self):
        assert memory_check("", 10_000) is None
        assert memory_check(None, 10_000) is None

    def test_message_lists_titles_and_valid_slices(self):
        check = memory_check(MEMORY, 10_000)
        message = check(_history())
        assert message.startswith(f"{COVERAGE_REJECTION_PREFIX} Memory not read yet: ")
        for title in TITLES:
            assert title in message
        assert "10,000-char cap" in message
        slices = [(int(a), int(b)) for a, b in _SLICE.findall(message)]
        # Adjacent unseen sections are merged into one slice.
        assert slices == [(0, len(MEMORY))]
        # Printing the slices clears the gate.
        assert check(_history(*(MEMORY[a:b] for a, b in slices))) is None

    def test_slices_split_under_cap(self):
        cap = 1_000
        check = memory_check(MEMORY, cap)
        message = check(_history())
        slices = [(int(a), int(b)) for a, b in _SLICE.findall(message)]
        assert len(slices) > 1
        assert all(0 < b - a < cap for a, b in slices)
        assert slices[0][0] == 0 and slices[-1][1] == len(MEMORY)
        assert all(slices[i][1] == slices[i + 1][0] for i in range(len(slices) - 1))
        outputs = [MEMORY[a:b] + "\n" for a, b in slices]
        assert all("characters omitted" not in REPLEntry.format_output(o, cap) for o in outputs)
        assert check(_history(*outputs)) is None

    def test_non_adjacent_ranges_not_merged(self):
        a, b = _section("Around this work")
        history = _history(MEMORY[a:b])
        message = memory_check(MEMORY, 10_000)(history)
        slices = [(int(x), int(y)) for x, y in _SLICE.findall(message)]
        assert len(slices) == 2
        assert slices[0][1] <= a and slices[1][0] >= b

    def test_none_once_all_seen(self):
        assert memory_check(MEMORY, 10_000)(_history(MEMORY)) is None

    def test_hard_split_without_newline(self):
        text = "y" * 25
        assert memory_slices(text, [(0, 25)], 11) == [(0, 10), (10, 20), (20, 25)]


def _files(**patches):
    return [
        ChangedFile(filename=f"{n}.py", status=FileStatus.MODIFIED, patch=p)
        for n, p in patches.items()
    ]


PATCH = "@@ -1,5 +1,5 @@\n" + "\n".join(f"+    value_{i} = compute({i})" for i in range(5))


class TestCombineChecks:
    def test_joins_messages(self):
        check = combine_checks(
            coverage_check(_files(a=PATCH), 10_000), memory_check(MEMORY, 10_000)
        )
        message = check(_history())
        assert message.startswith(COVERAGE_REJECTION_PREFIX)
        assert message.count(COVERAGE_REJECTION_PREFIX) == 1
        assert "a.py" in message and "Memory not read yet" in message

    def test_one_message_when_one_fails(self):
        check = combine_checks(
            coverage_check(_files(a=PATCH), 10_000), memory_check(MEMORY, 10_000)
        )
        message = check(_history(PATCH))
        assert "a.py" not in message and "Memory not read yet" in message

    def test_none_when_all_pass(self):
        check = combine_checks(
            coverage_check(_files(a=PATCH), 10_000), memory_check(MEMORY, 10_000)
        )
        assert check(_history(PATCH, MEMORY)) is None

    def test_skips_none(self):
        only = memory_check(MEMORY, 10_000)
        assert combine_checks(None, only, None) is only
        assert combine_checks(None, None) is None
        assert combine_checks() is None
        check = combine_checks(coverage_check(_files(a=PATCH), 10_000), memory_check("", 10_000))
        assert check(_history(PATCH)) is None


# --------------------------------------------------------------------------
# CodespyRLM with memory_check
# --------------------------------------------------------------------------


class _Sig(dspy.Signature):
    """Review."""

    changes: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ScriptedInterpreter:
    def __init__(self, script):
        self._tools: dict = {}
        self._script = script

    @property
    def tools(self):
        return self._tools

    def start(self):
        pass

    def execute(self, code, variables=None):
        return self._script.get(code.strip())

    def shutdown(self):
        pass


SCRIPT = {
    "print(prefrontal_memory)": MEMORY,
    "print(x)": "nothing",
    "SUBMIT(answer='done')": FinalOutput({"answer": "done"}),
}


def _run(rlm, codes):
    lm = DummyLM([{"reasoning": "r", "code": c} for c in codes])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        return rlm(changes="x", interpreter_factory=lambda: ScriptedInterpreter(SCRIPT))


class TestMemoryGate:
    def test_early_submit_rejected_then_accepted(self):
        rlm = CodespyRLM(_Sig, max_iters=5, submit_check=memory_check(MEMORY, 10_000))
        pred = _run(
            rlm, ["SUBMIT(answer='done')", "print(prefrontal_memory)", "SUBMIT(answer='done')"]
        )
        assert pred.answer == "done"
        assert rlm.coverage_rejections == 1
        first = pred.trajectory[0]["output"]
        assert first.startswith(COVERAGE_REJECTION_PREFIX) and "Memory not read yet" in first

    def test_accepted_when_budget_spent(self, caplog):
        rlm = CodespyRLM(_Sig, max_iters=2, submit_check=memory_check(MEMORY, 10_000))
        with caplog.at_level(logging.WARNING, logger="codespy.agents.context_safe"):
            pred = _run(rlm, ["print(x)", "SUBMIT(answer='done')"])
        assert pred.answer == "done"
        assert rlm.coverage_rejections == 0
        assert "budget spent" in caplog.text and "Memory not read yet" in caplog.text


# --------------------------------------------------------------------------
# log_rlm_outcome(memory=...)
# --------------------------------------------------------------------------


class TestMemoryTelemetry:
    def _pred(self, *outputs):
        return dspy.Prediction(
            answer="x",
            final_reasoning="done",
            trajectory=[{"reasoning": "r", "code": "c", "output": o} for o in outputs],
        )

    def test_logs_memory_seen_and_unseen_titles(self, caplog):
        a, b = _section("Decisions")
        with caplog.at_level(logging.INFO, logger="codespy.agents.rlm_tools"):
            log_rlm_outcome(
                self._pred(MEMORY[a:b]), "doc", "svc", max_output_chars=10_000, memory=MEMORY
            )
        assert "memory_seen=1/4 sections" in caplog.text
        assert "(unseen: Briefing: svc, Around this work, Recurring patterns)" in caplog.text

    def test_all_seen(self, caplog):
        with caplog.at_level(logging.INFO, logger="codespy.agents.rlm_tools"):
            log_rlm_outcome(self._pred(MEMORY), "doc", "svc", memory=MEMORY)
        assert "memory_seen=4/4 sections" in caplog.text
        assert "unseen:" not in caplog.text

    def test_empty_trajectory(self, caplog):
        pred = dspy.Prediction(answer="x", final_reasoning="done", trajectory=[])
        with caplog.at_level(logging.INFO, logger="codespy.agents.rlm_tools"):
            log_rlm_outcome(pred, "summary", "repo", memory=MEMORY)
        assert "iterations=0/?" in caplog.text
        assert "memory_seen=0/4 sections" in caplog.text

    def test_unchanged_without_memory(self, caplog):
        with caplog.at_level(logging.INFO, logger="codespy.agents.rlm_tools"):
            log_rlm_outcome(self._pred("x"), "doc", "svc", max_output_chars=10_000)
            log_rlm_outcome(self._pred("x"), "doc", "svc", max_output_chars=10_000, memory="")
        assert "memory_seen" not in caplog.text
        lines = [r.getMessage() for r in caplog.records]
        assert lines[0] == lines[1]
        assert lines[0].endswith("unseen_files=0")

    def test_no_line_without_any_coverage_arg(self, caplog):
        with caplog.at_level(logging.INFO, logger="codespy.agents.rlm_tools"):
            log_rlm_outcome(self._pred("x"), "doc", "svc", memory=None)
        assert "iterations=" not in caplog.text


# --------------------------------------------------------------------------
# with_prefrontal_memory
# --------------------------------------------------------------------------


class _PlainSig(dspy.Signature):
    """Do the task."""

    a: str = dspy.InputField()
    out: str = dspy.OutputField()


class TestWithPrefrontalMemory:
    def test_base_block_present(self):
        sig = with_prefrontal_memory(_PlainSig)
        assert PREFRONTAL_FIELD in sig.input_fields
        assert list(sig.input_fields)[0] == PREFRONTAL_FIELD
        assert sig.instructions.startswith("Do the task.")
        assert MEMORY_BLOCK_HEADER in sig.instructions
        assert "trust the code" in sig.instructions
        assert MEMORY_READ_BLOCK_HEADER not in sig.instructions

    def test_gated_block_only_with_require_read(self):
        sig = with_prefrontal_memory(_PlainSig, require_read=True)
        assert MEMORY_BLOCK_HEADER in sig.instructions
        assert MEMORY_READ_BLOCK_HEADER in sig.instructions
        assert "sent back" in sig.instructions

    def test_idempotent(self):
        once = with_prefrontal_memory(_PlainSig, require_read=True)
        twice = with_prefrontal_memory(once, require_read=True)
        assert twice.instructions.count(MEMORY_BLOCK_HEADER) == 1
        assert twice.instructions.count(MEMORY_READ_BLOCK_HEADER) == 1
        assert list(twice.input_fields).count(PREFRONTAL_FIELD) == 1
        base_then_gated = with_prefrontal_memory(
            with_prefrontal_memory(_PlainSig), require_read=True
        )
        assert base_then_gated.instructions.count(MEMORY_BLOCK_HEADER) == 1
        assert MEMORY_READ_BLOCK_HEADER in base_then_gated.instructions

    def test_cache_per_flag(self):
        plain = with_prefrontal_memory(_PlainSig)
        gated = with_prefrontal_memory(_PlainSig, require_read=True)
        assert plain is not gated
        assert with_prefrontal_memory(_PlainSig) is plain
        assert with_prefrontal_memory(_PlainSig, require_read=True) is gated
        assert pf_mod._SIGNATURES[(_PlainSig, False)] is plain
        assert pf_mod._SIGNATURES[(_PlainSig, True)] is gated

    def test_blocks_have_no_braces(self):
        for block in (pf_mod.MEMORY_BLOCK, pf_mod.MEMORY_READ_BLOCK):
            assert "{" not in block and "}" not in block

    def test_code_review_cap_still_filled(self):
        sig = with_output_cap(
            with_prefrontal_memory(CodeReviewSignature, require_read=True), 40_000
        )
        assert "{max_output_chars}" not in sig.instructions
        assert "40,000" in sig.instructions
        assert MEMORY_READ_BLOCK_HEADER in sig.instructions


# --------------------------------------------------------------------------
# Agent wiring
# --------------------------------------------------------------------------

_REVIEW = Path(__file__).resolve().parents[1] / "src" / "codespy" / "agents" / "review"
GATED = ["code_review", "doc", "seo", "supply_chain"]
MEASURED = ["summary", "audit", "plan", "implement", "scope"]


@pytest.mark.parametrize("name", GATED)
def test_gated_agents_wire_memory_gate(name):
    src = (_REVIEW / name / "agent.py").read_text()
    assert "gated_memory_signature(" in src
    assert "self._settings.memory.prefrontal.in_prompt" in src
    assert "memory_check(memory, max_output_chars)" in src
    assert f'get_max_output_chars("{name}")' in src
    assert "max_output_chars=max_output_chars," in src
    assert "memory=memory," in src


@pytest.mark.parametrize("name", MEASURED)
def test_measured_agents_not_gated(name):
    src = (_REVIEW / name / "agent.py").read_text()
    assert "require_read=True" not in src
    assert "memory_check" not in src
    call = re.search(r"log_rlm_outcome\(.*?\n\s*\)", src, re.S)
    assert call and "memory=" in call.group(0)


class TestLastStepsLog:
    def test_logged_on_extract(self, caplog):
        pred = dspy.Prediction(
            answer="x",
            final_reasoning="Extract forced final output",
            trajectory=[
                {"reasoning": "r", "code": f"print({i})", "output": f"out {i}"} for i in range(5)
            ],
        )
        with caplog.at_level(logging.INFO, logger="codespy.agents.rlm_tools"):
            log_rlm_outcome(pred, "summary", "repo")
        text = caplog.text
        assert "last steps before extract" in text
        assert "step 3: code='print(2)'" in text and "step 5: code='print(4)'" in text
        assert "print(1)" not in text

    def test_not_logged_on_submit(self, caplog):
        pred = dspy.Prediction(
            answer="x", final_reasoning="done",
            trajectory=[{"reasoning": "r", "code": "c", "output": "o"}],
        )
        with caplog.at_level(logging.INFO, logger="codespy.agents.rlm_tools"):
            log_rlm_outcome(pred, "summary", "repo")
        assert "last steps" not in caplog.text


class TestMemoryInPrompt:
    MEM = "## Decisions\n- keep {braces} and the {max_output_chars} placeholder literal"

    def test_text_in_instructions_no_field(self):
        sig = pf_mod.with_memory_in_prompt(_PlainSig, self.MEM)
        assert PREFRONTAL_FIELD not in sig.input_fields
        assert sig.instructions.startswith("Do the task.")
        assert MEMORY_BLOCK_HEADER in sig.instructions
        assert MEMORY_READ_BLOCK_HEADER not in sig.instructions
        body = sig.instructions.split(pf_mod.MEMORY_PROMPT_BEGIN, 1)[1]
        assert self.MEM in body and body.rstrip().endswith(pf_mod.MEMORY_PROMPT_END)

    def test_gated_in_prompt_returns_no_repl_memory(self):
        sig, repl = pf_mod.gated_memory_signature(_PlainSig, MEMORY, True)
        assert repl == ""
        assert "Recurring patterns" in sig.instructions
        assert memory_check(repl, 10_000) is None

    def test_gated_variable_mode_unchanged(self):
        sig, repl = pf_mod.gated_memory_signature(_PlainSig, MEMORY, False)
        assert repl == MEMORY
        assert sig is with_prefrontal_memory(_PlainSig, require_read=True)

    def test_no_memory(self):
        assert pf_mod.gated_memory_signature(_PlainSig, "", True) == (_PlainSig, "")

    def test_code_review_cap_filled_once(self):
        sig = with_output_cap(
            pf_mod.with_memory_in_prompt(CodeReviewSignature, "## Decisions\n- x"), 40_000
        )
        assert "{max_output_chars}" not in sig.instructions
        assert "40,000" in sig.instructions
