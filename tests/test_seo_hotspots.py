"""SEO hotspot pre-scan, prompt wording and agent wiring."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest

from codespy.agents.review.models import PRContext, ReviewContext, ReviewMetadata
from codespy.agents.review.scope.models import ScopeResult, ScopeType
from codespy.agents.review.seo import hotspots as hs
from codespy.agents.review.seo.agent import SEO_AREAS, SEOReviewSignature, seo_signature
from codespy.config_dspy import PRContextConfig
from codespy.tools.git.models import ChangedFile, FileStatus

BUTTON_WITH_MODAL = """\
"use client";
import { CdaStructuredTextValue } from "datocms-structured-text-utils";
import { useState } from "react";
import DefaultModal from "../Modal/DefaultModal";

export interface IButtonWithModal {
  modalTitle: string;
  modalContent: CdaStructuredTextValue;
}

const ButtonWithModal = ({ modalTitle, modalContent }: IButtonWithModal) => {
  const [showModal, setShowModal] = useState(false);
  return (
    <>
      <DefaultModal
        isOpen={showModal}
        title={modalTitle}
        content={modalContent}
      />
      <button onClick={() => setShowModal(true)}>Explain this +</button>
    </>
  );
};

export default ButtonWithModal;
"""

DEFAULT_MODAL = """\
"use client";
import { DefaultModal as PeaksDefaultModal } from "@peaks-com/ui-components";

const DefaultModal = (props) => <PeaksDefaultModal {...props} />;

export default DefaultModal;
"""

CONSUMER = """\
import ButtonWithModal from "@/components/website/ButtonWithModal/ButtonWithModal";

export const Costs = ({ block }) => (
  <ButtonWithModal modalTitle="Fund costs" modalContent={block.content} />
);
"""

BAR_GRAPH = """\
"use client";
import { useState } from "react";

const BarGraph = ({ dataSet }) => {
  const defaultDataSet = dataSet.find((set) => set.selected) || dataSet[0];
  const [selectedSet, setSelectedSet] = useState(defaultDataSet);
  return <p>{selectedSet.label}</p>;
};

export default BarGraph;
"""

NAV_DROPDOWN = """\
const NavigationDropdown = ({ navigation, subnav }) => (
  <div>
    {navigation.map((nav) => {
      const isActive = subnav?.id === nav.id;
      return (
        <div key={nav.id}>
          {isActive && hasSubNavigation && (
            <ul>{nav.links.map((l) => <li key={l.id}><a href={l.href}>{l.t}</a></li>)}</ul>
          )}
        </div>
      );
    })}
  </div>
);

export default NavigationDropdown;
"""

NO_STATE = """\
const Wrapper = ({ isOpen }) => <Modal isOpen={isOpen} />;
export default Wrapper;
"""

BTN = "src/components/website/ButtonWithModal/ButtonWithModal.tsx"
MODAL = "src/components/website/Modal/DefaultModal.tsx"
COSTS = "src/components/website/Costs/Costs.tsx"
BAR = "src/components/website/BarGraph/BarGraph.tsx"
NAV = "src/components/shared/Header/NavigationDropdown.tsx"


def _write(root: Path, files: dict[str, str]) -> list[str]:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return sorted(files)


def _site(root: Path) -> list[str]:
    return _write(root, {
        BTN: BUTTON_WITH_MODAL, MODAL: DEFAULT_MODAL, COSTS: CONSUMER,
        BAR: BAR_GRAPH, NAV: NAV_DROPDOWN,
    })


def _by(found, rule, filename):
    return [h for h in found if h.rule == rule and h.filename == filename]


def test_disclosure_prop_with_external_source_and_consumer(tmp_path):
    files = _site(tmp_path)
    found = hs.scan(tmp_path, files)
    [h] = _by(found, "disclosure-prop", BTN)
    assert h.component == "ButtonWithModal"
    assert h.tag == "DefaultModal"
    assert h.external_source == "@peaks-com/ui-components"
    assert "modalContent" in h.content_props
    assert COSTS in h.used_by
    assert h.line == BUTTON_WITH_MODAL.splitlines().index('        isOpen={showModal}') + 1


def test_selected_dataset_and_mount_on_interaction(tmp_path):
    found = hs.scan(tmp_path, _site(tmp_path))
    assert _by(found, "selected-dataset", BAR)
    assert _by(found, "mount-on-interaction", NAV)


def test_no_state_not_flagged_and_excluded_files_skipped(tmp_path):
    files = _write(tmp_path, {
        "src/Wrapper.tsx": NO_STATE,
        "src/BarGraph.stories.tsx": BAR_GRAPH,
        "src/BarGraph.test.tsx": BAR_GRAPH,
        "src/__tests__/BarGraph.tsx": BAR_GRAPH,
    })
    assert hs.scan(tmp_path, files) == []


def test_aria_expanded_is_not_a_disclosure_prop(tmp_path):
    files = _write(tmp_path, {"src/A.tsx": (
        "import { useState } from 'react';\n"
        "const A = () => { const [o] = useState(false); "
        "return <button aria-expanded={o}>x</button>; };\nexport default A;\n"
    )})
    assert hs.scan(tmp_path, files) == []


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unsupported")
def test_symlink_is_skipped(tmp_path):
    outside = tmp_path / "outside.tsx"
    outside.write_text(BAR_GRAPH)
    root = tmp_path / "site"
    (root / "src").mkdir(parents=True)
    os.symlink(outside, root / "src" / "Link.tsx")
    assert hs.scan(root, ["src/Link.tsx"]) == []


def test_pr_mode_neighbours_marked_context_only(tmp_path):
    all_files = _site(tmp_path)
    near = hs.neighbours([COSTS], all_files, tmp_path)
    assert BTN in near  # changed file imports it
    found = hs.scan(tmp_path, sorted({COSTS, *near}), all_files=all_files, changed=[COSTS])
    [h] = _by(found, "disclosure-prop", BTN)
    assert h.context_only
    assert "[context]" in hs.render(found)


def test_every_rule_appears_before_repeats(tmp_path):
    files = _site(tmp_path)
    for i in range(5):
        files += _write(tmp_path, {f"src/m/M{i}.tsx": BUTTON_WITH_MODAL.replace(
            "ButtonWithModal", f"M{i}")})
    found = hs.scan(tmp_path, files)
    first = [h.rule for h in found[:3]]
    assert set(first) == {"disclosure-prop", "selected-dataset", "mount-on-interaction"}


def test_render_caps_and_empty(tmp_path):
    assert hs.render([]) == ""
    found = hs.scan(tmp_path, _site(tmp_path))
    full = hs.render(found)
    assert full.startswith("HOTSPOTS")
    assert f"[disclosure-prop] {BTN}:" in full
    one = hs.render(found, max_items=1)
    assert one.count("\n[") == 1 and "more hotspots not shown" in one
    small = hs.render(found, max_chars=900)
    assert len(small) <= 900 + 60


def test_with_hotspots_is_cached():
    a = hs.with_hotspots(SEOReviewSignature)
    assert a is hs.with_hotspots(SEOReviewSignature)
    assert "hotspots" in a.input_fields
    assert hs.with_hotspots(a) is a


# --------------------------------------------------------------------------- prompt


def test_pr_and_path_mode_instructions():
    pr = seo_signature(False).instructions
    path = seo_signature(True).instructions
    assert "Review the diff, not the whole site" in pr
    assert "Reporting boundary" in pr and "[context]" in pr
    assert "Review the diff, not the whole site" not in path
    assert "FULL-SCOPE AUDIT" in path and "report existing issues" in path
    for text in (pr, path):
        assert "Your own sweep, ALWAYS" in text
        assert "STEP 3: HOTSPOT DETECTION" in text
        assert "Disclosure content" in text and "Selector shows one dataset" in text
        assert "Menus mounted on interaction" in text
    assert seo_signature(True) is seo_signature(True)
    assert seo_signature(False) is SEOReviewSignature


def test_path_mode_requires_area_coverage():
    pr = seo_signature(False).instructions
    path = seo_signature(True).instructions
    assert pr == SEOReviewSignature.instructions
    assert "STEP 4: COVERAGE" not in pr
    assert "STEP 4: COVERAGE" in path
    assert path.index("STEP 3:") < path.index("STEP 4: COVERAGE") < path.index(
        "GROUPING AND SEVERITY BY CONTENT"
    )
    coverage = path.split("STEP 4: COVERAGE", 1)[1].split("GROUPING", 1)[0]
    output_areas = pr.split("Area is one of:", 1)[1].split("Prefix with", 1)[0]
    for area in SEO_AREAS:
        assert area in coverage
        assert area in output_areas  # ledger areas match the reportable title Areas
    assert len(SEO_AREAS) == output_areas.count(",") + 1
    assert "Hard stop" in coverage
    assert "Think step-by-step" not in path


# --------------------------------------------------------------------------- wiring


def _settings():
    s = MagicMock()
    s.is_signature_enabled.return_value = True
    s.get_seo_scopes.return_value = []
    s.memory.enabled = False
    s.get_max_iters.return_value = 5
    s.get_max_llm_calls.return_value = 16
    s.get_rlm_threshold.return_value = 0.30
    s.get_max_concurrent_scopes.return_value = 2
    s.review.min_confidence = 0.0
    s.review.excluded_directories = ["node_modules"]
    s.review.pr_context = PRContextConfig()
    return s


def _run(tmp_path, files: dict[str, str], full_scope: bool, scan_error: bool = False):
    _write(tmp_path, files)
    (tmp_path / "package.json").write_text('{"dependencies": {"next": "15"}}')
    scope = ScopeResult(
        repo="o/r", subroot="web", scope_type=ScopeType.APPLICATION, has_changes=True,
        changed_files=[
            ChangedFile(filename=f"web/{rel}", status=FileStatus.ADDED, patch="+x")
            for rel in files
        ],
        reason="test",
    )
    captured: dict = {}

    def capture_build(sig, *a, **k):
        captured["sig"] = sig
        return MagicMock()

    async def capture_run(agent, **kwargs):
        captured["kwargs"] = kwargs
        return dspy.Prediction(issues=[])

    ctx = ReviewContext(
        pr_context=PRContext(repo_slug="github.com/o/r", pr_number=1, pr_title="t",
                             pr_url="https://github.com/o/r/pull/1", summary="s"),
        metadata=ReviewMetadata(repo_path=tmp_path, run_id="r", is_full_path=full_scope),
    )
    mod = "codespy.agents.review.seo.agent"
    patches = [
        patch(f"{mod}.get_settings", return_value=_settings()),
        patch(f"{mod}.get_cost_tracker", return_value=MagicMock()),
        patch(f"{mod}.build_rlm_agent", side_effect=capture_build),
        patch(f"{mod}.run_rlm", side_effect=capture_run),
        patch(f"{mod}.log_rlm_outcome"),
        patch(f"{mod}.get_prefrontal", return_value=None),
        patch(f"{mod}.resolve_scope_root", return_value=tmp_path),
        patch(f"{mod}.SEOReviewer._create_tools", AsyncMock(return_value=([], []))),
        patch(f"{mod}.cleanup_mcp_contexts", AsyncMock()),
    ]
    if scan_error:
        patches.append(patch(f"{mod}.scan", side_effect=RuntimeError("boom")))
    for p in patches:
        p.start()
    try:
        from codespy.agents.review.seo.agent import SEOReviewer

        SEOReviewer().forward(scopes=[scope], review_context=ctx)
    finally:
        for p in patches:
            p.stop()
    return captured


@pytest.mark.parametrize("full_scope", [True, False])
def test_hotspots_field_and_kwarg_when_found(tmp_path, full_scope):
    cap = _run(tmp_path, {BTN: BUTTON_WITH_MODAL, MODAL: DEFAULT_MODAL, BAR: BAR_GRAPH},
               full_scope)
    assert set(cap["kwargs"]) == set(cap["sig"].input_fields)
    assert "hotspots" in cap["kwargs"]
    assert "[selected-dataset]" in cap["kwargs"]["hotspots"]
    assert ("FULL-SCOPE AUDIT" in cap["sig"].instructions) is full_scope
    assert "Your own sweep, ALWAYS" in cap["sig"].instructions


def test_no_hotspots_no_field(tmp_path):
    cap = _run(tmp_path, {"src/Plain.tsx": "export default function Plain() { return <p/>; }\n"},
               True)
    assert set(cap["kwargs"]) == set(cap["sig"].input_fields)
    assert "hotspots" not in cap["kwargs"]
    assert "Your own sweep, ALWAYS" in cap["sig"].instructions


def test_scan_error_soft_fails(tmp_path):
    cap = _run(tmp_path, {BAR: BAR_GRAPH}, True, scan_error=True)
    assert "kwargs" in cap and "hotspots" not in cap["kwargs"]
