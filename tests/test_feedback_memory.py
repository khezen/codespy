"""Feedback memory: one episode per scope, skip on unchanged hash, no-op when disabled."""

from unittest.mock import MagicMock, patch

from codespy.agents.review.feedback import record_feedback
from codespy.agents.review.models import IssueFeedback, PRContext
from codespy.agents.review.scope.models import ScopeResult, ScopeType
from codespy.config_dspy import PRContextConfig

FB = [
    IssueFeedback(fingerprint="1", filename="svc/a/x.py", verdict="rejected", title="A"),
    IssueFeedback(fingerprint="2", filename="svc/b/y.py", verdict="accepted", title="B"),
    IssueFeedback(fingerprint="3", filename="svc/a/z.py", verdict="discussed", title="C",
                  replies=["@bob: intended"]),
]


def _ctx(feedback=FB):
    return PRContext(repo_slug="github.com/o/r", pr_number=3, pr_title="t",
                     pr_url="https://github.com/o/r/pull/3", summary="s",
                     review_feedback=list(feedback))


def _scope(subroot):
    return ScopeResult(repo="o/r", subroot=subroot, scope_type=ScopeType.SERVICE,
                       has_changes=True, changed_files=[], reason="t")


def _settings(enabled=True):
    s = MagicMock()
    s.memory.enabled = enabled
    s.review.pr_context = PRContextConfig()
    return s


def _run(store, settings=None, scopes=None, feedback=FB):
    hippos = []

    class FakeHippo:
        def __init__(self, **kw):
            self.kw = kw
            self.observed = []
            self.episode = None
            hippos.append(self)

        def observe(self, text):
            self.observed.append(text)

        def end_episode(self, store, artifacts=None):
            self.artifacts = artifacts

    submitted = []
    with (
        patch("codespy.agents.memory.hippocampus.Hippocampus", FakeHippo),
        patch("codespy.agents.memory.hippocampus.episode.submit_episode_save",
              side_effect=lambda fn, name="": submitted.append((fn, name))),
        patch("codespy.config_memory.get_episode_store", return_value=store),
        patch("codespy.config_memory.get_cerebral", return_value=None),
    ):
        n = record_feedback(settings or _settings(), _ctx(feedback),
                            scopes if scopes is not None else [_scope("svc/a"), _scope("svc/b")],
                            "run1")
        for fn, _ in submitted:
            fn()
    return n, hippos, submitted


def test_one_episode_per_scope_with_feedback():
    store = MagicMock()
    store.latest_artifact.return_value = None
    store.load_context.return_value = None
    n, hippos, submitted = _run(store)
    assert n == 2 and len(submitted) == 2
    assert all(name == "feedback-episode-save" for _, name in submitted)
    by_task = {h.kw["question"].split(": ")[1]: h for h in hippos}
    a = by_task["svc/a"]
    assert a.kw["task_name"] == "feedback"
    assert [t.type for t in a.kw["topics"]] == ["project_scope", "pull_request"]
    assert '"A"' in a.observed[0] and '"C"' in a.observed[0] and '"B"' not in a.observed[0]
    assert set(a.artifacts) == {"feedback", "feedback_hash"}


def test_skip_when_hash_unchanged():
    store = MagicMock()
    store.load_context.return_value = None
    store.latest_artifact.return_value = None
    _, hippos, _ = _run(store)
    hashes = {h.kw["question"].split(": ")[1]: h.artifacts["feedback_hash"] for h in hippos}

    store2 = MagicMock()
    store2.latest_artifact.side_effect = lambda task, topic_ids, name: (
        hashes["svc/a"] if topic_ids[0].endswith("svc/a") else None
    )
    n, hippos2, _ = _run(store2)
    assert n == 1
    assert hippos2[0].kw["question"].split(": ")[1] == "svc/b"


def test_noop_when_memory_disabled_or_no_feedback():
    store = MagicMock()
    assert _run(store, settings=_settings(enabled=False))[0] == 0
    assert _run(store, feedback=[])[0] == 0
    assert _run(None)[0] == 0
    store.latest_artifact.assert_not_called()


def test_never_raises():
    store = MagicMock()
    store.latest_artifact.side_effect = RuntimeError("db down")
    assert _run(store)[0] == 0


def test_semantic_retain_excludes_artifacts():
    """Episodic store keeps feedback + hash; Cerebral only gets Distiller observations."""
    from codespy.agents.memory.hippocampus import Hippocampus

    store = MagicMock()
    store.latest_artifact.return_value = None
    store.load_context.return_value = None
    cerebral = MagicMock()
    saved, retained = [], []
    store.save_episode.side_effect = saved.append
    cerebral.retain_episode.side_effect = lambda ep, repo_full_name=None: retained.append(ep)

    submitted = []
    with (
        patch.object(Hippocampus, "_distill", lambda self, traj, q: None),
        patch("codespy.agents.memory.hippocampus.episode.submit_episode_save",
              side_effect=lambda fn, name="": submitted.append(fn)),
        patch("codespy.config_memory.get_episode_store", return_value=store),
        patch("codespy.config_memory.get_cerebral", return_value=cerebral),
    ):
        settings = _settings()
        settings.get_memory_budget.return_value = None
        n = record_feedback(settings, _ctx(), [_scope("svc/a")], "run1")
        for fn in submitted:
            fn()

    assert n == 1
    [ep] = saved
    assert set(ep.artifacts) == {"feedback", "feedback_hash"}
    [sem] = retained
    assert sem.artifacts == {}
    assert sem.id == ep.id and sem.run_id == ep.run_id  # same episode otherwise
    assert ep.artifacts  # original episode not mutated
