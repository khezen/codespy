"""Every review.pr_context setting is exposed and documented with its default."""

from pathlib import Path

import pytest
import yaml

from codespy.config_dspy import PRContextConfig

import json

ROOT = Path(__file__).resolve().parent.parent
FIELDS = list(PRContextConfig.model_fields.items())


def _fmt(field) -> str:
    """Default as written in docs/env: bools lowercase, lists as compact JSON."""
    d = field.get_default(call_default_factory=True)
    if isinstance(d, bool):
        return str(d).lower()
    if isinstance(d, list):
        return json.dumps(d, separators=(",", ":"))
    return str(d)


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


@pytest.mark.parametrize("name,field", FIELDS)
def test_codespy_yaml_default_matches(name, field):
    cfg = yaml.safe_load(_read("codespy.yaml"))
    assert cfg["review"]["pr_context"][name] == field.get_default(call_default_factory=True)
    assert f"REVIEW_PR_CONTEXT_{name.upper()}" in _read("codespy.yaml")


@pytest.mark.parametrize("name,field", FIELDS)
def test_env_example(name, field):
    assert f"# REVIEW_PR_CONTEXT_{name.upper()}={_fmt(field)}" in _read(".env.example")


@pytest.mark.parametrize("name,field", FIELDS)
def test_action_input_env_and_docker_arg(name, field):
    action = yaml.safe_load(_read("action.yml"))
    inp = "pr-context-" + name.replace("_", "-")
    env = f"REVIEW_PR_CONTEXT_{name.upper()}"
    assert inp in action["inputs"]
    assert f"Default from codespy.yaml: {_fmt(field)}" in action["inputs"][inp]["description"]
    text = _read("action.yml")
    assert f"{env}: ${{{{ inputs.{inp} }}}}" in text
    assert f'-e {env}"' in text


@pytest.mark.parametrize("name,field", FIELDS)
def test_documented(name, field):
    assert f"`review.pr_context.{name}` | `REVIEW_PR_CONTEXT_{name.upper()}` | `{_fmt(field)}`" in _read(
        "docs/configuration.md"
    )
    inp = "pr-context-" + name.replace("_", "-")
    assert inp in _read(".github/workflows/codespy-review.yml.example")
    assert inp in _read("docs/usage.md")


def test_action_command_input():
    action = yaml.safe_load(_read("action.yml"))
    assert action["inputs"]["command"]["default"] == "review"
