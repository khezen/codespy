"""Every review.pr_context setting is exposed and documented with its default."""

from pathlib import Path

import pytest
import yaml

from codespy.config_dspy import PRContextConfig

ROOT = Path(__file__).resolve().parent.parent
FIELDS = list(PRContextConfig.model_fields.items())


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


@pytest.mark.parametrize("name,field", FIELDS)
def test_codespy_yaml_default_matches(name, field):
    cfg = yaml.safe_load(_read("codespy.yaml"))
    assert cfg["review"]["pr_context"][name] == field.default
    assert f"REVIEW_PR_CONTEXT_{name.upper()}" in _read("codespy.yaml")


@pytest.mark.parametrize("name,field", FIELDS)
def test_env_example(name, field):
    default = str(field.default).lower() if isinstance(field.default, bool) else field.default
    assert f"# REVIEW_PR_CONTEXT_{name.upper()}={default}" in _read(".env.example")


@pytest.mark.parametrize("name,field", FIELDS)
def test_action_input_env_and_docker_arg(name, field):
    action = yaml.safe_load(_read("action.yml"))
    inp = "pr-context-" + name.replace("_", "-")
    env = f"REVIEW_PR_CONTEXT_{name.upper()}"
    assert inp in action["inputs"]
    default = str(field.default).lower() if isinstance(field.default, bool) else str(field.default)
    assert f"Default from codespy.yaml: {default}" in action["inputs"][inp]["description"]
    text = _read("action.yml")
    assert f"{env}: ${{{{ inputs.{inp} }}}}" in text
    assert f'-e {env}"' in text


@pytest.mark.parametrize("name,field", FIELDS)
def test_documented(name, field):
    default = str(field.default).lower() if isinstance(field.default, bool) else field.default
    assert f"`review.pr_context.{name}` | `REVIEW_PR_CONTEXT_{name.upper()}` | `{default}`" in _read(
        "docs/configuration.md"
    )
    inp = "pr-context-" + name.replace("_", "-")
    assert inp in _read(".github/workflows/codespy-review.yml.example")
    assert inp in _read("docs/usage.md")


def test_action_command_input():
    action = yaml.safe_load(_read("action.yml"))
    assert action["inputs"]["command"]["default"] == "review"
