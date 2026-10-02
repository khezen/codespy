"""Tests for delta operations coercion shim.

These tests verify that the delta_ops_compat module correctly handles
JSON-encoded operations strings from models without native structured output.
"""

from __future__ import annotations

import json

# Import hindsight_api first (available in test environment with venv)
from hindsight_api.engine.reflect import delta_ops

# Now import the module under test
from codespy.agents.memory.cerebral.delta_ops_compat import (
    _decode_operation_strings,
    _normalize,
    _normalize_dict_operations,
    install_delta_ops_coercion,
)


class TestNormalizeDictOperations:
    """Tests for _normalize_dict_operations."""

    def test_operations_as_json_string(self):
        """Dict with operations as JSON string -> decoded to list."""
        op = {"op": "append_block", "section_id": "s1", "text": "hello"}
        raw = {"operations": json.dumps([op])}

        result, changed = _normalize_dict_operations(raw)

        assert changed is True
        assert result["operations"] == [op]

    def test_operations_as_nested_json_string(self):
        """Dict with operations as nested JSON object -> inner list used."""
        op = {"op": "append_block", "section_id": "s1", "text": "hello"}
        inner = {"operations": [op]}
        raw = {"operations": json.dumps(inner)}

        result, changed = _normalize_dict_operations(raw)

        assert changed is True
        assert result["operations"] == [op]

    def test_operations_as_list_of_json_strings(self):
        """Dict with operations as list of JSON strings -> decoded to dicts."""
        op1 = {"op": "append_block", "section_id": "s1", "text": "hello"}
        op2 = {"op": "replace_block", "section_id": "s1", "block_id": "b1", "text": "world"}
        raw = {"operations": [json.dumps(op1), json.dumps(op2)]}

        result, changed = _normalize_dict_operations(raw)

        assert changed is True
        assert result["operations"] == [op1, op2]

    def test_well_formed_dict_unchanged(self):
        """Well-formed dict with list operations passes through unchanged."""
        ops = [{"op": "append_block", "section_id": "s1", "text": "hello"}]
        raw = {"operations": ops}

        result, changed = _normalize_dict_operations(raw)

        assert changed is False
        assert result is raw  # Identity preserved

    def test_non_json_operations_string_logged_and_unchanged(self, caplog):
        """Non-JSON operations string logs warning and returns unchanged."""
        import logging

        raw = {"operations": "not json"}

        with caplog.at_level(
            logging.WARNING, logger="codespy.agents.memory.cerebral.delta_ops_compat"
        ):
            result, changed = _normalize_dict_operations(raw)

        assert changed is False
        assert result is raw  # Identity preserved
        assert "non-JSON string" in caplog.text

    def test_no_operations_key(self):
        """Dict without operations key passes through unchanged."""
        raw = {"other_key": "value"}

        result, changed = _normalize_dict_operations(raw)

        assert changed is False
        assert result is raw

    def test_operations_none(self):
        """Dict with operations=None passes through unchanged."""
        raw = {"operations": None}

        result, changed = _normalize_dict_operations(raw)

        assert changed is False
        assert result["operations"] is None


class TestDecodeOperationStrings:
    """Tests for _decode_operation_strings."""

    def test_decode_json_strings_in_list(self):
        """JSON strings in list are decoded to dicts."""
        op1 = {"op": "append_block", "section_id": "s1", "text": "hello"}
        op2 = {"op": "replace_block", "section_id": "s1", "block_id": "b1", "text": "world"}
        ops = [json.dumps(op1), json.dumps(op2)]

        result = _decode_operation_strings(ops)

        assert result == [op1, op2]

    def test_mixed_list_partially_decoded(self):
        """Mixed list with some JSON strings and some dicts."""
        op1 = {"op": "append_block", "section_id": "s1", "text": "hello"}
        op2 = {"op": "replace_block", "section_id": "s1", "block_id": "b1", "text": "world"}
        ops = [json.dumps(op1), op2, "not json"]

        result = _decode_operation_strings(ops)

        assert result == [op1, op2, "not json"]

    def test_no_json_strings_identity_preserved(self):
        """List without JSON strings returns same list (identity preserved)."""
        ops = [{"op": "append_block"}, {"op": "replace_block"}]

        result = _decode_operation_strings(ops)

        assert result is ops  # Identity preserved


class TestNormalize:
    """Tests for _normalize."""

    def test_string_input_with_encoded_operations(self):
        """String input with encoded operations is parsed and normalized."""
        op = {"op": "append_block", "section_id": "s1", "text": "hello"}
        # The string contains JSON where operations is a JSON-encoded string
        text = json.dumps({"operations": json.dumps([op])})

        result, changed = _normalize(text)

        assert changed is True
        assert result["operations"] == [op]

    def test_string_input_well_formed(self):
        """String input with well-formed operations is parsed and unchanged."""
        op = {"op": "append_block", "section_id": "s1", "text": "hello"}
        text = json.dumps({"operations": [op]})

        result, changed = _normalize(text)

        # changed is False because no actual decoding was needed (operations was already a list)
        assert changed is False
        assert result == {"operations": [op]}

    def test_dict_input(self):
        """Dict input is normalized."""
        op = {"op": "append_block", "section_id": "s1", "text": "hello"}
        raw = {"operations": json.dumps([op])}

        result, changed = _normalize(raw)

        assert changed is True
        assert result["operations"] == [op]

    def test_list_input_unchanged(self):
        """List input passes through unchanged."""
        ops = [{"op": "append_block"}]

        result, changed = _normalize(ops)

        assert changed is False
        assert result is ops

    def test_invalid_json_string_unchanged(self):
        """Invalid JSON string returns unchanged."""
        text = "not valid json"

        result, changed = _normalize(text)

        assert changed is False
        assert result == text


class TestInstallDeltaOpsCoercion:
    """Tests for install_delta_ops_coercion."""

    def setup_method(self):
        """Reset delta_ops module before each test."""
        # Store the original function
        self._original = delta_ops.parse_delta_operation_list
        # Ensure no coercion is installed - try to unwrap if needed
        if hasattr(self._original, "_codespy_delta_coercion"):
            # Get the original function from our wrapper
            # The wrapper stores the original in its closure
            if hasattr(self._original, "__wrapped__"):
                delta_ops.parse_delta_operation_list = self._original.__wrapped__
            else:
                # If __wrapped__ is not available, we need to get the original
                # from the wrapper's closure
                closure = getattr(self._original, "__closure__", None)
                if closure:
                    delta_ops.parse_delta_operation_list = closure[0].cell_contents

    def teardown_method(self):
        """Restore original function after each test."""
        delta_ops.parse_delta_operation_list = self._original

    def test_install_wraps_function(self):
        """install_delta_ops_coercion wraps parse_delta_operation_list."""
        # Get the current function
        original = delta_ops.parse_delta_operation_list

        # Install the coercion
        install_delta_ops_coercion()

        # Should have been wrapped
        assert delta_ops.parse_delta_operation_list is not original
        assert hasattr(delta_ops.parse_delta_operation_list, "_codespy_delta_coercion")

    def test_idempotent_install(self):
        """Multiple installs have no additional effect."""
        # Install once
        install_delta_ops_coercion()
        wrapped = delta_ops.parse_delta_operation_list

        # Install again
        install_delta_ops_coercion()
        wrapped2 = delta_ops.parse_delta_operation_list

        # Should be the same wrapper
        assert wrapped is wrapped2

    def test_wrapper_handles_encoded_operations(self):
        """The wrapper correctly handles JSON-encoded operations."""
        # Install the coercion
        install_delta_ops_coercion()

        # Test with encoded operations
        op = {"op": "append_block", "section_id": "s1", "text": "hello"}
        raw = {"operations": json.dumps([op])}

        # This would previously raise TypeError
        result = delta_ops.parse_delta_operation_list(raw)

        assert len(result.operations) == 1
        assert result.operations[0].op == "append_block"

    def test_wrapper_passes_well_formed_unchanged(self):
        """The wrapper passes well-formed payloads unchanged."""
        # Install the coercion
        install_delta_ops_coercion()

        # Test with well-formed operations
        from hindsight_api.engine.reflect.delta_ops import AppendBlockOp

        ops = [AppendBlockOp(section_id="s1", text="hello")]
        raw = {"operations": ops}

        result = delta_ops.parse_delta_operation_list(raw)

        assert len(result.operations) == 1
        assert result.operations[0].section_id == "s1"


class TestRealWorldScenarios:
    """Tests simulating real-world error scenarios."""

    def setup_method(self):
        """Store original function before test."""
        self._original = delta_ops.parse_delta_operation_list

    def teardown_method(self):
        """Restore original function after test."""
        delta_ops.parse_delta_operation_list = self._original

    def test_nemotron_double_encoding(self):
        """Simulate the Nemotron double-encoding issue that caused the original bug."""
        # Install the coercion
        install_delta_ops_coercion()

        # Simulate what Nemotron returns: operations is a JSON-encoded string
        actual_ops = [
            {"op": "append_block", "section_id": "overview", "text": "Test content"},
            {"op": "add_section", "heading": "New Section", "blocks": ["Block content"]},
        ]
        nemotron_response = {"operations": json.dumps(actual_ops)}

        # This was the original failure: TypeError: operations must be a list, got <class 'str'>
        # With the shim, it should work
        result = delta_ops.parse_delta_operation_list(nemotron_response)

        assert len(result.operations) == 2
        assert result.operations[0].op == "append_block"
        assert result.operations[1].op == "add_section"

    def test_nested_operations_wrapping(self):
        """Test handling of double-wrapped operations dict."""
        # Install the coercion
        install_delta_ops_coercion()

        # Sometimes the model wraps the whole thing in another operations key
        inner = {"operations": [{"op": "append_block", "section_id": "s1", "text": "hello"}]}
        raw = {"operations": json.dumps(inner)}

        result = delta_ops.parse_delta_operation_list(raw)

        assert len(result.operations) == 1
        assert result.operations[0].op == "append_block"

    def test_text_input_with_encoded_operations(self):
        """Test handling of text input containing encoded operations."""
        # Install the coercion
        install_delta_ops_coercion()

        # Text input that contains JSON with string-encoded operations
        ops = [{"op": "replace_block", "section_id": "s1", "block_id": "b1", "text": "new text"}]
        text = json.dumps({"operations": json.dumps(ops)})

        result = delta_ops.parse_delta_operation_list(text)

        assert len(result.operations) == 1
        assert result.operations[0].op == "replace_block"


class TestCallerDictNotMutated:
    """Tests ensuring the caller's dict is not mutated."""

    def test_caller_dict_not_mutated(self):
        """The caller's dict should never be mutated."""
        op = {"op": "append_block", "section_id": "s1", "text": "hello"}
        original_raw = {"operations": json.dumps([op]), "other_key": "value"}
        original_copy = json.loads(json.dumps(original_raw))

        result, changed = _normalize_dict_operations(original_raw)

        assert changed is True
        # Original dict should be unchanged
        assert original_raw == original_copy
        # Result should be a new dict
        assert result is not original_raw
        assert result["operations"] == [op]
        assert result["other_key"] == "value"
