"""Tests for Analyst class functionality."""

import inspect
from unittest.mock import MagicMock

from scienceai.analyst import Analyst


class TestExtractStructuredDataSignature:
    """Tests for extract_structured_data method signature and parameters."""

    def test_collection_message_parameter_exists(self) -> None:
        """Test that collection_message parameter exists in Analyst.extract_structured_data."""
        sig = inspect.signature(Analyst.extract_structured_data)
        assert "collection_message" in sig.parameters, "collection_message not found in Analyst.extract_structured_data"

    def test_schema_before_collection_message_order(self) -> None:
        """Test that schema parameter comes before collection_message in signature."""
        sig = inspect.signature(Analyst.extract_structured_data)
        params = list(sig.parameters.keys())

        schema_idx = params.index("schema") if "schema" in params else -1
        msg_idx = params.index("collection_message") if "collection_message" in params else -1

        assert schema_idx < msg_idx, "schema does not come before collection_message"

    def test_tool_definition_collection_message(self) -> None:
        """Test that tool definition includes collection_message with proper description."""
        mock_db = MagicMock()
        analyst = Analyst(mock_db, name="test_analyst", goal="test_goal")
        tool_def = analyst.extract_structured_data(return_tool=True)

        props = tool_def["function"]["parameters"]["properties"]
        assert "collection_message" in props, "collection_message not found in tool definition"

        desc = props["collection_message"]["description"]
        assert "context/spin" in desc.lower() or "derivation" in desc.lower() or "purpose" in desc.lower(), (
            "Tool description missing context/spin, derivation, or purpose keywords"
        )

    def test_tool_definition_property_order(self) -> None:
        """Test that tool definition properties have schema before collection_message."""
        mock_db = MagicMock()
        analyst = Analyst(mock_db, name="test", goal="test goal")
        tool_def = analyst.extract_structured_data(return_tool=True)

        props = tool_def["function"]["parameters"]["properties"]
        prop_keys = list(props.keys())

        schema_idx = prop_keys.index("schema") if "schema" in prop_keys else -1
        msg_idx = prop_keys.index("collection_message") if "collection_message" in prop_keys else -1

        assert schema_idx < msg_idx, "schema does not come before collection_message in tool definition"
