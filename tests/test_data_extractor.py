"""Tests for data extraction functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from scienceai.data_extractor import extract_data, reflect_on_data_extraction


class TestExtractDataFormatting:
    """Tests for extract_data string formatting with collection_message parameter."""

    @pytest.mark.asyncio
    async def test_extract_data_collection_message_formatting(self) -> None:
        """Test that extract_data handles collection_message parameter without formatting errors."""
        tool_schema = {"function": {"name": "test_tool"}}

        with patch("scienceai.data_extractor.get_model_for_role") as mock_get_model:
            mock_get_model.return_value = "gpt-4o"
            with patch("scienceai.data_extractor.async_client") as mock_client:
                mock_client.chat.completions.create = AsyncMock()

                try:
                    await extract_data(tool_schema, "some corpus text", collection_message="My Context Message")
                    # If we get here without a formatting error, the test passes
                    # Other errors are expected since we're mocking poorly
                except Exception as e:
                    if "Invalid format specifier" in str(e):
                        pytest.fail(f"Formatting error detected: {e}")
                    # Other errors are expected since we are mocking poorly,
                    # but formatting error happens BEFORE async calls usually


class TestReflectOnDataExtractionFormatting:
    """Tests for reflect_on_data_extraction formatting with collection_message parameter."""

    @pytest.mark.asyncio
    async def test_reflect_on_data_extraction_collection_message_formatting(self) -> None:
        """Test that reflect_on_data_extraction handles collection_message parameter without formatting errors."""
        with patch("scienceai.data_extractor.get_model_for_role") as mock_get_model:
            mock_get_model.return_value = "gpt-4o"
            with patch("scienceai.data_extractor.async_client") as mock_client:
                mock_client.chat.completions.create = AsyncMock()

                try:
                    extraction_dict = {
                        "some_field_value": "some value",
                        "some_field_source_quote": "some quote",
                    }
                    await reflect_on_data_extraction(
                        extraction_dict, "corpus text", collection_message="My Context Message"
                    )
                    # If we get here without a formatting error, the test passes
                except Exception as e:
                    if "Invalid format specifier" in str(e):
                        pytest.fail(f"Formatting error detected: {e}")
                    # Other errors are expected since we're mocking poorly
