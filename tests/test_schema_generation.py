#!/usr/bin/env python3
"""Test script to verify that schema_to_tool properly generates nested object properties for derivation fields."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from scienceai.data_extractor import schema_to_tool


def main():
    # Test schema with a number field that should have derivation
    test_schema = [{"type": "number", "name": "test_value", "description": "A test numeric value", "required": True}]

    # Generate the tool
    tool = schema_to_tool(test_schema)

    # Check if derivation has proper nested structure
    derivation_field = tool["function"]["parameters"]["properties"].get("test_value_derivation")

    print("Generated Tool Schema:")
    print(json.dumps(tool, indent=2))
    print("\n" + "=" * 80 + "\n")

    if derivation_field:
        print("✓ test_value_derivation field exists")

        if derivation_field.get("type") == "object":
            print("✓ derivation is type 'object'")

            if "properties" in derivation_field:
                print("✓ derivation has 'properties' defined")
                print(f"  Properties: {list(derivation_field['properties'].keys())}")

                required_props = ["operation", "operation_description", "sources", "computation"]
                for prop in required_props:
                    if prop in derivation_field["properties"]:
                        print(f"  ✓ {prop} is defined")
                    else:
                        print(f"  ✗ {prop} is MISSING")
            else:
                print("✗ derivation missing 'properties' - THIS IS THE BUG!")
        else:
            print(f"✗ derivation has wrong type: {derivation_field.get('type')}")
    else:
        print("✗ test_value_derivation field not found")


if __name__ == "__main__":
    main()
