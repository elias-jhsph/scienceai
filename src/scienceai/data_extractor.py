import json
import logging
import os
import re
from enum import Enum
from typing import Any

from num2words import num2words

from .llm import MODEL_REASONING, async_client, client, get_config, get_model_for_role
from .llm import use_tools_sync as use_tools

logger = logging.getLogger(__name__)


class ExtractionMode(Enum):
    """
    Extraction modes that control strictness and retry behavior.

    EXPLORATORY: Lenient mode for discovery. Disables convergence detection and early exit.
                 On last retry, removes problem fields and returns partial data.

    FOCUSED: Default balanced mode. Uses convergence detection and early exit.
             Full retry logic with validation.

    RIGID: Strict mode for precise extraction. All schema fields forced to required.
           Uses convergence detection and early exit.
    """

    EXPLORATORY = "exploratory"
    FOCUSED = "focused"
    RIGID = "rigid"


data_types_file = "data_types.json"
data_types_docs_file = "data_types_docs.json"

path_to_app = os.path.dirname(os.path.abspath(__file__))
data_types_file = os.path.join(path_to_app, data_types_file)
data_types_docs_file = os.path.join(path_to_app, data_types_docs_file)


def load_json_file(filename: str) -> dict[str, Any]:
    """Load and return the data from a JSON file."""
    with open(filename) as file:
        return json.load(file)


data_types: dict[str, Any] = load_json_file(data_types_file)
data_types_docs: dict[str, Any] = load_json_file(data_types_docs_file)


def validate_data_type_spec(spec, force_type=None):
    if force_type:
        if force_type not in data_types:
            return False, f"Data type '{force_type}' not found in data types."
        properties = data_types[force_type]["spec"]
        specification = spec
    else:
        if "type" not in spec:
            return False, "Key 'type' not found in specification."
        if spec["type"] not in data_types:
            return False, f"Data type '{spec['type']}' not found in data types."
        properties = data_types[spec["type"]]["spec"]
        specification = spec.copy()
        del specification["type"]
    for key, value in properties.items():
        if key not in specification:
            return False, f"Key '{key}' not found in specification."
        if value["type"] == "string":
            if not isinstance(specification[key], str):
                return False, f"Value for '{key}' is not a string."
        elif value["type"] == "boolean":
            if not isinstance(specification[key], bool):
                return False, f"Value for '{key}' is not a boolean."
        elif value["type"] == "number":
            if not isinstance(specification[key], int | float):
                return False, f"Value for '{key}' is not a number."
        elif value["type"] == "object":
            try:
                for k, v in json.loads(specification[key]):
                    if k in value["keys"]:
                        if value["keys"][k] == "string":
                            if isinstance(v, float | int):
                                return False, f"Value for '{k}' is not a string."
                        elif value["keys"][k] == "boolean":
                            if not isinstance(v, bool):
                                return False, f"Value for '{k}' is not a boolean."
                        else:
                            return False, f"Invalid data type '{value['keys'][k]}' for key '{k}' in object spec."
            except json.JSONDecodeError:  # Catch JSON decoding errors
                return False, f"Value for '{key}' is not a valid JSON object."
        elif value["type"] == "array":
            if not isinstance(specification[key], list):
                return False, f"Value for '{key}' is not an array."
            for item in specification[key]:
                if not isinstance(item, str):
                    return False, f"Item in array '{key}' is not a string."
    if len(specification) != len(properties):
        return False, "specification has extra keys."
    return True, "specification is valid."


def generate_description(data_type_key, data_type_value, all_keys):
    """Generate a description for each data type."""
    message = f"Data Type: {data_type_key}\n"
    message += f"Description: {data_type_value['spec_description']}\n"
    message += "Specifications Required:\n"
    properties = {}
    for spec_key, spec_value in data_type_value["spec"].items():
        message += f"  - {spec_key} ({spec_value['type']}): {spec_value['description']}\n"
        properties[spec_key] = {"type": spec_value["type"], "description": spec_value["description"]}
        if spec_value["type"] == "array":
            properties[spec_key]["items"] = {"type": "string"}
    tools = [
        {
            "type": "function",
            "function": {
                "strict": True,
                "name": "generate_example_data",
                "description": "creates example data for a given data type",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "additionalProperties": False,
                    "required": list(properties.keys()),
                },
            },
        }
    ]

    system_message = (
        "You are a helpful AI assistant. You have been asked to generate example data for data types. "
        "These data types are used to extract data from research papers. Each aspect of the example data "
        "must be creative and scientific. You must follow the specifications provided for each "
        "data type. Don't repeat the same example data for the same data type. Be creative! "
        "If there is a boolean value, make sure to vary it between examples. "
        "(Not every data type should be required: true!)"
        "\n\n\n Keep in mind that the full list of data types is: " + ", ".join(all_keys) + "\n\n\n"
        "Be sure to generate example data for the following data type: " + data_type_key + "\n\n"
        "Make sure your example data is not too similar to other data types."
    )

    user_message_prefix = "Generate made up example data for this data type that is creative and scientific:\n\n"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message_prefix + message},
    ]

    arguments = {
        "messages": messages,
        "tools": tools,
        "model": get_model_for_role(MODEL_REASONING),
        "reasoning_effort": "low",
        "tool_choice": {"type": "function", "function": {"name": "generate_example_data"}},
    }

    examples = []
    retry = 0
    valid_calls = []
    output_dictionary = None
    last_tool_call_id = None
    while len(examples) < 3 and retry < 3:
        while valid_calls == [] and retry < 3:
            if retry > 0:
                logger.info("Retrying...")
            chat_response = client.chat.completions.create(**arguments)
            if chat_response.choices[0].message.tool_calls:
                # Store the tool_call_id for later use
                last_tool_call_id = chat_response.choices[0].message.tool_calls[0].id
                valid_calls = use_tools(chat_response, arguments, call_functions=False)
                if valid_calls:
                    for call in valid_calls:
                        if call["name"] == "generate_example_data":
                            output_dictionary = call["parameters"]
                            valid, error_message = validate_data_type_spec(output_dictionary, force_type=data_type_key)
                            if not valid:
                                logger.warning(f"{error_message}: {output_dictionary}")
                                valid_calls = []
                            else:
                                output_dictionary["type"] = data_type_key
            else:
                logger.warning("No tool calls used")
            retry += 1
        if valid_calls and output_dictionary and last_tool_call_id:
            retry = 0
            # Build assistant message with tool_calls for the tools API format
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": last_tool_call_id,
                        "type": "function",
                        "function": {"name": valid_calls[0]["name"], "arguments": json.dumps(output_dictionary)},
                    }
                ],
            }
            # Use role='tool' with tool_call_id (not the deprecated role='function')
            tool_response = {"role": "tool", "tool_call_id": last_tool_call_id, "content": "valid"}
            messages += [assistant_message, tool_response, {"role": "user", "content": user_message_prefix + message}]
            examples.append(output_dictionary)
            valid_calls = []
            last_tool_call_id = None

    return message, examples


for data_type in data_types:
    if data_type not in data_types_docs:
        data_types_docs[data_type] = {}
        data_types_docs[data_type]["description"], data_types_docs[data_type]["examples"] = generate_description(
            data_type, data_types[data_type], list(data_types.keys())
        )
        with open(data_types_docs_file, "w") as file:
            json.dump(data_types_docs, file, indent=2)


def generate_schema(corpus, goal=None, retries=5, mode=ExtractionMode.FOCUSED):
    """
    Generate a data extraction schema for a corpus.

    Args:
        corpus: Sample text from the papers
        goal: The extraction goal
        retries: Number of generation attempts
        mode: ExtractionMode controlling schema strictness:
              - EXPLORATORY: More text_block types, all fields optional
              - FOCUSED: Balanced approach (default)
              - RIGID: All fields required
    """
    # Mode-specific guidance
    mode_guidance = ""
    if mode == ExtractionMode.EXPLORATORY:
        mode_guidance = (
            "\n\n**EXTRACTION MODE: EXPLORATORY**\n"
            "This is a discovery/exploratory extraction. Prioritize:\n"
            "- Use `text_block` liberally for descriptive fields\n"
            "- Mark ALL fields as required: false (data may be missing)\n"
            "- Prefer broader, flexible schemas over precise numeric types\n"
            "- Goal is to see what's available across papers, not precise numbers\n\n"
        )
    elif mode == ExtractionMode.RIGID:
        mode_guidance = (
            "\n\n**EXTRACTION MODE: RIGID**\n"
            "This is a strict extraction requiring precise data. Prioritize:\n"
            "- Use structured numeric types (effect_estimate, sample_statistics, etc.)\n"
            "- Mark core fields as required: true (extraction should fail if missing)\n"
            "- Only request data that MUST be present in qualifying papers\n\n"
        )

    # Construct the system message
    system_message = (
        "You are an expert data schema designer. Create a structured schema to extract specific "
        "information from research papers.\n\n"
        + mode_guidance
        + "CRITICAL: You must ONLY use these valid data types: "
        + ", ".join(data_types.keys())
        + "\n\n"
        "DO NOT use 'list', 'object_list', or any other data type not in the list above. "
        "For multiple items, use types ending in '_list' like 'number_list', 'text_block', 'date_list', etc.\n\n"
        "IMPORTANT: Data types like 'number', 'fraction', 'unit_number' automatically generate metadata fields (_value, _source_quote, _source_location, _unit). "
        "Do NOT manually specify fields ending in '_source_quote', '_source_location', '_unit', or '_value'. "
        "For example, request ONLY 'exposed_group_n' (type: number), NOT 'exposed_group_n_value', 'exposed_group_n_source_quote', etc.\n\n"
        "SEMANTIC FIELD NAMING FOR COMPARATIVE DATA:\n"
        "When the goal involves comparisons or contrasts (e.g., exposed vs unexposed, treatment vs control):\n"
        " - Create field names that encode meaning and directionality, not generic labels like 'group1' or 'group2'\n"
        " - For comparative data, ALWAYS include:\n"
        "   * A field with 'contrast' or 'comparison' in the name (type: text_block) describing what is being compared\n"
        "   * Separate label fields for each group (e.g., 'exposed_group_label', 'reference_group_label')\n"
        "   * Use descriptive names for numeric fields (e.g., 'exposed_group_n', 'reference_group_n' NOT 'group1_n', 'group2_n')\n"
        "\n"
        "Example for exposure/treatment comparison:\n"
        "✅ GOOD schema fields:\n"
        "  - name: 'exposure_contrast', type: 'text_block', description: 'Description of the exposure comparison (e.g., high dose vs low dose)'\n"
        "  - name: 'exposed_group_label', type: 'text_block', description: 'Label for the exposed/treatment group'\n"
        "  - name: 'reference_group_label', type: 'text_block', description: 'Label for the reference/control group'\n"
        "  - name: 'exposed_group_n_at_risk', type: 'number', description: 'Number at risk in exposed group'\n"
        "  - name: 'reference_group_n_at_risk', type: 'number', description: 'Number at risk in reference group'\n"
        "\n"
        "**IMPORTANT - Group Mapping Verification:**\n"
        "For comparative data, ALWAYS include a verification field to prevent group-swapping errors:\n"
        "  - name: 'group_mapping_verification', type: 'text_block', description: 'Explicit statement confirming: \"[exposed_group_label] has n=[exposed_group_n], [reference_group_label] has n=[reference_group_n]\" - this must match source quotes'\n"
        "\n"
        "❌ BAD schema fields:\n"
        "  - name: 'group1_n' (doesn't indicate what group1 represents)\n"
        "  - name: 'group2_n' (doesn't indicate what group2 represents)\n"
        "\n"
        "GENERAL principal: Field names are documentation. Make them self-explanatory so downstream analysis "
        "doesn't require external context to interpret the data.\n\n"
        "**CHOOSING BETWEEN TEXT AND NUMERIC TYPES:**\n\n"
        "✅ USE `text_block` for EXPLORATORY/DESCRIPTIVE goals:\n"
        "  - Summarizing methodology, describing outcomes, categorizing papers\n"
        "  - Understanding what's reported, identifying themes\n"
        "  - Any goal focused on 'what', 'describe', 'summarize', 'categorize'\n\n"
        "✅ USE STRUCTURED NUMERIC TYPES when goal mentions CALCULATIONS:\n"
        "  - Signal words: 'for pooling', 'for meta-analysis', 'as numeric fields', 'for calculations'\n"
        "  - `effect_estimate`/`effect_estimate_list`: effect sizes with CI and p-value\n"
        "  - `sample_statistics`/`sample_statistics_list`: n, mean, SD, median, IQR\n"
        "  - `contingency_table_2x2`: 2x2 cell counts for binary outcomes\n"
        "  - `proportion_with_ci`: percentages with confidence intervals\n"
        "  - `measurement_with_error`: value ± uncertainty\n\n"
        "**DEFAULT BEHAVIOR:**\n"
        "  - If goal is vague/exploratory → prefer text_block (safe for discovery)\n"
        "  - If goal explicitly mentions needing numbers for analysis → use numeric types\n\n"
        "**REQUIRED vs OPTIONAL FIELDS:**\n"
        "Each field needs a 'required' boolean. Choose wisely:\n\n"
        "✅ required: true - Data explicitly requested in the goal that SHOULD be in every paper:\n"
        "  - Core data points directly mentioned in the goal\n"
        "  - Essential identifiers (exposure definition, outcome definition)\n"
        "  - Primary numeric data requested for analysis\n\n"
        "✅ required: false - Supplementary data or data that may not exist in all papers:\n"
        "  - Contextual/background information not explicitly requested\n"
        "  - Statistical details that aren't always reported (CIs, p-values, subgroup data)\n"
        "  - Fields where absence is informative, not an error\n"
        "  - When unsure if data will be in every paper → use required: false\n\n"
        "Example: Goal is 'extract sample sizes and primary outcomes'\n"
        "  - sample_size: required: true (explicitly requested)\n"
        "  - primary_outcome_definition: required: true (explicitly requested)\n"
        "  - secondary_outcomes: required: false (not requested, may not exist)\n"
        "  - funding_source: required: false (helpful context, not requested)\n\n"
        "PRINCIPLE: required: false is safer - extraction won't fail if data is missing. "
        "Only use required: true for data that MUST be in every paper for the analysis to work.\n\n"
        "Design your schema to be specific and extractable from the papers. "
        "Focus on: " + (goal if goal else "extracting relevant structured data") + "\n\n"
    )

    for key, value in data_types_docs.items():
        examples = []
        for example in value["examples"]:
            examples.append(str(example))
        system_message += value["description"] + "\n" + key + " Examples:\n" + "\n".join(examples) + "\n\n"

    system_message += "\n\n\n"

    if goal:
        preamble = "Read the corpus and generate the schema with data types that will best address the following: "
        system_message += preamble + goal
        if not system_message.endswith("."):
            system_message += "."
    else:
        system_message += "Read the corpus and generate the schema with data types that will best sum up the corpus. "
    system_message += "\n\n IMPORTANT: basic bibliographic information has already been extracted from the corpus so "
    system_message += "you should not include that in the schema, so no need to ask for author names, titles, etc. "
    system_message += "Also, the schema will be used to extract data from "
    system_message += "each paper in the corpus SEPARATELY. Make sure the schema is general enough to apply to all "
    system_message += "papers in the corpus. Do not ask for ANY information that sums up the corpus as a whole."

    user_message = "Generate an analysis schema for this corpus"

    if goal:
        user_message += " that will best address the following goal - " + goal

    final_ask = (
        "Please return a JSON string representing the schema for the analysis which should "
        "be an array of objects with a 'type' key and other keys as "
        "specified in the data types. Try to limit the number of objects requested to 8 or less."
    )

    user_message += ":\n\n" + corpus + "\n\n\n\n\n" + final_ask

    # Function schema for structured output
    function_schema = {
        "type": "function",
        "function": {
            "name": "define_schema",
            "description": "Defines the schema for data extraction from research papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "required": {"type": "boolean"},
                                "type": {"type": "string", "enum": list(data_types.keys())},
                                "categories": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Required for 'categorical_value' type. List of possible categories.",
                                },
                                "field_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Required for 'named_number_set' type. List of field names to extract.",
                                },
                                "unit": {
                                    "type": "string",
                                    "description": "Required for 'unit_number' and 'unit_number_list' types.",
                                },
                            },
                            "additionalProperties": True,
                            "required": ["name", "description", "required", "type"],
                        },
                    }
                },
                "additionalProperties": False,
                "required": ["fields"],
            },
        },
    }

    messages = [{"role": "user", "content": system_message}, {"role": "user", "content": user_message}]

    retry = 0
    valid = False

    while not valid and retry < retries:
        try:
            o4_arguments = {
                "messages": messages,
                "model": get_model_for_role(MODEL_REASONING),
                "reasoning_effort": "high",
                "tools": [function_schema],
                "tool_choice": {"type": "function", "function": {"name": "define_schema"}},
            }
            response = client.chat.completions.create(**o4_arguments)

            # Enhanced logging to diagnose tool call issues
            response_content = response.choices[0].message.content if response.choices else None
            response_tool_calls = response.choices[0].message.tool_calls if response.choices else None
            logger.info(
                f"generate_schema response - has_content: {bool(response_content)}, has_tool_calls: {bool(response_tool_calls)}"
            )
            if response_content:
                logger.debug(f"Response content (truncated): {response_content[:500]}...")
            if response_tool_calls:
                logger.debug(f"Response tool_calls count: {len(response_tool_calls)}")
            else:
                logger.warning(
                    "No tool_calls in response - LLM may have returned text instead of using the define_schema tool"
                )

            valid_calls = use_tools(response, o4_arguments, call_functions=False)
            logger.debug(f"valid_calls from use_tools: {valid_calls}")

            output_dictionary = {}
            if valid_calls:
                for call in valid_calls:
                    output_dictionary = call["parameters"]
                logger.debug(
                    f"output_dictionary keys: {list(output_dictionary.keys()) if output_dictionary else 'empty'}"
                )
            else:
                logger.warning("valid_calls is empty - no parseable tool calls found")

            # Validate the schema
            fields = output_dictionary.get("fields", [])
            logger.info(f"Schema fields count: {len(fields)}")

            # validation_error = None
            # invalid_field = None
            if not fields:
                logger.warning("No fields in schema - this will cause validation to fail")
                # validation_error = "No fields were generated. Please generate at least one field."
            else:
                valid = True  # Assume valid until proven otherwise
                for field in fields:
                    valid, error_message = validate_data_type_spec(field)
                    if not valid:
                        logger.warning(f"Validation Error: {error_message} - Field: {field}")
                        # validation_error = error_message
                        # invalid_field = field
                        break

            if valid:
                logger.info(f"Schema generated successfully: {output_dictionary['fields']}")
                return output_dictionary["fields"]

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing response: {e}")

        retry += 1
        logger.info(f"Retrying... Attempt {retry}")

    return None


def _sanitize_schema_for_vertex(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively sanitize a schema to ensure Vertex AI strict validation compliance.

    Vertex AI requires that every field listed in 'required' must have a
    corresponding definition in 'properties'. This function ensures that
    requirement is met at all levels of nesting.

    Args:
        schema: The schema dictionary to sanitize

    Returns:
        The sanitized schema
    """

    def _recursive_sanitize(obj: dict[str, Any]) -> None:
        """Recursively process schema ensuring required fields are defined."""
        if not isinstance(obj, dict):
            return

        # Fix list-type values for 'type' field - Vertex AI only accepts strings
        # Convert ["number", "null"] to {"type": "number", "nullable": True} (OpenAPI 3.0.x format)
        if "type" in obj and isinstance(obj["type"], list):
            type_list = obj["type"]
            # Take the first non-null type as the primary type
            primary_type = next((t for t in type_list if t != "null"), type_list[0])
            # If "null" was in the list, mark as nullable
            is_nullable = "null" in type_list
            logger.debug(f"Sanitizing schema: Converting type {type_list} to '{primary_type}' (nullable={is_nullable})")
            obj["type"] = primary_type
            if is_nullable:
                obj["nullable"] = True

        # If this level has both 'properties' and 'required', validate them
        if "properties" in obj and "required" in obj:
            properties = obj["properties"]
            required_fields = obj["required"]

            for field_name in required_fields:
                if field_name not in properties:
                    # Inject a default definition for the missing field
                    # Use 'string' as a safe default type
                    logger.warning(
                        f"Sanitizing schema: Adding missing property definition for required field '{field_name}'"
                    )
                    properties[field_name] = {
                        "type": "string",
                        "description": f"Auto-generated definition for required field '{field_name}'",
                    }

        # Recurse into properties
        if "properties" in obj:
            for _prop_name, prop_schema in obj["properties"].items():
                if isinstance(prop_schema, dict):
                    _recursive_sanitize(prop_schema)

        # Recurse into array items
        if "items" in obj and isinstance(obj["items"], dict):
            _recursive_sanitize(obj["items"])

    # Work on a copy to avoid mutating the input
    import copy

    sanitized = copy.deepcopy(schema)

    # Sanitize the top level function parameters
    if "function" in sanitized and "parameters" in sanitized["function"]:
        _recursive_sanitize(sanitized["function"]["parameters"])

    return sanitized


def schema_to_tool(schema, mode=ExtractionMode.FOCUSED):
    """
    Convert a schema to an OpenAI function tool.

    Args:
        schema: The extraction schema
        mode: ExtractionMode controlling required field behavior
              - RIGID: All fields forced to required
              - FOCUSED/EXPLORATORY: Respect schema's required settings
    """

    import re

    def camel_to_snake(s):
        s = re.sub(r"[^a-zA-Z0-9_.-]", "_", s)
        return "".join(["_" + c.lower() if c.isupper() else c for c in s]).lstrip("_")

    selected_data_types_full = {}

    required_full = []

    # In RIGID mode, override all required fields to True
    rigid_mode = mode == ExtractionMode.RIGID

    # Fields that should never be marked as required in the extraction tool
    # These are either metadata fields OR optional statistical fields that may not be reported
    optional_fields = [
        # Metadata fields (choose quote OR derivation)
        "source_quote",
        "derivation",
        "source_location",
        "unit",
        "computation",
        # Optional statistical fields that papers may not report
        "ci_level",  # Usually 95%, often not explicitly stated
        "is_reference",  # Only relevant for reference categories
        "t_statistic",  # Often not reported
        "df",
        "df_numerator",
        "df_denominator",  # Degrees of freedom often omitted
        "se",  # Standard error often omitted when CI given
        "error_lower",
        "error_upper",  # Only for asymmetric errors
        # Sample statistics - papers report different subsets
        "mean",
        "sd",
        "median",
        "q1",
        "q3",
        "min",
        "max",
        # Survival analysis - optional fields
        "log_rank_p",
        "rate_at_timepoint",
        "rate_timepoint",
        "median_survival_ci_lower",
        "median_survival_ci_upper",
        # Contingency table - totals can be computed
        "total_exposed",
        "total_unexposed",
    ]

    for data_type_requested in schema:
        required = []
        selected_data_types = {}
        data_type_def = data_types[data_type_requested["type"]]["tool"]
        if data_type_def["mode"] == "prefix":
            required_keys = list(data_type_def.keys())
            required_keys.remove("mode")
            name = camel_to_snake(data_type_requested["name"])
            for key in required_keys:
                # In RIGID mode, all fields are required; otherwise respect schema
                is_required = rigid_mode or data_type_requested["required"]
                if is_required:
                    # Only make core value fields required, not metadata or optional stats
                    if key not in optional_fields:
                        required.append(name + "_" + key)
                new_type = data_type_def[key]["type"]
                new_description = data_type_def[key]["description"]
                for spec_key in data_types[data_type_requested["type"]]["spec"]:
                    if spec_key == "name":
                        new_description = new_description.replace("NAME", name)
                    else:
                        new_description = new_description.replace(spec_key.upper(), str(data_type_requested[spec_key]))

                # For object types (like derivation), include full nested properties
                if new_type == "object" and "properties" in data_type_def[key]:
                    selected_data_types[name + "_" + key] = {
                        "type": new_type,
                        "description": new_description,
                        "properties": data_type_def[key]["properties"],
                        "required": data_type_def[key].get("required", []),
                        "additionalProperties": False,
                    }
                else:
                    # Make optional fields nullable so LLM can use null for unreported values
                    if key in optional_fields and new_type == "number":
                        selected_data_types[name + "_" + key] = {
                            "type": "number",
                            "nullable": True,
                            "description": new_description + " Use null if not reported.",
                        }
                    else:
                        selected_data_types[name + "_" + key] = {"type": new_type, "description": new_description}
        if data_type_def["mode"] == "array":
            name = camel_to_snake(data_type_requested["name"])
            # In RIGID mode, all fields are required; otherwise respect schema
            is_required = rigid_mode or data_type_requested["required"]
            if is_required:
                required.append(name)
            new_description = data_type_def["description"]
            for spec_key in data_types[data_type_requested["type"]]["spec"]:
                if spec_key == "name":
                    new_description = new_description.replace("NAME", name)
                else:
                    new_description = new_description.replace(spec_key.upper(), str(data_type_requested[spec_key]))

            item_properties = {}
            for key in data_type_def["keys"]:
                new_type = data_type_def["keys"][key]["type"]
                new_description = data_type_def["keys"][key]["description"]
                for spec_key in data_types[data_type_requested["type"]]["spec"]:
                    if spec_key == "name":
                        new_description = new_description.replace("NAME", name)
                    else:
                        new_description = new_description.replace(spec_key.upper(), str(data_type_requested[spec_key]))

                # For object types (like derivation) in arrays, include full nested properties
                if new_type == "object" and "properties" in data_type_def["keys"][key]:
                    item_properties[key] = {
                        "type": new_type,
                        "description": new_description,
                        "properties": data_type_def["keys"][key]["properties"],
                        "required": data_type_def["keys"][key].get("required", []),
                        "additionalProperties": False,
                    }
                else:
                    # Make optional fields nullable so LLM can use null for unreported values
                    if key in optional_fields and new_type == "number":
                        item_properties[key] = {
                            "type": "number",
                            "nullable": True,
                            "description": new_description + " Use null if not reported.",
                        }
                    else:
                        item_properties[key] = {"type": new_type, "description": new_description}

            selected_data_types[name] = {
                "type": "array",
                "description": new_description,
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "additionalProperties": False,
                    "required": list(item_properties.keys()),
                },
            }
        required_full = required_full + required
        selected_data_types_full = {**selected_data_types_full, **selected_data_types}

    # Add optional discrepancy notes field for documenting source conflicts
    selected_data_types_full["data_discrepancy_notes"] = {
        "type": "string",
        "description": "OPTIONAL: Document any inconsistencies or conflicts found in the source material "
        "(e.g., 'Abstract reports n=150 but Table 2 shows n=142 for the same group', "
        "'CI reported as 1.25-0.78 which appears reversed'). "
        "Leave empty if no discrepancies found. This helps flag data quality issues "
        "without forcing incorrect extractions.",
    }
    # Note: data_discrepancy_notes is NOT added to required_full - it's optional

    # Add early exit field for fundamentally inappropriate extractions
    selected_data_types_full["totally_inappropriate_extraction"] = {
        "type": "string",
        "description": "ONLY fill this if the extraction request is FUNDAMENTALLY inappropriate for this paper. "
        "Examples: paper is about completely different topic, paper is not a research study, "
        "paper doesn't contain ANY of the requested data types (not just some missing fields). "
        "If filled, the entire extraction is marked invalid. "
        "Leave EMPTY for normal extractions, even partial ones. "
        "Use this ONLY when extraction makes no sense, not for difficult or incomplete extractions.",
    }
    # Note: totally_inappropriate_extraction is NOT added to required_full - it's optional

    selected_data_types_full["successfully_extracted"] = {
        "type": "boolean",
        "description": "Was the data successfully extracted and were all required fields populated "
        "correctly and were any not required fields that were included correctly "
        "populated (with no empty strings or blanks)?",
    }
    # required.append(name + "_successfully_extracted")
    required_full.append("successfully_extracted")
    tool = {
        "type": "function",
        "function": {
            "name": "extract_data",
            "description": "Stores accurate data from this research paper along with all quotations "
            "needed to source the data. Important: if needed separate different quotes by '... '. "
            "If a property/parameter is not required and you can not find information for it, "
            "exclude it completely DO NOT leave it blank.",
            "parameters": {
                "type": "object",
                "properties": selected_data_types_full.copy(),
                "additionalProperties": False,
                "required": required_full.copy(),
            },
        },
    }

    # Sanitize the schema to ensure Vertex AI compliance
    # This ensures all fields in 'required' are defined in 'properties' at all nesting levels
    tool = _sanitize_schema_for_vertex(tool)

    return tool


def verify_computation(derivation, expected_value):
    """Verify that derivation computation produces the expected value

    Args:
        derivation: Dict with 'operation', 'sources', etc.
        expected_value: The value that should result from the computation

    Returns:
        bool: True if computation is correct (or cannot be verified), False if incorrect
    """
    try:
        operation = derivation.get("operation")
        sources = derivation.get("sources", [])

        if not sources:
            return False

        # Extract numeric values from sources
        values = []
        for s in sources:
            extracted = s.get("extracted_value")
            if extracted is None:
                return True  # Cannot verify if values missing
            try:
                values.append(float(extracted))
            except (ValueError, TypeError):
                # Non-numeric derivation (lookup, custom) - cannot auto-verify
                return True

        # Perform computation based on operation type
        if operation == "sum":
            computed = sum(values)
        elif operation == "subtraction":
            computed = values[0]
            for v in values[1:]:
                computed -= v
        elif operation == "multiplication":
            computed = 1
            for v in values:
                computed *= v
        elif operation == "division":
            computed = values[0]
            for v in values[1:]:
                if v == 0:
                    return False  # Division by zero
                computed /= v
        elif operation == "average":
            computed = sum(values) / len(values)
        elif operation in ["lookup", "custom"]:
            # Cannot automatically verify lookup/custom operations
            return True
        else:
            # Unknown operation
            return False

        # Compare with expected value (allow small floating point errors)
        try:
            expected_float = float(expected_value)
            return abs(computed - expected_float) < 0.001
        except (ValueError, TypeError):
            return False

    except Exception as e:  # Catch any unexpected errors during verification
        logger.warning(f"Error verifying computation:  {e}")
        return True  # Be lenient on verification errors


async def reflect_on_data_extraction(
    extraction_dict,
    corpus,
    retries=3,
    limit_corpus=True,
    justification=None,
    return_problem_fields=False,
    collection_message="",
):
    """
    Validates data extraction results against source corpus with support for both direct quotes
    and derived/summarized information.

    Args:
        extraction_dict: Dictionary containing extracted data with source quotes and values
        corpus: Original text source
        retries: Number of reflection attempts
        limit_corpus: Whether to include full corpus in reflection prompt
        justification: Optional justification from extractor for borderline cases
        return_problem_fields: If True, returns (error_msg, [field_names]) tuple instead of just error_msg
        collection_message: Purpose/use context - determines justification standard for derivations
    """

    def replace_numbers_with_words(text):
        def replacer(match):
            number = match.group()
            return num2words(int(number))

        return re.sub(r"\d+", replacer, text)

    def check_source(source, pre_processed_corpus):
        """Validates that source text exists within corpus after normalization"""
        # FIX: Handle None explicitly
        if source is None:
            return "Source quote is None"
        source = replace_numbers_with_words(source.lower())
        source = re.sub(r"[\"'`''" "]", "", source)
        sources = re.split(r"\.\.\.|\. ", source)
        for source in sources:
            source = "".join(e for e in source if e.isalnum())
            if source not in pre_processed_corpus:
                return f"Source not found: {source}"
        return None

    # Helper to format returns based on return_problem_fields setting
    def make_return(error_msg, problem_fields=None):
        if return_problem_fields:
            return (error_msg, problem_fields or [])
        return error_msg

    # Pre-process corpus for comparison
    pre_processed_corpus = corpus.lower()
    pre_processed_corpus = replace_numbers_with_words(pre_processed_corpus)
    pre_processed_corpus = re.sub(r"[\"'`''" "]", "", pre_processed_corpus)
    pre_processed_corpus = "".join(e for e in pre_processed_corpus if e.isalnum())

    if not extraction_dict.get("successfully_extracted", False):
        return make_return("Data not successfully extracted: " + str(extraction_dict))

    # Validate source quotes OR derivations
    for key, value in extraction_dict.items():
        if key.endswith("_source_quote"):
            # Check if this field has a corresponding derivation instead
            derivation_key = key.replace("_source_quote", "_derivation")

            if derivation_key in extraction_dict:
                # DERIVATION path - validate the derivation
                derivation = extraction_dict[derivation_key]

                # CRITICAL: Validate that derivation is a dict, not a string
                if not isinstance(derivation, dict):
                    field_base = key.replace("_source_quote", "")
                    return make_return(
                        f"Derivation field '{derivation_key}' must be an object/dictionary with operation, sources, computation, etc. Got type {type(derivation).__name__} instead. Value: {derivation}",
                        [field_base],
                    )

                # 1. Validate all source quotes in derivation exist in corpus
                for source_idx, source_obj in enumerate(derivation.get("sources", [])):
                    quote = source_obj.get("quote")
                    if not quote:
                        field_base = key.replace("_source_quote", "")
                        return make_return(f"Derivation {key}: source {source_idx} missing quote", [field_base])

                    quote_check = check_source(quote, pre_processed_corpus)
                    if quote_check is not None:
                        field_base = key.replace("_source_quote", "")
                        return make_return(f"Derivation {key}: {quote_check} (source {source_idx})", [field_base])

                # 2. Verify computation is correct
                value_key = key.replace("_source_quote", "_value")
                if not value_key.endswith("_value"):  # Handle numerator/denominator
                    value_key = key.replace("_source_quote", "")

                if value_key in extraction_dict:
                    if not verify_computation(derivation, extraction_dict[value_key]):
                        return make_return(
                            f"Derivation computation incorrect for {value_key}: expected {extraction_dict[value_key]}, check {derivation.get('computation')}",
                            [value_key.replace("_value", "")],
                        )

                logger.debug(f"Validated derivation for {value_key}")

            elif value is not None and value != "":
                # DIRECT QUOTE path - existing validation
                source_check = check_source(value, pre_processed_corpus)
                if source_check is not None:
                    field_base = key.replace("_source_quote", "")
                    return make_return(
                        "The value of '"
                        + key
                        + "' not found in corpus, update this value so its contents can be found verbatim in the corpus (seperate sections broken up by '...'): "
                        + str(value),
                        [field_base],
                    )

                # NEW: Check if the extracted VALUE is actually supported by this quote
                # This prevents using a quote like "10...13" to justify a value of "23" without a derivation
                value_key = key.replace("_source_quote", "_value")
                if not value_key.endswith("_value"):
                    value_key = key.replace("_source_quote", "")

                if value_key in extraction_dict:
                    extracted_val = extraction_dict[value_key]
                    # Skip validation for booleans - they represent interpretations, not literal values
                    # Note: must check bool BEFORE int because bool is a subclass of int in Python
                    if isinstance(extracted_val, bool):
                        pass  # Booleans are inferences from quotes, not literal matches
                    # Only check numeric values or short strings to avoid false positives on long text
                    elif isinstance(extracted_val, int | float) or (
                        isinstance(extracted_val, str) and len(extracted_val) < 20
                    ):
                        # Normalize quote and value for comparison
                        norm_quote = str(value).lower()
                        norm_val = str(extracted_val).lower()

                        # Simple check: is the value inside the quote?
                        # We also check num2words for numbers (e.g. "twenty-three" vs 23)
                        val_in_quote = norm_val in norm_quote

                        if not val_in_quote and isinstance(extracted_val, int | float):
                            # Try word form for numbers
                            try:
                                word_val = num2words(int(extracted_val)).lower()
                                val_in_quote = word_val in norm_quote
                            except Exception:  # nosec # Catch any errors during num2words conversion
                                # Ignore errors during value checking
                                pass

                        if not val_in_quote and isinstance(extracted_val, int | float):
                            # If value is a number and NOT in the quote, it likely requires derivation (sum/calc)
                            field_base = value_key.replace("_value", "")
                            return make_return(
                                f"Value {extracted_val} for '{value_key}' is NOT found in the source quote '{value}'. "
                                f"If this value was calculated (e.g., sum of multiple numbers), you MUST use '_derivation' "
                                f"instead of '_source_quote'.",
                                [field_base],
                            )

                logger.debug(f"Validated source for {key}")

            elif value == "":
                field_base = key.replace("_source_quote", "")
                return make_return(
                    "If a property/parameter is not required and you can not find information for it, "
                    "exclude it completely DO NOT leave it blank. " + str(key),
                    [field_base],
                )
            # else: value is None and no derivation - field might be optional, let it pass

        if isinstance(value, list):
            for item in value:
                for k in item:
                    if k.endswith("_source_quote"):
                        if check_source(item[k], pre_processed_corpus) is not None:
                            field_base = k.replace("_source_quote", "")
                            return make_return(
                                "Source was not found in corpus: (" + k + ") " + str(item[k]), [field_base]
                            )
                        logger.debug(f"Validated source for {k}")
                        if item[k] == "":
                            field_base = k.replace("_source_quote", "")
                            return make_return(
                                "If a property/parameter is not required and you can not find information for it, "
                                "exclude it completely DO NOT leave it blank. " + str(k),
                                [field_base],
                            )

    system_message = f"""You are a careful data analyst validating information extraction results.
Your task is to verify that all extracted DATA VALUES are justified by the source material.

COLLECTION MESSAGE (Purpose/Justification Standard):
{collection_message}
This tells you HOW the data will be used, which determines the JUSTIFICATION STANDARD required:

HIGH-RIGOR (meta-analysis, pooling):
- Use this context to judge whether derivations have COMPLETE documentation
- ✅ VALID: Derivation with full computation chain AND all source quotes
- ❌ INVALID: Derivation with missing steps or incomplete source quotes

STANDARD-RIGOR (summary, overview):
- Derivations with reasonable documentation are acceptable
- Inference chains can be more compact

CRITICAL INSTRUCTION:
You must IGNORE metadata fields such as 'source_location', 'successfully_extracted', and 'data_discrepancy_notes'
when checking for direct support. These fields are generated by the extractor and are NOT expected to be found in the source text.
Focus ONLY on validating the actual data content (values, units, descriptions) and their corresponding 'source_quote'.

Valid extractions include:
1. Direct quotes and explicit statements from the source
2. Numerical values clearly stated in or computed from the source
3. Summaries that accurately capture information from one or more source statements
4. Labels, categories, or groupings that logically organize information present in the source
5. Derived values (min/max, averages, ranges, etc.) calculated from source data
6. Normalized or standardized versions of source information

**HANDLING CONFLICTING SOURCE DATA:**
If the 'data_discrepancy_notes' field documents conflicts (e.g., "Abstract says n=150 but Table shows n=142"):
- This is VALID and should NOT cause validation failure
- The extractor correctly identified and documented the conflict
- Verify the CHOSEN value matches ONE of the documented conflicting sources
- Extractions that document discrepancies are BETTER than forcing incorrect reconciliations

**CRITICAL GROUP MAPPING VALIDATION (for comparative/stratified data):**
When the extraction contains group comparisons (exposed vs reference, group1 vs group2, etc.):
1. Check that GROUP LABELS match their corresponding VALUES
2. If a source quote mentions "Group A (n=X)" and exposed_group_label="Group A", then exposed_group_n MUST be X
3. Watch for SWAPPED GROUPS - this is a common error where values for group A are placed in group B's columns
4. Verify the sample sizes (n) align with the correct group labels throughout

The key requirement is that ALL extracted DATA VALUES must be fully supportable using ONLY
the provided source material. Additions, assumptions, or external knowledge beyond the source are not allowed.
However, do NOT fail validation just because 'Page 1' or 'Abstract' (location metadata) is not in the text."""

    user_message = (
        "Please validate if this data extraction is fully justified by its sources:\n\n"
        "Data Extracted:\n\n" + str(extraction_dict)
    )

    if not limit_corpus:
        user_message += "\n\nSource Material:" + corpus
    else:
        user_message += "\n\nAre all information elements fully justified by their corresponding sources?"

    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

    arguments = {"messages": messages, "model": get_config().default_fast_model}

    if justification:
        arguments["messages"] += [{"role": "user", "content": "Here is the corpus:\n\n" + corpus}]
        arguments["messages"] += [
            {
                "role": "user",
                "content": "Here is the justification for the failure if it feels the extraction was successful and it is not getting validated which will be passed to the reflect_on_data_extraction function\n\n"
                + justification,
            }
        ]

    chat_response = await async_client.chat.completions.create(**arguments)
    messages.append({"role": "assistant", "content": chat_response.choices[0].message.content})

    tools = [
        {
            "type": "function",
            "function": {
                "strict": True,
                "name": "check_extracted_data",
                "description": """Validates if extracted data is fully justified by source material.

Extracted information is valid if the DATA VALUES can be supported entirely by the sources.
IGNORE metadata fields like 'source_location' and 'successfully_extracted' - these do not need to be in the source text.

Valid justifications include:
- Direct quotation of source text
- Accurate summarization of source content
- Logical grouping or categorization of source information
- Mathematical calculations or derivations from source values
- Normalization or standardization of source information

The key requirement is that all DATA VALUES must be clearly traceable to and supported
by the source material alone.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "valid": {
                            "type": "boolean",
                            "description": "Whether the extracted data is 100% justified by sources",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Detailed explanation of validation result, especially if invalid",
                        },
                        "problem_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of field names (without _value/_source_quote suffix) that have validation issues. Empty if valid=true.",
                        },
                    },
                    "required": ["valid", "reason", "problem_fields"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    arguments = {
        "messages": messages,
        "model": get_config().default_fast_model,
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "check_extracted_data"}},
    }

    if justification:
        arguments["model"] = get_model_for_role(MODEL_REASONING)
        arguments["reasoning_effort"] = "high"

    retry = 0
    while retry < retries:
        if retry > 0:
            logger.info("Retrying validation...")
        chat_response = await async_client.chat.completions.create(**arguments)
        if chat_response.choices[0].message.tool_calls:
            valid_calls = use_tools(chat_response, arguments, call_functions=False)
            if valid_calls:
                for call in valid_calls:
                    if call["name"] == "check_extracted_data":
                        if call["parameters"]["valid"]:
                            logger.info("Data extraction validated successfully")
                            if return_problem_fields:
                                return (None, [])
                            return None
                        else:
                            problem_fields = call["parameters"].get("problem_fields", [])
                            return make_return(
                                f"Data extraction failed validation: {call['parameters']['reason']}", problem_fields
                            )
        retry += 1
    return make_return("Data extraction validation failed to complete")


async def extract_data(
    tool, corpus, retries=4, mode=ExtractionMode.FOCUSED, inappropriate_exit_threshold=2, collection_message=""
):
    """
    Extract data from a corpus using the provided tool.

    Args:
        tool: The extraction tool schema
        corpus: The source text to extract from
        retries: Number of retry attempts
        mode: ExtractionMode controlling strictness:
              - EXPLORATORY: Lenient, removes problem fields on last retry, no early exit
              - FOCUSED: Default balanced mode with convergence detection
              - RIGID: Strict mode (all fields required in schema)
        inappropriate_exit_threshold: Number of consecutive 'totally_inappropriate_extraction'
                                     signals before early exit (only in FOCUSED/RIGID modes)
        collection_message: Purpose/use context - determines justification standard for derivations

    Returns:
        dict: Extracted data or {"failed_collection": reason}
    """

    system_message = f"""You are a careful data analyst. Dutifully find the data in the provided research paper.

COLLECTION MESSAGE (Purpose/Justification Standard):
{collection_message}
This tells you HOW the data will be used, which determines the JUSTIFICATION STANDARD required:

HIGH-RIGOR (meta-analysis, pooling):
- Derivations ARE acceptable but must have COMPLETE documentation
- ✅ GOOD: "Total N = 45" derived from "Group A: 23 patients" + "Group B: 22 patients" with full computation chain
- ❌ BAD: "Total N = 45" with only one source quote or missing computation
- Every step must be traceable to verbatim quotes

STANDARD-RIGOR (summary, overview):
- Derivations acceptable with reasonable documentation
- ✅ GOOD: "Total N = 45" with operation and at least one supporting quote
- Inference chains can be more compact

IMPORTANT INSTRUCTIONS FOR DATA EXTRACTION:

1. SOURCE QUOTES vs DERIVATIONS:
   - Use 'source_quote' fields when the value is DIRECTLY STATED in the paper as a single quote
   - Use 'derivation' fields when the value requires CALCULATION from multiple sources (e.g., summing numbers, looking up values from different sections)
   - NEVER use both source_quote and derivation for the same field - choose ONE

2. DERIVATION STRUCTURE (when calculation is needed):
   A derivation field must be a properly structured object with these required components:
   {{
     "operation": one of ["sum", "subtraction", "division", "multiplication", "lookup", "average", "custom"],
     "operation_description": "Human-readable explanation of what was calculated",
     "sources": [
       {{
         "quote": "Exact quote from paper",
         "location": "Page/section where found",
         "extracted_value": numeric_value_from_this_quote
       }},
       // ... more source objects as needed
     ],
     "computation": "The actual formula, e.g., '13 + 10 = 23'"
   }}

   IMPORTANT FOR DERIVATIONS:
   - For multi-step inferences, include quotes from MULTIPLE sections of the paper
   - A single quote (especially from just the title) is rarely sufficient to justify a derived conclusion
   - Include supporting evidence for each logical step in your derivation chain
   - Build the chain: quote A + quote B → intermediate conclusion → final derived value

3. EXAMPLES:
   - Direct quote: If paper says "23 patients", use source_quote = "23 patients"
   - Derivation needed: If paper says "13 in group A" and "10 in group B", and you need total, use derivation with operation="sum"

4. CRITICAL FOR COMPARATIVE/GROUP DATA:
   When extracting data for multiple groups (e.g., exposed vs reference, treatment vs control):

   a) VERIFY GROUP-VALUE ALIGNMENT: Before filling in values, explicitly identify:
      - Which group label goes in 'exposed_group' columns
      - Which group label goes in 'reference_group' columns
      - Confirm sample sizes (n) match the correct group

   b) COMMON ERROR TO AVOID: When a source quote mentions BOTH groups like:
      "Group A (n=9, mean 18.2) vs Group B (n=21, mean 15.8)"
      DO NOT accidentally swap the values! Verify each value goes with its correct group.

   c) SELF-CHECK: After extraction, mentally read back:
      "[exposed_label] had [exposed_n] participants with [exposed_value]"
      Confirm this matches what the paper actually states.

5. DATA DISCREPANCY NOTES:
   If you find CONFLICTING or INCONSISTENT data in the paper, use the 'data_discrepancy_notes' field to document it:
   - Example: "Abstract reports n=150 for control group but Table 2 shows N=142 for the same group"
   - Example: "CI appears reversed in Table 3 (1.25-0.78 instead of 0.78-1.25)"
   - Example: "Follow-up rate 95% of 200 = 190, but Results section states n=187"

   This is BETTER than:
   - Forcing one value when sources conflict
   - Inventing reconciliations not stated in the paper
   - Failing the extraction entirely

   When conflicts exist, extract the most reliable source (typically Tables > Abstract > Text)
   and document the discrepancy.

6. TOTALLY INAPPROPRIATE EXTRACTION:
   If this paper is FUNDAMENTALLY inappropriate for this extraction (completely different topic,
   not a research study, contains NONE of the requested data types), fill in the
   'totally_inappropriate_extraction' field explaining why. This will end the extraction early.
   Use this ONLY for fundamental mismatches, NOT for difficult or partial extractions.

DO NOT provide derivation as a string - it must be a structured object as shown above."""

    user_message = "Extract the requested data using the extract_data tool:\n\n" + corpus + "\n"

    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

    arguments = {
        "messages": messages,
        "tools": [tool],
        "model": get_model_for_role(MODEL_REASONING),
        "reasoning_effort": "high",
        "tool_choice": {"type": "function", "function": {"name": tool["function"]["name"]}},
    }

    retry = 0
    output_dictionary = None
    explanation = None

    # Track attempts and feedback for convergence detection
    attempts = []
    feedbacks = []
    inappropriate_count = 0
    convergence_retry_used = False
    last_problem_fields = []

    while not output_dictionary and retry < retries:
        if retry > 0:
            logger.info("Retrying...")
        chat_response = await async_client.chat.completions.create(**arguments)
        if chat_response.choices[0].message.tool_calls:
            valid_calls = use_tools(chat_response, arguments, call_functions=False)
            if valid_calls:
                for call in valid_calls:
                    if call["name"] == tool["function"]["name"]:
                        output_dictionary = call["parameters"]

                        # Check for totally_inappropriate_extraction (FOCUSED/RIGID modes only)
                        inappropriate_msg = output_dictionary.get("totally_inappropriate_extraction", "")
                        if inappropriate_msg and mode != ExtractionMode.EXPLORATORY:
                            inappropriate_count += 1
                            logger.warning(
                                f"Inappropriate extraction signal ({inappropriate_count}/{inappropriate_exit_threshold}): {inappropriate_msg[:100]}..."
                            )
                            if inappropriate_count >= inappropriate_exit_threshold:
                                # Early exit - paper is not appropriate for this extraction
                                return {"failed_collection": f"Paper inappropriate for extraction: {inappropriate_msg}"}
                        else:
                            inappropriate_count = 0  # Reset if extraction proceeds normally

                        # Track this attempt
                        attempts.append(output_dictionary.copy())

                        # Validate - in EXPLORATORY mode on last retry, get problem field names
                        is_last_retry = retry + 1 >= retries
                        use_field_tracking = mode == ExtractionMode.EXPLORATORY and is_last_retry

                        check_result = await reflect_on_data_extraction(
                            output_dictionary,
                            corpus,
                            return_problem_fields=use_field_tracking,
                            collection_message=collection_message,
                        )

                        # Handle return format based on mode
                        if use_field_tracking:
                            check, problem_fields = check_result
                            last_problem_fields = problem_fields
                        else:
                            check = check_result

                        feedbacks.append(check if check else "Success")

                        if check is not None:
                            if is_last_retry and not check.startswith("The value of '"):
                                logger.warning(f"Data extraction failed (last): {check}")
                                new_message = {
                                    "role": "system",
                                    "content": "This data extraction may or may not have been completed successfully:\n\n"
                                    + str(output_dictionary)
                                    + "\n\n\nTheir is skepticism surrounding this extraction for the following reason:\n"
                                    + check
                                    + "\n\n This last skepticsm came from an analysis that did not include the corpus text - therefor it is not a final verdict and you may need to justify it further.",
                                }
                                justification_message = {
                                    "role": "user",
                                    "content": "Please explain the extraction process and if you feel its appropriate make a case for why this extraction is valid despite the issues.",
                                }
                                arguments["messages"] += [justification_message]
                                del arguments["tools"]
                                del arguments["tool_choice"]
                                chat_response = await async_client.chat.completions.create(**arguments)
                                if chat_response.choices[0].message.content:
                                    justification = chat_response.choices[0].message.content
                                    check = await reflect_on_data_extraction(
                                        output_dictionary,
                                        corpus,
                                        justification=justification,
                                        collection_message=collection_message,
                                    )
                                    if check is not None:
                                        # EXPLORATORY mode: remove problem fields and return partial data
                                        if mode == ExtractionMode.EXPLORATORY and last_problem_fields:
                                            logger.info(
                                                f"Exploratory mode: removing problem fields {last_problem_fields}"
                                            )
                                            output_dictionary = _remove_problem_fields(
                                                output_dictionary, last_problem_fields
                                            )
                                            # Continue to success path
                                        else:
                                            explanation = check + "\n\nLast attempt:\n\n" + str(output_dictionary)
                                            output_dictionary = None
                                    else:
                                        logger.info("Data extraction successful at last attempt")
                            elif is_last_retry:
                                # EXPLORATORY mode: remove problem fields and return partial data
                                if mode == ExtractionMode.EXPLORATORY and last_problem_fields:
                                    logger.info(f"Exploratory mode: removing problem fields {last_problem_fields}")
                                    output_dictionary = _remove_problem_fields(output_dictionary, last_problem_fields)
                                    # Continue to success path
                                else:
                                    explanation = check
                            else:
                                logger.warning(f"Data extraction failed: {check} Retrying...")
                                new_message = {
                                    "role": "system",
                                    "content": (
                                        "This data extraction was not completed successfully:\n\n"
                                        + str(output_dictionary)
                                        + "\n\n\nThis was not successful for the following reason:\n"
                                        + check
                                        + "\n\n Please try to address this issue."
                                    ),
                                }
                                arguments["messages"] += [new_message]
                                output_dictionary = None
        else:
            logger.warning("No tool calls used")
        retry += 1

        # Check for convergence after normal retries exhausted (FOCUSED/RIGID modes only)
        if retry >= retries and output_dictionary is None and not convergence_retry_used:
            if mode != ExtractionMode.EXPLORATORY and len(attempts) >= 2:
                logger.debug("Checking for convergence...")
                convergence = await detect_convergence(attempts, feedbacks)
                if convergence.get("converging", False):
                    logger.info(f"Convergence detected: {convergence['reason']} - granting retry extension")
                    retries += retries  # Double the retries for one more round
                    convergence_retry_used = True
                    # Restore tools if they were removed during justification flow
                    if "tools" not in arguments:
                        arguments["tools"] = [tool]
                        arguments["tool_choice"] = {"type": "function", "function": {"name": tool["function"]["name"]}}
                else:
                    logger.warning(f"Not converging: {convergence['reason']} - ending extraction")

    if output_dictionary is not None:
        del output_dictionary["successfully_extracted"]
        # Remove metadata fields
        if "data_discrepancy_notes" in output_dictionary and not output_dictionary["data_discrepancy_notes"]:
            del output_dictionary["data_discrepancy_notes"]
        if "totally_inappropriate_extraction" in output_dictionary:
            del output_dictionary["totally_inappropriate_extraction"]
        # Clean up null values that indicate "not reported"
        output_dictionary = clean_missing_values(output_dictionary)
        return output_dictionary.copy()
    if explanation:
        return {"failed_collection": explanation}
    return {}


def _remove_problem_fields(extraction_dict, problem_fields):
    """
    Remove fields identified as problematic from the extraction dictionary.

    Args:
        extraction_dict: The extraction dictionary
        problem_fields: List of field base names (without _value/_source_quote suffix)

    Returns:
        dict: Extraction with problem fields removed
    """
    if not problem_fields:
        return extraction_dict

    cleaned = {}
    for key, value in extraction_dict.items():
        # Check if this key belongs to a problem field
        is_problem = False
        for problem_field in problem_fields:
            if key.startswith(problem_field + "_") or key == problem_field:
                is_problem = True
                break

        if not is_problem:
            cleaned[key] = value

    return cleaned


def clean_missing_values(data):
    """
    Recursively remove null values from extraction results.

    - For dicts: removes keys where value is null/None
    - For lists: processes each item recursively
    - Preserves actual data (including 0 and False)
    """
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if value is None:
                continue
            elif isinstance(value, dict | list):
                cleaned_value = clean_missing_values(value)
                if cleaned_value or cleaned_value == 0 or cleaned_value is False:
                    cleaned[key] = cleaned_value
            else:
                cleaned[key] = value
        return cleaned
    elif isinstance(data, list):
        cleaned_list = []
        for item in data:
            if item is None:
                continue
            if isinstance(item, dict | list):  # Reverted to original logic as schema_or_tool is undefined
                cleaned_item = clean_missing_values(item)
                if cleaned_item or cleaned_item == 0 or cleaned_item is False:
                    cleaned_list.append(cleaned_item)
            else:
                cleaned_list.append(item)
        return cleaned_list
    else:
        return data


async def detect_convergence(attempts, feedbacks):
    """
    Analyze extraction attempts to determine if the process is converging or diverging.

    Args:
        attempts: List of extraction dictionaries from each attempt
        feedbacks: List of validation feedback strings from each attempt

    Returns:
        dict: {"converging": bool, "reason": str}
    """
    if len(attempts) < 2:
        return {"converging": False, "reason": "Not enough attempts to analyze"}

    system_message = """You are analyzing a data extraction process that has gone through multiple attempts.
Each attempt has produced an extraction and received feedback.

Your job is to determine if the extraction is CONVERGING (getting better, fixing issues) or DIVERGING (stuck, cycling, or getting worse).

Signs of CONVERGENCE:
- Error messages are changing (addressing previous issues)
- Number of errors is decreasing
- Extraction quality is improving
- The model is making meaningful corrections

Signs of DIVERGENCE:
- Same errors appearing repeatedly
- Errors cycling back and forth
- Quality getting worse or staying the same
- Model seems confused or stuck

Return your assessment via the tool."""

    # Format the attempts and feedbacks for analysis
    analysis_content = "Here are the extraction attempts and their feedback:\n\n"
    for i, (attempt, feedback) in enumerate(zip(attempts, feedbacks, strict=False)):
        analysis_content += f"=== ATTEMPT {i + 1} ===\n"
        analysis_content += f"Extraction (summarized): {str(attempt)[:500]}...\n"
        analysis_content += f"Feedback: {feedback}\n\n"

    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": analysis_content}]

    tools = [
        {
            "type": "function",
            "function": {
                "strict": True,
                "name": "assess_convergence",
                "description": "Report whether the extraction process is converging or diverging",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "converging": {
                            "type": "boolean",
                            "description": "True if the extraction is improving/converging, False if stuck/diverging",
                        },
                        "reason": {"type": "string", "description": "Brief explanation of the assessment"},
                    },
                    "required": ["converging", "reason"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    try:
        response = await async_client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "assess_convergence"}},
        )

        if response.choices[0].message.tool_calls:
            call = response.choices[0].message.tool_calls[0]
            result = json.loads(call.function.arguments)
            return result
    except Exception as e:
        logger.error(f"Convergence detection error: {e}")

    return {"converging": False, "reason": "Could not assess convergence"}
