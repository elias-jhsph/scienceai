import json
import os
import re

from num2words import num2words

from .llm import client, async_client, use_tools_sync as use_tools


data_types = None
data_types_file = "data_types.json"

data_types_docs = None
data_types_docs_file = "data_types_docs.json"

path_to_app = os.path.dirname(os.path.abspath(__file__))
data_types_file = os.path.join(path_to_app, data_types_file)
data_types_docs_file = os.path.join(path_to_app, data_types_docs_file)


def load_json_file(filename):
    """ Load and return the data from a JSON file. """
    with open(filename, 'r') as file:
        return json.load(file)


data_types = load_json_file(data_types_file)
data_types_docs = load_json_file(data_types_docs_file)


def validate_data_type_spec(spec, force_type=None):
    if force_type:
        if force_type not in data_types:
            return False, f"Data type '{force_type}' not found in data types."
        properties = data_types[force_type]['spec']
        specification = spec
    else:
        if 'type' not in spec:
            return False, "Key 'type' not found in specification."
        if spec['type'] not in data_types:
            return False, f"Data type '{spec['type']}' not found in data types."
        properties = data_types[spec['type']]['spec']
        specification = spec.copy()
        del specification['type']
    for key, value in properties.items():
        if key not in specification:
            return False, f"Key '{key}' not found in specification."
        if value['type'] == 'string':
            if not isinstance(specification[key], str):
                return False, f"Value for '{key}' is not a string."
        elif value['type'] == 'boolean':
            if not isinstance(specification[key], bool):
                return False, f"Value for '{key}' is not a boolean."
        elif value['type'] == 'number':
            if not isinstance(specification[key], int) and not isinstance(specification[key], float):
                return False, f"Value for '{key}' is not a number."
        elif value['type'] == 'object':
            try:
                for k, v in json.loads(specification[key]):
                    if k in value['keys']:
                        if value['keys'][k] == 'string':
                            if not isinstance(v, str):
                                return False, f"Value for '{k}' is not a string."
                        elif value['keys'][k] == 'boolean':
                            if not isinstance(v, bool):
                                return False, f"Value for '{k}' is not a boolean."
                        else:
                            return False, f"Invalid data type '{value['keys'][k]}' for key '{k}' in object spec."
            except json.JSONDecodeError:
                return False, f"Value for '{key}' is not a valid JSON object."
        elif value['type'] == 'array':
            if not isinstance(specification[key], list):
                return False, f"Value for '{key}' is not an array."
            for item in specification[key]:
                if not isinstance(item, str):
                    return False, f"Item in array '{key}' is not a string."
    if len(specification) != len(properties):
        return False, "specification has extra keys."
    return True, "specification is valid."


def generate_description(data_type_key, data_type_value, all_keys):
    """ Generate a description for each data type. """
    message = f"Data Type: {data_type_key}\n"
    message += f"Description: {data_type_value['spec_description']}\n"
    message += "Specifications Required:\n"
    properties = {}
    for spec_key, spec_value in data_type_value['spec'].items():
        message += f"  - {spec_key} ({spec_value['type']}): {spec_value['description']}\n"
        properties[spec_key] = {"type": spec_value['type'], "description": spec_value['description']}
        if spec_value['type'] == 'array':
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
                    "additionalProperties": False, "required": list(properties.keys())
                }
            }
        }
    ]

    system_message = ("You are a helpful AI assistant. You have been asked to generate example data for data types. "
                      "These data types are used to extract data from research papers. Each aspect of the example data "
                      "must be creative and scientific. You must follow the specifications provided for each "
                      "data type. Don't repeat the same example data for the same data type. Be creative! "
                      "If there is a boolean value, make sure to vary it between examples. "
                      "(Not every data type should be required: true!)"
                      "\n\n\n Keep in mind that the full list of data types is: " + ", ".join(all_keys) + "\n\n\n"
                      "Be sure to generate example data for the following data type: " + data_type_key + "\n\n"
                      "Make sure your example data is not too similar to other data types.")

    user_message_prefix = "Generate made up example data for this data type that is creative and scientific:\n\n"

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message_prefix + message
        }
    ]

    arguments = {"messages": messages, "tools": tools,  "model": "o4-mini", 'reasoning_effort': 'low',
                 "tool_choice": {"type": "function", "function": {"name": "generate_example_data"}}}

    examples = []
    retry = 0
    valid_calls = []
    output_dictionary = None
    while len(examples) < 3 and retry < 3:
        while valid_calls == [] and retry < 3:
            if retry > 0:
                print("Retrying...")
            chat_response = client.chat.completions.create(**arguments)
            if chat_response.choices[0].message.tool_calls:
                valid_calls = use_tools(chat_response, arguments, call_functions=False)
                if valid_calls:
                    for call in valid_calls:
                        if call["name"] == "generate_example_data":
                            output_dictionary = call["parameters"]
                            valid, error_message = validate_data_type_spec(output_dictionary, force_type=data_type_key)
                            if not valid:
                                print(error_message, output_dictionary)
                                valid_calls = []
                            else:
                                output_dictionary["type"] = data_type_key
            else:
                print("No tool calls used")
            retry += 1
        if valid_calls and output_dictionary:
            retry = 0
            messages += [{"role": "function", "name": valid_calls[0]["name"], "content": "valid"},
                                       {"role": "user", "content": user_message_prefix + message}]
            examples.append(output_dictionary)
            valid_calls = []

    return message, examples


for data_type in data_types:
    if data_type not in data_types_docs:
        data_types_docs[data_type] = {}
        data_types_docs[data_type]["description"], data_types_docs[data_type]["examples"] = (
            generate_description(data_type, data_types[data_type], list(data_types.keys())))
        with open(data_types_docs_file, 'w') as file:
            json.dump(data_types_docs, file, indent=2)


def generate_schema(corpus, goal=None, retries=5):
    # Construct the system message
    system_message = ("You are an expert data schema designer. Create a structured schema to extract specific "
                      "information from research papers.\n\n"
                      "CRITICAL: You must ONLY use these valid data types: " + ", ".join(data_types.keys()) + "\n\n"
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
                      "Design your schema to be specific and extractable from the papers. "
                      "Focus on: " + (goal if goal else "extracting relevant structured data") + "\n\n")

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

    final_ask = ("Please return a JSON string representing the schema for the analysis which should "
                 "be an array of objects with a 'type' key and other keys as "
                 "specified in the data types. Try to limit the number of objects requested to 8 or less.")

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
                                "type": {"type": "string", "enum": list(data_types.keys())}
                            },
                            "additionalProperties": True,
                            "required": ["name", "description", "required", "type"]
                        }
                    }
                },
                "additionalProperties": False,
                "required": ["fields"]
            }
        }
    }

    messages = [
        {
            "role": "user",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    retry = 0
    valid = False

    while not valid and retry < retries:
        try:
            o4_arguments = {"messages": messages, "model": "o4-mini", 'reasoning_effort': 'high', "tools": [function_schema],
                            "tool_choice": {"type": "function", "function": {"name": "define_schema"}}}
            response = client.chat.completions.create(**o4_arguments)

            valid_calls = use_tools(response, o4_arguments, call_functions=False)
            output_dictionary = {}
            if valid_calls:
                for call in valid_calls:
                    output_dictionary = call['parameters']

            # Validate the schema
            for field in output_dictionary.get("fields", []):
                valid, error_message = validate_data_type_spec(field)
                if not valid:
                    print(f"Validation Error: {error_message}")
                    break

            if valid:
                print("Schema generated successfully.", output_dictionary["fields"])
                return output_dictionary["fields"]

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing response: {e}")

        retry += 1
        print(f"Retrying... Attempt {retry}")

    return None


def schema_to_tool(schema):
    def camel_to_snake(s):
        return ''.join(['_' + c.lower() if c.isupper() else c for c in s]).lstrip('_')

    selected_data_types_full = {}

    required_full = []

    for data_type_requested in schema:
        required = []
        selected_data_types = {}
        data_type_def = data_types[data_type_requested["type"]]["tool"]
        if data_type_def["mode"] == "prefix":
            required_keys = list(data_type_def.keys())
            required_keys.remove("mode")
            name = data_type_requested["name"].replace(" ", "_")
            # Fields that should never be marked as required in the extraction tool
            # These are either metadata fields OR optional statistical fields that may not be reported
            optional_fields = [
                # Metadata fields (choose quote OR derivation)
                "source_quote", "derivation", "source_location", "unit", "computation",
                # Optional statistical fields that papers may not report
                "ci_level",           # Usually 95%, often not explicitly stated
                "is_reference",       # Only relevant for reference categories
                "t_statistic",        # Often not reported
                "df", "df_numerator", "df_denominator",  # Degrees of freedom often omitted
                "se",                 # Standard error often omitted when CI given
                "error_lower", "error_upper",  # Only for asymmetric errors
                # Sample statistics - papers report different subsets
                "mean", "sd", "median", "q1", "q3", "min", "max",
                # Survival analysis - optional fields
                "log_rank_p", "rate_at_timepoint", "rate_timepoint",
                "median_survival_ci_lower", "median_survival_ci_upper",
                # Contingency table - totals can be computed
                "total_exposed", "total_unexposed",
            ]
            for key in required_keys:
                if data_type_requested["required"]:
                    # Only make core value fields required, not metadata or optional stats
                    if key not in optional_fields:
                        required.append(name + "_" + key)
                new_type = data_type_def[key]["type"]
                new_description = data_type_def[key]["description"]
                for spec_key in data_types[data_type_requested["type"]]["spec"]:
                    if spec_key == 'name':
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
                        "additionalProperties": False
                    }
                else:
                    selected_data_types[name + "_" + key] = {"type": new_type, "description": new_description}
        if data_type_def["mode"] == "array":
            name = data_type_requested["name"].replace(" ", "_")
            if data_type_requested["required"]:
                required.append(name)
            new_description = data_type_def["description"]
            for spec_key in data_types[data_type_requested["type"]]["spec"]:
                if spec_key == 'name':
                    new_description = new_description.replace("NAME", name)
                else:
                    new_description = new_description.replace(spec_key.upper(), str(data_type_requested[spec_key]))

            item_properties = {}
            for key in data_type_def["keys"]:
                new_type = data_type_def["keys"][key]["type"]
                new_description = data_type_def["keys"][key]["description"]
                for spec_key in data_types[data_type_requested["type"]]["spec"]:
                    if spec_key == 'name':
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
                        "additionalProperties": False
                    }
                else:
                    item_properties[key] = {"type": new_type, "description": new_description}

            selected_data_types[name] = {"type": "array", "description": new_description,
                                         "items": {"type": "object", "properties": item_properties,
                                                   "additionalProperties": False, "required": list(item_properties.keys())}}
        required_full = required_full + required
        selected_data_types_full = {**selected_data_types_full, **selected_data_types}

    # Add optional discrepancy notes field for documenting source conflicts
    selected_data_types_full["data_discrepancy_notes"] = {
        "type": "string", 
        "description": "OPTIONAL: Document any inconsistencies or conflicts found in the source material "
                       "(e.g., 'Abstract reports n=150 but Table 2 shows n=142 for the same group', "
                       "'CI reported as 1.25-0.78 which appears reversed'). "
                       "Leave empty if no discrepancies found. This helps flag data quality issues "
                       "without forcing incorrect extractions."
    }
    # Note: data_discrepancy_notes is NOT added to required_full - it's optional

    selected_data_types_full["successfully_extracted"] = \
        {"type": "boolean", "description": "Was the data successfully extracted and were all required fields populated "
                                           "correctly and were any not required fields that were included correctly "
                                           "populated (with no empty strings or blanks)?"}
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
                "required": required_full.copy()
            }
        }
    }

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
            
    except Exception as e:
        print(f"Error verifying computation:  {e}")
        return True  # Be lenient on verification errors


async def reflect_on_data_extraction(extraction_dict, corpus, retries=3, limit_corpus=True, justification=None):
    """
    Validates data extraction results against source corpus with support for both direct quotes
    and derived/summarized information.

    Args:
        extraction_dict: Dictionary containing extracted data with source quotes and values
        corpus: Original text source
        retries: Number of reflection attempts
        limit_corpus: Whether to include full corpus in reflection prompt
    """

    def replace_numbers_with_words(text):
        def replacer(match):
            number = match.group()
            return num2words(int(number))

        return re.sub(r'\d+', replacer, text)

    def check_source(source, pre_processed_corpus):
        """Validates that source text exists within corpus after normalization"""
        # FIX: Handle None explicitly
        if source is None:
            return "Source quote is None"
        source = replace_numbers_with_words(source.lower())
        source = re.sub(r"[\"'`''""]", '', source)
        sources = re.split(r"\.\.\.|\. ", source)
        for source in sources:
            source = ''.join(e for e in source if e.isalnum())
            if source not in pre_processed_corpus:
                return (f"Source not found: {source}")
        return None

    # Pre-process corpus for comparison
    pre_processed_corpus = corpus.lower()
    pre_processed_corpus = replace_numbers_with_words(pre_processed_corpus)
    pre_processed_corpus = re.sub(r"[\"'`''""]", '', pre_processed_corpus)
    pre_processed_corpus = ''.join(e for e in pre_processed_corpus if e.isalnum())

    if not extraction_dict.get("successfully_extracted", False):
        return "Data not successfully extracted: " + str(extraction_dict)

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
                    return f"Derivation field '{derivation_key}' must be an object/dictionary with operation, sources, computation, etc. Got type {type(derivation).__name__} instead. Value: {derivation}"
                
                # 1. Validate all source quotes in derivation exist in corpus
                for source_idx, source_obj in enumerate(derivation.get("sources", [])):
                    quote = source_obj.get("quote")
                    if not quote:
                        return f"Derivation {key}: source {source_idx} missing quote"
                    
                    quote_check = check_source(quote, pre_processed_corpus)
                    if quote_check is not None:
                        return f"Derivation {key}: {quote_check} (source {source_idx})"
                
                # 2. Verify computation is correct
                value_key = key.replace("_source_quote", "_value")
                if not value_key.endswith("_value"):  # Handle numerator/denominator
                    value_key = key.replace("_source_quote", "")
                    
                if value_key in extraction_dict:
                    if not verify_computation(derivation, extraction_dict[value_key]):
                        return f"Derivation computation incorrect for {value_key}: expected {extraction_dict[value_key]}, check {derivation.get('computation')}"
                
                print(f"Validated derivation for {value_key}")
            
            elif value is not None and value != "":
                # DIRECT QUOTE path - existing validation
                source_check = check_source(value, pre_processed_corpus)
                if source_check is not None:
                    return "The value of '" + key + "' not found in corpus, update this value so its contents can be found verbatim in the corpus (seperate sections broken up by '...'): " + str(value)
                
                # NEW: Check if the extracted VALUE is actually supported by this quote
                # This prevents using a quote like "10...13" to justify a value of "23" without a derivation
                value_key = key.replace("_source_quote", "_value")
                if not value_key.endswith("_value"):
                    value_key = key.replace("_source_quote", "")
                
                if value_key in extraction_dict:
                    extracted_val = extraction_dict[value_key]
                    # Only check numeric values or short strings to avoid false positives on long text
                    if isinstance(extracted_val, (int, float)) or (isinstance(extracted_val, str) and len(extracted_val) < 20):
                        # Normalize quote and value for comparison
                        norm_quote = str(value).lower()
                        norm_val = str(extracted_val).lower()
                        
                        # Simple check: is the value inside the quote?
                        # We also check num2words for numbers (e.g. "twenty-three" vs 23)
                        val_in_quote = norm_val in norm_quote
                        
                        if not val_in_quote and isinstance(extracted_val, (int, float)):
                            # Try word form for numbers
                            try:
                                word_val = num2words(int(extracted_val)).lower()
                                val_in_quote = word_val in norm_quote
                            except:
                                pass
                        
                        if not val_in_quote and isinstance(extracted_val, (int, float)):
                            # If value is a number and NOT in the quote, it likely requires derivation (sum/calc)
                            return (f"Value {extracted_val} for '{value_key}' is NOT found in the source quote '{value}'. "
                                    f"If this value was calculated (e.g., sum of multiple numbers), you MUST use '_derivation' "
                                    f"instead of '_source_quote'.")

                print(f"Validated source for {key}")
                
            elif value == "":
                return ("If a property/parameter is not required and you can not find information for it, "
                        "exclude it completely DO NOT leave it blank. ") + str(key)
            # else: value is None and no derivation - field might be optional, let it pass

        if isinstance(value, list):
            for item in value:
                for k in item:
                    if k.endswith("_source_quote"):
                        if check_source(item[k], pre_processed_corpus) is not None:
                            return "Source was not found in corpus: (" + k + ") " + str(extraction_dict[k])
                        print(f"Validated source for {k}")
                        if item[k] == "":
                            return ("If a property/parameter is not required and you can not find information for it, "
                                    "exclude it completely DO NOT leave it blank. ") + str(k)

    system_message = """You are a careful data analyst validating information extraction results.
Your task is to verify that all extracted DATA VALUES are justified by the source material.

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

    user_message = ("Please validate if this data extraction is fully justified by its sources:\n\n"
                    "Data Extracted:\n\n" + str(extraction_dict))

    if not limit_corpus:
        user_message += "\n\nSource Material:" + corpus
    else:
        user_message += "\n\nAre all information elements fully justified by their corresponding sources?"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    arguments = {"messages": messages, "model": "gpt-5-mini"}

    if justification:
        arguments["messages"] += [{"role": "user", "content": "Here is the corpus:\n\n" + corpus}]
        arguments["messages"] += [{"role": "user", "content": "Here is the justification for the failure if it feels the extraction was successful and it is not getting validated which will be passed to the reflect_on_data_extraction function\n\n" + justification}]

    
    chat_response = await async_client.chat.completions.create(**arguments)
    messages.append({"role": "assistant", "content": chat_response.choices[0].message.content})

    tools = [{
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
                        "description": "Whether the extracted data is 100% justified by sources"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of validation result, especially if invalid"
                    }
                },
                "required": ["valid", "reason"],
                "additionalProperties": False
            }
        }
    }]

    arguments = {
        "messages": messages,
        "model": "gpt-5-mini",
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "check_extracted_data"}}
    }

    if justification:
        arguments["model"] = "o4-mini"
        arguments["reasoning_effort"] = "high"

    retry = 0
    while retry < retries:
        if retry > 0:
            print("Retrying validation...")
        chat_response = await async_client.chat.completions.create(**arguments)
        if chat_response.choices[0].message.tool_calls:
            valid_calls = use_tools(chat_response, arguments, call_functions=False)
            if valid_calls:
                for call in valid_calls:
                    if call["name"] == "check_extracted_data":
                        if call["parameters"]["valid"]:
                            print("Data extraction validated successfully")
                            return None
                        else:
                            return f"Data extraction failed validation: {call['parameters']['reason']}"
        retry += 1
    return "Data extraction validation failed to complete"


async def extract_data(tool, corpus, retries=5):

    system_message = """You are a careful data analyst. Dutifully find the data in the provided research paper.

IMPORTANT INSTRUCTIONS FOR DATA EXTRACTION:

1. SOURCE QUOTES vs DERIVATIONS:
   - Use 'source_quote' fields when the value is DIRECTLY STATED in the paper as a single quote
   - Use 'derivation' fields when the value requires CALCULATION from multiple sources (e.g., summing numbers, looking up values from different sections)
   - NEVER use both source_quote and derivation for the same field - choose ONE

2. DERIVATION STRUCTURE (when calculation is needed):
   A derivation field must be a properly structured object with these required components:
   {
     "operation": one of ["sum", "subtraction", "division", "multiplication", "lookup", "average", "custom"],
     "operation_description": "Human-readable explanation of what was calculated",
     "sources": [
       {
         "quote": "Exact quote from paper",
         "location": "Page/section where found",
         "extracted_value": numeric_value_from_this_quote
       },
       // ... more source objects as needed
     ],
     "computation": "The actual formula, e.g., '13 + 10 = 23'"
   }

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

DO NOT provide derivation as a string - it must be a structured object as shown above."""

    user_message = "Extract the requested data using the extract_data tool:\n\n" + corpus + "\n"

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    arguments = {"messages": messages, "tools": [tool], "model": "o4-mini", "reasoning_effort": "high",
                 "tool_choice": {"type": "function", "function": {"name": tool["function"]["name"]}}}

    retry = 0
    output_dictionary = None
    while not output_dictionary and retry < retries:
        if retry > 0:
            print("Retrying...")
        chat_response = await async_client.chat.completions.create(**arguments)
        if chat_response.choices[0].message.tool_calls:
            valid_calls = use_tools(chat_response, arguments, call_functions=False)
            if valid_calls:
                for call in valid_calls:
                    if call["name"] == tool["function"]["name"]:
                        output_dictionary = call["parameters"]
                        check = await reflect_on_data_extraction(output_dictionary, corpus)
                        if check is not None:
                            if retry + 1 >= retries and not check.startswith("The value of '"):
                                print("Data extraction failed (last):", check)
                                new_message = {
                                    "role": "system",
                                    "content": "This data extraction may or may not have been completed successfully:\n\n" +
                                                str(output_dictionary) +
                                                "\n\n\nTheir is skepticism surrounding this extraction for the following reason:\n" + check +
                                                "\n\n This last skepticsm came from an analysis that did not include the corpus text - therefor it is not a final verdict and you may need to justify it further."
                                }
                                justification_message = {
                                    "role": "user",
                                    "content": "Please explain the extraction process and if you feel its appropriate make a case for why this extraction is valid despite the issues."
                                }
                                arguments["messages"] += [justification_message]
                                del arguments["tools"]
                                del arguments["tool_choice"]
                                chat_response = await async_client.chat.completions.create(**arguments)
                                if chat_response.choices[0].message.content:
                                    justification = chat_response.choices[0].message.content
                                    check = await reflect_on_data_extraction(output_dictionary, corpus, justification=justification)
                                    if check is not None:
                                        explanation = check + "\n\nLast attempt:\n\n" + str(output_dictionary)
                                        output_dictionary = None
                                    else:
                                        print("Data extraction successful at last attempt")
                            elif retry + 1 >= retries:
                                explanation = check
                            else:
                                print("Data extraction failed:", check, "Retrying...")
                                new_message = {
                                "role": "system",
                                "content": ("This data extraction was not completed successfully:\n\n" +
                                            str(output_dictionary) +
                                            "\n\n\nThis was not successful for the following reason:\n" + check +
                                            "\n\n Please try to address this issue.")
                                }
                                arguments["messages"] += [new_message]
                                output_dictionary = None                        
        else:
            print("No tool calls used")
        retry += 1
    if output_dictionary is not None:
        del output_dictionary["successfully_extracted"]
        # Remove empty discrepancy notes (keep if populated)
        if "data_discrepancy_notes" in output_dictionary and not output_dictionary["data_discrepancy_notes"]:
            del output_dictionary["data_discrepancy_notes"]
        return output_dictionary.copy()
    if explanation:
        return {"failed_extraction": explanation}
    return {}

