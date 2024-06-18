import json
import os
from .llm import client, use_tools


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
                "name": "generate_example_data",
                "description": "creates example data for a given data type",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys())
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

    arguments = {"messages": messages, "tools": tools, "model": "gpt-4o",
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
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_analysis_schema",
                "description": "creates example data for a given data type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "schema": {"type": "string",
                                   "description": "A JSON string representing the schema for the analysis which should "
                                                  "be an array of objects with a 'type' key and other keys as "
                                                  "specified in the data types."},
                    },
                    "required": ["schema"]
                }
            }
        }
    ]

    system_message = ("You are an expert Data Scientist. You have been asked to generate the schema which will be used "
                      "to extract data from research papers. The schema must be creative and scientific. "
                      "You must follow the specifications provided for each data type.\n\nSpecifications for Schema:\n")

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

    user_message += ":\n\n" + corpus + "\n"

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

    arguments = {"messages": messages, "tools": tools, "model": "gpt-4o",
                 "tool_choice": {"type": "function", "function": {"name": "generate_analysis_schema"}}}

    retry = 0
    valid_calls = []
    output_dictionary = None
    while valid_calls == [] and retry < retries:
        if retry > 0:
            print("Retrying...")
        chat_response = client.chat.completions.create(**arguments)
        if chat_response.choices[0].message.tool_calls:
            valid_calls = use_tools(chat_response, arguments, call_functions=False)
            if valid_calls:
                for call in valid_calls:
                    if call["name"] == "generate_analysis_schema":
                        output_dictionary = call["parameters"]
                        if "schema" in output_dictionary:
                            try:
                                output_dictionary = json.loads(output_dictionary["schema"])
                                for data_type in output_dictionary:
                                    valid, error_message = validate_data_type_spec(data_type)
                                    if not valid:
                                        print(error_message, output_dictionary)
                                        valid_calls = []
                            except json.JSONDecodeError:
                                print("Schema is not a valid JSON string:", output_dictionary["schema"])
                                valid_calls = []
        else:
            print("No tool calls used")
        retry += 1
        if valid_calls and output_dictionary:
            return output_dictionary
    return None


def schema_to_tool(schema):

    selected_data_types = {}
    required = []

    for data_type_requested in schema:
        data_type_def = data_types[data_type_requested["type"]]["tool"]
        if data_type_def["mode"] == "prefix":
            required_keys = list(data_type_def.keys())
            required_keys.remove("mode")
            name = data_type_requested["name"].replace(" ", "_")
            for key in required_keys:
                if data_type_requested["required"]:
                    required.append(name + "_" + key)
                new_type = data_type_def[key]["type"]
                new_description = data_type_def[key]["description"]
                for spec_key in data_types[data_type_requested["type"]]["spec"]:
                    if spec_key == 'name':
                        new_description = new_description.replace("NAME", name)
                    else:
                        new_description = new_description.replace(spec_key.upper(), str(data_type_requested[spec_key]))
                selected_data_types[name + "_" + key] = {"type": new_type, "description": new_description}
            selected_data_types[name + "_successfully_extracted"] = \
                {"type": "boolean", "description": "Was the data for "+name+" successfully extracted?"}
            required.append(name + "_successfully_extracted")
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
                item_properties[key] = {"type": new_type, "description": new_description}

            selected_data_types[name] = {"type": "array", "description": new_description,
                                         "items": {"type": "object", "properties": item_properties}}

            selected_data_types[name + "_successfully_extracted"] = \
                {"type": "boolean", "description": "Was the data for "+name+" successfully extracted?"}
            required.append(name + "_successfully_extracted")

    required = list(set(required))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_data",
                "description": "extracts data from a research paper using a schema",
                "parameters": {
                    "type": "object",
                    "properties": selected_data_types,
                    "required": required
                }
            }
        }
    ]
    return tools


def reflect_on_data_extraction(extraction, corpus, retries=3):

        system_message = "You are a careful data analyst. Reflect on the data extraction process."

        user_message = ("Was this data correctly extracted from this research paper? Explain why or why not. "
                        "\n\nData Extracted:\n\n") + extraction + "\n\nResearch Paper:" + corpus

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

        arguments = {"messages": messages, "model": "gpt-4o"}

        chat_response = client.chat.completions.create(**arguments)

        messages.append({"role": "assistant", "content": chat_response.choices[0].message.content})

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "check_extracted_data",
                    "description": "Logs if the extracted data was 100% perfectly extracted.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "valid": {"type": "boolean", "description": "Whether the extracted data was "
                                                                        "100% perfectly extracted. Do not "
                                                                        "set this to true if there were any "
                                                                        "errors in the extraction or even"
                                                                        "minor mistakes."}
                        }
                    },
                    "required": ["valid"]
                }
            }
        ]

        arguments = {"messages": messages, "model": "gpt-4o", "tools": tools,
                     "tool_choice": {"type": "function", "function": {"name": "check_extracted_data"}}}

        retry = 0
        valid_calls = []
        while valid_calls == [] and retry < retries:
            if retry > 0:
                print("Retrying...")
            chat_response = client.chat.completions.create(**arguments)
            if chat_response.choices[0].message.tool_calls:
                valid_calls = use_tools(chat_response, arguments, call_functions=False)
                if valid_calls:
                    for call in valid_calls:
                        if call["name"] == "check_extracted_data":
                            return call["parameters"]["valid"]
            retry += 1
        return False


def remove_failed_data(data):
    successfull_keys = []
    for key, value in data.items():
        if key.endswith("_successfully_extracted"):
            if value:
                successfull_keys.append(key.replace("_successfully_extracted", ""))
    new_data = {}
    for key in successfull_keys:
        for k, v in data.items():
            if k.startswith(key) and not k.endswith("_successfully_extracted"):
                new_data[k] = v
    return new_data


def extract_data(tools, corpus, retries=5):

    system_message = "You are an careful data analyst. Dutifully find the data in the provided research paper."

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

    arguments = {"messages": messages, "tools": tools, "model": "gpt-4o",
                 "tool_choice": {"type": "function", "function": {"name": "extract_data"}}}

    retry = 0
    valid_calls = []
    output_dictionary = None
    while valid_calls == [] and retry < retries:
        if retry > 0:
            print("Retrying...")
        chat_response = client.chat.completions.create(**arguments)
        if chat_response.choices[0].message.tool_calls:
            valid_calls = use_tools(chat_response, arguments, call_functions=False)
            if valid_calls:
                for call in valid_calls:
                    if call["name"] == "extract_data":
                        output_dictionary = call["parameters"]
                        output_dictionary = remove_failed_data(output_dictionary)
                        check = reflect_on_data_extraction(str(output_dictionary), corpus)
                        if not check:
                            print("Data extraction failed. Retrying...")
                            valid_calls = []
        else:
            print("No tool calls used")
        retry += 1
        if output_dictionary:
            return output_dictionary
    return None

