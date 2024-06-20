from openai import OpenAI
import tiktoken
import json
import os
import threading
import traceback


class GenericClientWrapper:
    def __init__(self, client, stop_event):
        self._client = client
        self._stop_event = stop_event

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if self._stop_event.is_set():
            quit(0)
        if callable(attr):
            def wrapper(*args, **kwargs):
                if self._stop_event.is_set():
                    quit(0)
                try:
                    return attr(*args, **kwargs)
                except Exception as e:
                    print(f"Request failed: {e}")
                    return None
            return wrapper
        return attr


base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
target_key = os.path.join(base_key_path, "scienceai-keys.json")
if not os.path.exists(target_key):
    new_key = input("Please enter OpenAI key: ")
    if not os.path.exists(os.path.dirname(os.path.dirname(target_key))):
        os.mkdir(os.path.dirname(os.path.dirname(target_key)))
    if not os.path.exists(os.path.dirname(target_key)):
        os.mkdir(os.path.dirname(target_key))
    with open(target_key, "w") as file:
        json.dump({"openai": new_key}, file)
with open(target_key, "r") as file:
    key_list = json.load(file)
openai_key = key_list.get("openai")
if openai_key is None:
    raise Exception("Open AI key not in scienceai-keys.json")

try:
    __client = OpenAI(api_key=openai_key)


    def is_api_key_valid():
        response = __client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hello, "}],
                                                      max_tokens=1)


    is_api_key_valid()
except Exception as e:
    delete = input("Error creating OpenAI client - delete saved key? (Y/n): ")
    if delete.lower() == "y":
        os.remove(target_key)
        new_key = input("Please enter OpenAI key: ")
        with open(target_key, "w") as file:
            json.dump({"openai": new_key}, file)
        __client = OpenAI(api_key=new_key)
    else:
        raise e

enc = tiktoken.encoding_for_model("gpt-4")

stop_event = threading.Event()

client = GenericClientWrapper(__client, stop_event)


def update_stop_event(event):
    client._stop_event = event


def check_tool(function_name, parameters, tools):
    function_names = []
    for tool in tools:
        function_names.append(tool["function"]["name"])
    if function_name not in function_names:
        return {"role": "system",
                "content": "ERROR: The function name is not in the list of functions provided to you. "
                           "Please try again."}
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError:
            return {"role": "system",
                    "content": "ERROR: You're function call did not parse as valid JSON. Please try again."}
    if isinstance(parameters, dict):
        for key in parameters:
            if key not in tools[function_names.index(function_name)]["function"]["parameters"]["properties"]:
                return {"role": "system",
                        "content": "ERROR: The parameter '" + key +
                                   "' is not in the list of parameters provided to you. Please try again."}
    return None


def trim_history(history, token_limit):
    for i in range(len(history)):
        if len(enc.encode(str(history))) > token_limit:
            history.pop(1)
        else:
            return history


def use_tools(chat_response, arguments, function_dict={}, call_functions=True, pre_tool_call=False):
    """
    Use the tools specified in the tool_calls dict.
    :param chat_response:
    :type chat_response: dict
    :param arguments:
    :type arguments: dict
    :param function_dict:
    :type function_dict: dict
    :param call_functions:
    :type call_functions: bool
    :param pre_tool_call:
    :type pre_tool_call: bool
    :return: list of history items
    :rtype: list
    """
    if isinstance(chat_response, dict):
        tool_calls = chat_response['tool_calls']
        content = chat_response['content']
    else:
        tool_calls = chat_response.choices[0].message.tool_calls
        content = chat_response.choices[0].message.content
    tools = arguments['tools']
    tool_calls_list = []
    if tool_calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_calls_list.append({
                    "function": {
                        "arguments": tool_call['function']['arguments'],
                        "name": tool_call['function']['name'],
                    },
                    "id": tool_call['id'], "type": 'function'
                })
            else:
                tool_calls_list.append({'function': {
                    "arguments": tool_call.function.arguments,
                    "name": tool_call.function.name,
                }, 'id': tool_call.id, 'type': 'function'})
    if call_functions:
        if tool_calls_list:
            new_history = [{"content": content, "role": "assistant", "tool_calls": tool_calls_list}]
        else:
            new_history = [{"content": content, "role": "assistant"}]
    if pre_tool_call:
        return new_history
    tool_results = []
    tool_errors = []
    valid_calls = []
    for tool_call in tool_calls_list:
        function_name = tool_call['function']['name']
        tool_schema = None
        for schema in tools:
            if schema['function']['name'] == function_name:
                tool_schema = schema
        results, errors = use_tool(tool_call['function'], tool_call['id'], tool_schema, function_dict=function_dict, call_functions=call_functions)
        if call_functions:
            tool_results += results
            tool_errors += errors
        else:
            valid_calls += results
    if call_functions:
        return new_history + tool_results+tool_errors
    else:
        return valid_calls


def use_tool(tool_call, tool_id, tool_schema, function_dict={}, call_functions=True):
    """
    Use the tool specified in the tool_call dict.
    :param tool_call:
    :type tool_call: dict
    :param tool_id:
    :type tool_id: str
    :param tool_schema:
    :type tool_schema: dict
    :param function_dict:
    :type function_dict: dict
    :param call_functions:
    :type call_functions: bool
    :return: results and errors
    :rtype: tuple
    """
    function_name = tool_call['name']
    results = []
    errors = []
    missing_function = False
    if call_functions:
        try:
            called_function = function_dict[function_name]
        except KeyError:
            missing_function = True
            errors.append({"content": "ERROR", "role": "tool",
                           "name": function_name})
            errors.append({"content": "Only use a valid function in your "
                                      "function list.", "role": "system"})
    if not missing_function:
        try:
            arguments = json.loads(tool_call['arguments'])
            try:
                if call_functions:
                    result = called_function(**arguments)
                    results.append({"role": "tool", "name": function_name, "content": str(result), "tool_call_id": tool_id})
                else:
                    results.append({"role": "function_call", "name": function_name, "parameters": arguments})
            except Exception as e:
                error_str = ("Error calling " + function_name + " function with passed arguments " +
                             str(arguments) + " : " + traceback.format_exc() + " \n " + str(e))
                errors.append({"content": error_str, "role": "tool", "tool_call_id": tool_id,
                               "name": function_name})
        except json.decoder.JSONDecodeError:
            required_arguments = tool_schema['function']['parameters']['required']
            if tool_call['arguments'] == "":
                new_history_item = {"content": "You're function call did not "
                                               "include any arguments. Please try again with the "
                                               "correct arguments: " + str(required_arguments),
                                    "role": "system"}
            else:
                new_history_item = {"content": "You're function call did not parse as valid JSON. "
                                               "Please try again", "role": "system"}
            errors.append({"content": "ERROR", "role": "tool", "name": function_name, "tool_call_id": tool_id})
            errors.append(new_history_item)
    return results, errors
