from openai import AsyncOpenAI, OpenAI
import tiktoken
import json
import os
import asyncio
import traceback

# Load API Key
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

# Async client for ingestion pipeline
async_client = AsyncOpenAI(api_key=openai_key)

# Sync client for agents and data extraction
client = OpenAI(api_key=openai_key)

enc = tiktoken.encoding_for_model("gpt-4")

STOP_EVENT = None

def update_stop_event(stop_event):
    global STOP_EVENT
    STOP_EVENT = stop_event


def trim_history(history, token_limit):
    for i in range(len(history)):
        if len(enc.encode(str(history))) > token_limit:
            history.pop(1)
        else:
            return history


async def use_tools(chat_response, arguments, function_dict={}, call_functions=True, pre_tool_call=False):
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
    
    tools = arguments.get('tools', [])
    tool_calls_list = []
    
    if tool_calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_calls_list.append({
                    "function": {
                        "strict": True,
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
    
    # If we are just extracting parameters (call_functions=False), we can do it synchronously or async, 
    # but since we are in an async function, we'll just return the list.
    if not call_functions:
        valid_calls = []
        for tool_call in tool_calls_list:
            function_name = tool_call['function']['name']
            try:
                arguments = json.loads(tool_call['function']['arguments'])
                valid_calls.append({"name": function_name, "parameters": arguments})
            except json.JSONDecodeError:
                pass # Skip invalid JSON
        return valid_calls

    # Execute tools concurrently
    tasks = []
    for tool_call in tool_calls_list:
        function_name = tool_call['function']['name']
        tool_schema = next((t for t in tools if t['function']['name'] == function_name), None)
        tasks.append(use_tool(tool_call['function'], tool_call['id'], tool_schema, function_dict=function_dict))
    
    results_and_errors = await asyncio.gather(*tasks)
    
    tool_results = []
    tool_errors = []
    
    for res, err in results_and_errors:
        tool_results.extend(res)
        tool_errors.extend(err)
        
    return new_history + tool_results + tool_errors


async def use_tool(tool_call, tool_id, tool_schema, function_dict={}):
    """
    Use the tool specified in the tool_call dict.
    """
    function_name = tool_call['name']
    results = []
    errors = []
    
    if function_name not in function_dict:
        errors.append({"content": "ERROR", "role": "tool", "name": function_name, "tool_call_id": tool_id})
        errors.append({"content": "Only use a valid function in your function list.", "role": "system"})
        return results, errors

    called_function = function_dict[function_name]
    
    try:
        arguments = json.loads(tool_call['arguments'])
        try:
            # Support both async and sync functions
            if asyncio.iscoroutinefunction(called_function):
                result = await called_function(**arguments)
            else:
                result = called_function(**arguments)
                
            results.append({"role": "tool", "name": function_name, "content": str(result), "tool_call_id": tool_id})
        except Exception as e:
            error_str = ("Error calling " + function_name + " function with passed arguments " +
                         str(arguments) + " : " + traceback.format_exc() + " \n " + str(e))
            errors.append({"content": error_str, "role": "tool", "tool_call_id": tool_id, "name": function_name})
            
    except json.decoder.JSONDecodeError:
        required_arguments = tool_schema['function']['parameters']['required'] if tool_schema else "unknown"
        if tool_call['arguments'] == "":
            new_history_item = {"content": "Your function call did not include any arguments. Please try again with the correct arguments: " + str(required_arguments), "role": "system"}
        else:
            new_history_item = {"content": "Your function call did not parse as valid JSON. Please try again", "role": "system"}
        errors.append({"content": "ERROR", "role": "tool", "name": function_name, "tool_call_id": tool_id})
        errors.append(new_history_item)
        
    return results, errors
def use_tools_sync(chat_response, arguments, function_dict={}, call_functions=True, pre_tool_call=False):
    """
    Synchronous version of use_tools for agent use.
    """
    if isinstance(chat_response, dict):
        tool_calls = chat_response['tool_calls']
        content = chat_response['content']
    else:
        tool_calls = chat_response.choices[0].message.tool_calls
        content = chat_response.choices[0].message.content
    
    tools = arguments.get('tools', [])
    tool_calls_list = []
    
    if tool_calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_calls_list.append({
                    "function": {
                        "strict": True,
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
    
    if not call_functions:
        valid_calls = []
        for tool_call in tool_calls_list:
            function_name = tool_call['function']['name']
            try:
                arguments_parsed = json.loads(tool_call['function']['arguments'])
                valid_calls.append({"name": function_name, "parameters": arguments_parsed})
            except json.JSONDecodeError:
                pass
        return valid_calls

    # Execute tools synchronously
    results_and_errors = []
    for tool_call in tool_calls_list:
        function_name = tool_call['function']['name']
        tool_schema = next((t for t in tools if t['function']['name'] == function_name), None)
        results_and_errors.append(use_tool_sync(tool_call['function'], tool_call['id'], tool_schema, function_dict=function_dict))
    
    tool_results = []
    tool_errors = []
    
    for res, err in results_and_errors:
        tool_results.extend(res)
        tool_errors.extend(err)
        
    return new_history + tool_results + tool_errors


def use_tool_sync(tool_call, tool_id, tool_schema, function_dict={}):
    """
    Synchronous version of use_tool for agent use.
    """
    function_name = tool_call['name']
    results = []
    errors = []
    
    if function_name not in function_dict:
        errors.append({"content": "ERROR", "role": "tool", "name": function_name, "tool_call_id": tool_id})
        errors.append({"content": "Only use a valid function in your function list.", "role": "system"})
        return results, errors

    called_function = function_dict[function_name]
    
    try:
        arguments = json.loads(tool_call['arguments'])
        try:
            # Call function synchronously
            result = called_function(**arguments)
            results.append({"role": "tool", "name": function_name, "content": str(result), "tool_call_id": tool_id})
        except Exception as e:
            error_str = ("Error calling " + function_name + " function with passed arguments " +
                         str(arguments) + " : " + traceback.format_exc() + " \n " + str(e))
            errors.append({"content": error_str, "role": "tool", "tool_call_id": tool_id, "name": function_name})
            
    except json.decoder.JSONDecodeError:
        required_arguments = tool_schema['function']['parameters']['required'] if tool_schema else "unknown"
        if tool_call['arguments'] == "":
            new_history_item = {"content": "Your function call did not include any arguments. Please try again with the correct arguments: " + str(required_arguments), "role": "system"}
        else:
            new_history_item = {"content": "Your function call did not parse as valid JSON. Please try again", "role": "system"}
        errors.append({"content": "ERROR", "role": "tool", "name": function_name, "tool_call_id": tool_id})
        errors.append(new_history_item)
        
    return results, errors
