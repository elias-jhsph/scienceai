import logging

from .llm import MODEL_REASONING, client, get_model_for_role

logger = logging.getLogger(__name__)


def add_reasoning_to_context(context):
    # remove "status" and "time" from each message
    context = [{k: v for k, v in message.items() if k not in ["status", "time"]} for message in context]

    # remove system messages
    context = [message for message in context if message["role"] != "system"]

    # remove any tool calls at the end of the context

    for i in range(len(context) - 1, -1, -1):
        if "tool_calls" in context[i]:
            context.pop(i)
        else:
            break

    final_message = {
        "content": "Please help me consider the information so far in this chat. Is there any issues with "
        "the data? Any helpful calculations to consider? Please only respond in an additive "
        "way, no need to repeat information already provided.",
        "role": "assistant",
    }

    context.append(final_message)
    try:
        arguments = {"messages": context, "model": get_model_for_role(MODEL_REASONING)}

        chat_response = client.chat.completions.create(**arguments)

        output = chat_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        return None
    return output
