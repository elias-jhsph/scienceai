from datetime import datetime
from scienceai.analyst import Analyst
from .database_manager import DatabaseManager
from .llm import client, use_tools
import os


path_to_app = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(path_to_app, "principle_investigator_base_prompt.txt"), "r") as file:
    system_message = file.read()


# PI Module
class PrincipalInvestigator:
    def __init__(self, dbr: DatabaseManager):
        self.db = dbr
        self.analysts = []
        analysts_db = dbr.get_all_analysts()
        for analyst_dict in analysts_db:
            self.analysts.append(Analyst(dbr, analyst_dict=analyst_dict))
        chat_db = self.db.get_database_chat()
        self.tool_callables = {
            "delegate_research": self.delegate_research
        }
        self.tools = [self.delegate_research(None, None, return_tool=True)]
        self.system_message = system_message
        first_message = ("Hello, I am ScienceAI. I first need to make sure all your papers are loaded into the system "
                         "before I can help you. I will let you know when I am ready to answer your questions. "
                         "This may take a long time if you uploaded many papers.")
        second_message = "All papers have been loaded into the system."
        defaults = [first_message, second_message]
        self.db.remove_old_default_messages(defaults)
        if len(chat_db) > 0:
            last_chat = chat_db[-1]
            if last_chat["content"] == first_message:
                self.db.update_last_chat("Pending")
                self.db.ingest_papers()
                self.db.process_all_papers()
                self.db.update_last_chat("Processed")
                second = {"content": second_message, "role": "system", "status": "Pending",
                          "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')}
                self.db.add_chat(second)
                self.db.update_last_chat("Processed")
            elif last_chat["content"] == second_message:
                self.db.update_last_chat("Processed")
            else:
                if last_chat["status"] == "Pending":
                    if last_chat.get("tool_calls"):
                        self.finish_tool_calls(last_chat)
                    elif last_chat["role"] == "user":
                        self.process_message(last_chat["content"], last_chat["role"], last_chat["status"], last_chat["time"],
                                             store_message=False)
        else:
            first = {"content": first_message, "role": "system", "status": "Pending",
                     "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')}
            self.db.add_chat(first)
            self.db.ingest_papers()
            self.db.process_all_papers()
            self.db.update_last_chat("Processed")
            second = {"content": second_message, "role": "system", "status": "Pending",
                      "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')}
            self.db.add_chat(second)
            self.db.update_last_chat("Processed")
        self.db.update_last_chat("Processed")

    def delegate_research(self, name, question, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "delegate_research",
                    "description": "Delegates a specific research question pertaining to the "
                                   "uploaded database of research papers to a new Analyst Agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Assign a meaningful name to each Analyst Agent that reflects their "
                                               "specific task or research focus in title case."
                            },
                            "question": {
                                "type": "string",
                                "description": "The specific sub-research question to be answered by the analyst. Make "
                                               "sure to include any relevant context or details that may be helpful to "
                                               "the analyst in performing their data collections and analysis, as well "
                                               "as specific forms and types of data evidence that may be required to "
                                               "support their conclusions when answering the question."
                            }
                        }
                    },
                    "required": ["name", "question"],
                }
            }

        new_analyst = None
        if question is None:
            raise Exception("ERROR: Please provide a question for the analyst to research.")
        if len(question) < 10:
            raise Exception("ERROR: Please provide a more detailed question for the analyst to research.")
        if name is None:
            raise Exception("ERROR: Please provide a name for the new analyst.")
        if len(name) < 3:
            raise Exception("ERROR: Please provide a longer name for the new analyst.")
        if len(name) > 50:
            raise Exception("ERROR: Please provide a shorter name for the new analyst.")
        if len(self.analysts) > 0:
            for analyst in self.analysts:
                if analyst.name == name and analyst.goal == question:
                    if analyst.answer is None:
                        new_analyst = analyst
                    else:
                        return ("Response from " + analyst.name + ":\n" + analyst.answer +
                                "\nEvidence provided by " + analyst.name + ":\n" + analyst.evidence)
        if not new_analyst:
            new_analyst = Analyst(self.db, name=name, goal=question)
            self.analysts.append(new_analyst)
        new_analyst.pursue_goal()
        return ("Response from " + name + ":\n" + new_analyst.answer +
                "\nEvidence provided by " + name + ":\n" + new_analyst.evidence)

    def tool_callback(self, response, function_name=None):
        self.messages.append(response)
        self.db.update_last_chat("Processed")
        self.db.add_chat(response["content"], response["role"],
                         "Pending", datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z'),
                         function_name=function_name)

    @staticmethod
    def tool_error_callback(response):
        from pprint import pprint as pp
        pp(response)

    def process_message(self, content, role, status, time, store_message=True):
        if status != "Pending":
            self.db.update_last_chat("Processed")
            return
        if role != "user":
            raise ValueError("Only new user messages can be processed")
        if store_message:
            chat_message = {"content": content, "role": role, "status": status, "time": time}
            self.db.add_chat(chat_message)
        called_tools = True
        while called_tools:
            called_tools = False
            temp_messages = [{"content": self.system_message, "role": "system"}] + self.db.get_database_chat()
            arguments = {"messages": temp_messages, "model": "gpt-4o", "tools": self.tools, "temperature": 0.2}
            chat_response = client.chat.completions.create(**arguments)
            if chat_response.choices[0].message.tool_calls:
                called_tools = True
            if chat_response.choices[0].message.content and not called_tools:
                self.db.update_last_chat("Processed")
                chat_message = {"content": chat_response.choices[0].message.content, "role": "assistant",
                                "status": "Pending", "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')}
                self.db.add_chat(chat_message)
            if called_tools:
                call_new_history = use_tools(chat_response, arguments, function_dict=self.tool_callables, pre_tool_call=True)
                for call in call_new_history:
                    if call["role"] == "assistant":
                        self.db.update_last_chat("Processed")
                        call["status"] = "Pending"
                        call["time"] = datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
                        if not call["content"]:
                            call["content"] = "Working on that now..."
                        self.db.add_chat(call)
                new_history = use_tools(chat_response, arguments, function_dict=self.tool_callables)
                for call in new_history:
                    if call["role"] != "assistant":
                        self.db.update_last_chat("Processed")
                        call["status"] = "Pending"
                        call["time"] = datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
                        self.db.add_chat(call)
        self.db.update_last_chat("Processed")

    def finish_tool_calls(self, last_chat):
        new_history = use_tools(last_chat, {"messages": self.db.get_database_chat(), "model": "gpt-4o",
                                            "tools": self.tools, "temperature": 0.2}, function_dict=self.tool_callables)
        for call in new_history:
            if call["role"] != "assistant":
                self.db.update_last_chat("Processed")
                call["status"] = "Pending"
                call["time"] = datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
                self.db.add_chat(call)
        self.db.update_last_chat("Processed")



