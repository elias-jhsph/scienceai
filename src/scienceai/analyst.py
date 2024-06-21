from .llm import client, use_tools
from .database_manager import DatabaseManager
from .data_extractor import generate_schema, extract_data, schema_to_tool
import os

short_id = {}

path_to_app = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(path_to_app, "analyst_base_prompt.txt"), "r") as f:
    analyst_system = f.read()


def reflect_on_evidence(goal, answer, evidence, retries=3):
    system_message = ("The analyst has answered the following question / goal with evidence. "
                      "You are a thoughtful Researcher, evaluate the evidence and "
                      "determine if the goal has been achieved or the question has been answered.")
    user_message = f"My goal/question: {goal}\n\nMy answer is:\n{answer}\n\nMy evidence:\n{evidence}."

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

    arguments = {"messages": messages, "model": "gpt-4o", }

    chat_response = client.chat.completions.create(**arguments)

    thoughts = chat_response.choices[0].message.content

    messages.append({"role": "assistant", "content": thoughts})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "check_completed_goal",
                "description": "Checks if the goal has been completed or the question has "
                               "been answered and the evidence is sufficient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "resolved": {
                            "type": "boolean",
                            "description": "Whether the goal has been completed or the question has been answered."
                        }
                    },
                },
                "required": ["resolved"]
            }
        }
    ]

    arguments = {"messages": messages, "model": "gpt-4o", "tools": tools,
                 "tool_choice": {"type": "function", "function": {"name": "check_completed_goal"}}}

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
                    if call["name"] == "check_completed_goal":
                        if not call["parameters"]["resolved"]:
                            return thoughts
                        else:
                            return ""
        retry += 1
    return thoughts


# Analyst Module
class Analyst:
    def __init__(self, db: DatabaseManager, analyst_dict={}, name="", goal="", attempts=5):
        self.db = db
        self.attempts = attempts
        if analyst_dict and name and goal:
            raise ValueError("Can not provide both analyst_dict and name and goal.")
        if "name" in analyst_dict and "goal" in analyst_dict:
            self.name = analyst_dict["name"]
            self.goal = analyst_dict["goal"]
            self.answer = analyst_dict.get("answer", None)
            self.evidence = analyst_dict.get("evidence", None)
        if name and goal:
            self.name = name
            self.goal = goal
            self.answer = None
            self.evidence = None
        try:
            self.db.get_analyst_metadata(self.name)
        except ValueError:
            self.db.create_analyst(name, goal, other={"goal_achieved": False})
        self.get_all_papers()
        self.tool_callables = {
            "get_all_papers": self.get_all_papers,
            "create_named_paper_list": self.create_named_paper_list,
            "get_named_paper_list": self.get_named_paper_list,
            "create_data_collection_request": self.create_data_collection_request,
            "complete_goal_by_answering_question_with_evidence": self.complete_goal_by_answering_question_with_evidence
        }
        self.tools = [
            self.get_all_papers(return_tool=True),
            self.create_named_paper_list(None, None, return_tool=True),
            self.get_named_paper_list(None, return_tool=True),
            self.create_data_collection_request(None, None, return_tool=True),
            self.complete_goal_by_answering_question_with_evidence(None, None, return_tool=True)
        ]
        self.follow_up_answer = None
        self.follow_up_evidence = None
        messages = self.db.get_analyst_context(self.name)
        answer_attempts = [message for message in messages if message["role"] == "tool" and message["name"] == "complete_goal_by_answering_question_with_evidence"]
        self.answer_attempts = len(answer_attempts)

    def get_context(self):
        return self.db.get_analyst_context(self.name)

    def get_all_papers(self, all=True, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "get_all_papers",
                    "description": "Prints all papers in the database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "all": {
                                "type": "boolean",
                                "description": "Whether to return all papers."
                            }
                        },
                    },
                    "required": ["all"],
                }
            }
        output = {}
        papers = self.db.get_all_papers_data()
        for paper in papers:
            short_id[paper['database']['paper_id'][:10]] = paper['database']['paper_id']
            output[paper['database']['paper_id'][:10]] = paper['metadata']['title'][0]
        return output

    def create_named_paper_list(self, name="", paper_ids=[], return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "create_named_paper_list",
                    "description": "Creates a permanent list of papers (this can not me mutate later).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the list."
                            },
                            "paper_ids": {
                                "type": "array",
                                "description": "The IDs of the papers to add to the list.",
                                "items": {
                                    "type": "string",
                                    "description": "The ID of the paper."
                                }
                            }
                        },
                    },
                    "required": ["name", "paper_ids"],
                }
            }
        if self.db.get_all_papers(analyst=self.name, named_list=name):
            raise ValueError("List '" + name + "' already exists.")
        for paper_id in paper_ids:
            self.db.add_paper_to_list(short_id[paper_id], self.name, name)
        return "List named '" + name + "' created with papers: " + str(paper_ids)

    def get_named_paper_list(self, name="", return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "get_named_paper_list",
                    "description": "Gets the papers in a list.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the list."
                            }
                        },
                    },
                    "required": ["name"],
                }
            }
        papers = self.db.get_all_papers_data(analyst=self.name, named_list=name)
        output = {}
        for paper in papers:
            short_id[paper['paper_id'][:10]] = paper['database']['paper_id']
            output[paper['paper_id'][:10]] = paper['metadata']['title'][0]
        return output

    def create_data_collection_request(self, collection_name="", collection_goal="",
                                       target_list=None, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "create_data_collection_request",
                    "description": "Creates a schema for a data collection request.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "The name of the data collection request."
                            },
                            "collection_goal": {
                                "type": "string",
                                "description": "The goal of the data collection request."
                            },
                            "target_list": {
                                "type": "string",
                                "description": "The name of the list of papers to collect data from."
                            }
                        },
                    },
                    "required": ["collection_name", "collection_goal"],
                }
            }

        if target_list:
            try:
                papers = self.db.get_all_papers_data(analyst=self.name, named_list=target_list)
            except ValueError:
                raise ValueError("List not found.")
        else:
            papers = self.db.get_all_papers_data()

        summaries = ""
        for paper in papers:
            summaries += paper["metadata"]["title"][0] + "\n\nSummary: " + paper["summary"] + "\n\n\n"
        schema = generate_schema(summaries, goal=collection_name+" - "+collection_goal)
        tool = schema_to_tool(schema)

        results = {}
        tracker = self.db.add_analyst_tool_tracker(self.name, collection_name)
        for paper in papers:
            result = extract_data(tool, paper["cleaned_text"])
            self.db.update_analyst_tool_tracker(tracker, paper["database"]["paper_id"], result)
            short_id[paper["database"]["paper_id"][:10]] = paper["database"]["paper_id"]
            results[paper["database"]["paper_id"][:10]] = result
        self.db.convert_analyst_tool_tracker(self.name, collection_name)
        return results

    def complete_goal_by_answering_question_with_evidence(self, answer="", evidence="", return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "complete_goal_by_answering_question_with_evidence",
                    "description": "Completes the analyst's goal by answering a question with evidence.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "This should be a detailed answer to the research question. All "
                                               "evidence needed to support the answer should be included in the "
                                               "evidence section."
                            },
                            "evidence": {
                                "type": "string",
                                "description": "This should be specific data points or findings from the data "
                                               "collection that support your answer, DO NOT reference data you do not "
                                               "directly provide as evidence. For example, if you are asked to provide "
                                               "the top 5 genes from each paper, you should provide the list of genes "
                                               "by paper as evidence."
                            }
                        },
                    },
                    "required": ["answer", "evidence"],
                }
            }
        thoughts = reflect_on_evidence(self.goal, answer, evidence)
        self.answer_attempts += 1
        if thoughts == "":
            self.answer = answer
            self.evidence = evidence
            return "Goal achieved:\n" + answer + "\n\nEvidence:\n" + evidence
        return "Goal not achieved. Here are some thoughts on why: " + thoughts

    def answer_followup_question(self, answer="", evidence="", return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "answer_followup_question",
                    "description": "Answers a follow-up question with evidence.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "The answer to the question."
                            },
                            "evidence": {
                                "type": "string",
                                "description": "The evidence to answer the question."
                            }
                        },
                    },
                    "required": ["answer", "evidence"],
                }
            }
        thoughts = reflect_on_evidence(self.goal, answer, evidence)
        if thoughts == "":
            self.follow_up_answer = answer
            self.follow_up_evidence = evidence
            return "Question answered:\n" + answer + "\n\nEvidence:\n" + evidence
        return "Question not answered. Here are some thoughts on why: " + thoughts

    def pursue_goal(self):
        messages = self.db.get_analyst_context(self.name)
        if not messages:
            user_message = f"Here is your goal/question: {self.goal}\n\n"
            messages = [
                {
                    "role": "system",
                    "content": analyst_system
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            for message in messages:
                self.db.add_analyst_context(self.name, message)
        while not self.answer:
            messages = self.db.get_analyst_context(self.name)
            arguments = {"messages": messages, "model": "gpt-4o", "tools": self.tools, "temperature": 0.2}
            chat_response = client.chat.completions.create(**arguments)
            new_history = use_tools(chat_response, arguments, function_dict=self.tool_callables)
            for call in new_history:
                self.db.add_analyst_context(self.name, call)
                messages.append(call)
            if self.answer_attempts > self.attempts and not self.answer:
                self.answer = ("The analyst has not been able to answer the question in the allotted attempts. "
                               "Refine the goal and make sure it is specific and longer to help the next analyst "
                               "succeed where this one failed. You should remind it that when it creates its "
                               "data collection requests it should include details on how to avoid those pitfalls.")
                reasons = [message["content"] for message in messages if message["role"] == "tool" and message["name"] == "complete_goal_by_answering_question_with_evidence"]
                self.evidence = ("Here are the reasons the analyst failed to reach its goal "
                                 "after ") + str(self.attempts) + " attempts:" + "\n\n" + "\n\n".join(reasons)
        self.db.add_analyst_metadata(self.name,
                                     {"goal_achieved": True, "answer": self.answer, "evidence": self.evidence})
