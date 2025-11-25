import time
from datetime import datetime

from .llm import client, use_tools_sync as use_tools
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
                      "determine if the goal has been achieved or the question has been answered. "
                      "NOTE: Paper IDs (short alphanumeric identifiers like '1e482c3c3a') are valid and acceptable "
                      "for identifying papers—title extraction is optional, not required.")
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

    arguments = {"messages": messages, "model": "o4-mini", 'reasoning_effort': 'medium'}

    chat_response = client.chat.completions.create(**arguments)

    thoughts = chat_response.choices[0].message.content

    messages.append({"role": "assistant", "content": thoughts})

    tools = [
        {
            "type": "function",
            "function": {
                "strict": True,
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
                    "required": ["resolved"],
                    "additionalProperties": False
                }
            }
        }
    ]

    arguments = {"messages": messages, "model": "o4-mini", 'reasoning_effort': 'medium', "tools": tools,
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
    def __init__(self, db: DatabaseManager, analyst_dict={}, name="", goal="", attempts=5, require_file_output=False):
        self.db = db
        self.attempts = attempts
        self.require_file_output = require_file_output
        if analyst_dict and name and goal:
            raise ValueError("Can not provide both analyst_dict and name and goal.")
        if "name" in analyst_dict and "goal" in analyst_dict:
            self.name = analyst_dict["name"]
            self.goal = analyst_dict["goal"]
            self.answer = analyst_dict.get("answer", None)
            self.evidence = analyst_dict.get("evidence", None)
            self.require_file_output = analyst_dict.get("require_file_output", False)
        if name and goal:
            self.name = name
            self.goal = goal
            self.answer = None
            self.evidence = None
        try:
            metadata = self.db.get_analyst_metadata(self.name)
            # Load require_file_output from metadata if it exists
            if not analyst_dict:
                self.require_file_output = metadata.get("require_file_output", False)
        except ValueError:
            self.db.create_analyst(name, goal, other={"goal_achieved": False, "require_file_output": require_file_output})
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
            self.complete_goal_by_answering_question_with_evidence_schema()
        ]
        self.follow_up_answer = None
        self.follow_up_evidence = None
        messages = self.db.get_analyst_context(self.name)
        answer_attempts = [message for message in messages if message["role"] == "tool" and message["name"] == "complete_goal_by_answering_question_with_evidence"]
        self.answer_attempts = len(answer_attempts)
        
        # Build system message with file output requirements
        file_output_instruction = ""
        if self.require_file_output:
            file_output_instruction = """\n\nCRITICAL REQUIREMENT: You MUST provide your answer with data collection files. 
When completing your goal, you are REQUIRED to use the 'data_collection_names' parameter with the names of the data collections you created.
Do NOT provide the answer without attaching the file(s). The user specifically requested downloadable file outputs for this analysis.
"""
        else:
            file_output_instruction = """\n\nIMPORTANT FOR LARGE DATASETS: If the user requests large datasets or file outputs (e.g., sample sizes from 100+ papers), use the 'data_collection_names' parameter:
- Provide a list of your data collection names (e.g., ["SampleSizeExtraction", "SubgroupAnalysis"])
- Give a concise text 'answer' summarizing your findings
- Do NOT repeat the data in the 'evidence' field—the system will automatically inject the file contents and generate download links
- Example: If you created "SampleSizeExtraction", pass data_collection_names=["SampleSizeExtraction"] and explain what the file contains in your answer
"""
        
        self.system_message = analyst_system + file_output_instruction

    def get_context(self):
        return self.db.get_analyst_context(self.name)

    def get_all_papers(self, all=True, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "strict": True,
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
                        "required": ["all"],
                        "additionalProperties": False
                    }
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
                    "strict": True,
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
                        "required": ["name", "paper_ids"],
                        "additionalProperties": False
                    }
                }
            }
        if name.lower().replace(" ", "") == "allpapers":
            return "List named 'ALL PAPERS' already exists by default and can be used to reference all papers."
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
                    "strict": True,
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
                        "additionalProperties": False,
                        "required": ["name"]
                    }
                }
            }
        if name == "ALL PAPERS":
            name = None
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
                    "strict": True,
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
                                "description": "The name of the list of papers to collect data from. "
                                               "or 'ALL PAPERS' to collect data from all papers."
                            }
                        },
                        "additionalProperties": False,
                        "required": ["collection_name", "collection_goal", "target_list"]
                    }
                }
            }

        if target_list:
            try:
                if target_list == "ALL PAPERS":
                    target_list = None
                papers = self.db.get_all_papers_data(analyst=self.name, named_list=target_list)
            except ValueError:
                raise ValueError("List not found.")
        else:
            papers = self.db.get_all_papers_data()

        summaries = ""
        for paper in papers:
            summaries += paper["metadata"]["title"][0] + "\n\nSummary: " + paper["summary"] + "\n\n\n"
        schema = generate_schema(summaries, goal=collection_name+" - "+collection_goal)
        if not schema:
            raise ValueError("Could not generate schema for data collection, be more specific in your goal.")
        tool = schema_to_tool(schema)
        print("Tool:", tool)
        results = {}
        tracker = self.db.add_analyst_tool_tracker(self.name, collection_name, datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
        time.sleep(3)
        
        # Create async tasks for parallel extraction
        import asyncio
        import concurrent.futures
        
        print(f"Starting parallel extraction for {len(papers)} papers...")
        
        async def extract_from_paper(paper):
            """Extract data from a single paper"""
            paper_id = paper["database"]["paper_id"]
            short_paper_id = paper_id[:10]
            print(f"*** Extracting from Paper: {short_paper_id}")
            
            # Run async extract_data
            result = await extract_data(tool, paper["cleaned_text"])
            
            # Update tracker immediately after extraction
            self.db.update_analyst_tool_tracker(tracker, paper_id, result)
            
            print(f"*** Completed extraction for Paper: {short_paper_id}")
            return short_paper_id, paper_id, result
        
        async def run_all_extractions():
            """Run all extractions concurrently"""
            tasks = [extract_from_paper(paper) for paper in papers]
            results = await asyncio.gather(*tasks)
            print(f"All {len(results)} extractions completed!")
            return results
        
        # Run in a new thread with its own event loop
        def run_in_thread():
            """Create new event loop and run extractions"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(run_all_extractions())
            finally:
                loop.close()
        
        # Execute in thread and wait for results
        print("Waiting for all extractions to complete...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            extraction_results = future.result()
        
        print(f"Received results for {len(extraction_results)} papers, building output...")
        
        # Build results dict (tracker already updated during extraction)
        for short_paper_id, paper_id, result in extraction_results:
            short_id[short_paper_id] = paper_id
            results[short_paper_id] = result
        
        self.db.convert_analyst_tool_tracker(self.name, collection_name)
        print(f"Data collection complete, returning results.")
        return results

    def complete_goal_by_answering_question_with_evidence_schema(self):
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
                        },
                        "data_collection_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of data collection names to attach as downloadable files. "
                                           "The system will automatically inject file contents into evidence and generate download links."
                        }
                    },
                    "required": ["answer", "evidence"]
                }
            }
        }

    def complete_goal_by_answering_question_with_evidence(self, answer="", evidence="", data_collection_names=None):
        import pandas as pd
        import os
        
        # Validate require_file_output enforcement
        if self.require_file_output and (not data_collection_names or len(data_collection_names) == 0):
            self.answer_attempts += 1
            return ("Answer not valid. You are REQUIRED to provide data collection files for this analysis. "
                    "Please create a data collection using create_data_collection_request, then reference it "
                    "using the data_collection_names parameter. Do not attempt to complete without attaching files.")
        
        # Handle data_collection_names if provided
        if data_collection_names:
            download_links = []
            injected_evidence = evidence + "\n\n## Attached Data Collections\n\n"
            
            for collection_name in data_collection_names:
                try:
                    # Generate CSV and get path
                    csv_path = self.db.convert_analyst_tool_tracker(self.name, collection_name)
                    
                    # Read CSV for injection
                    df = pd.read_csv(csv_path)
                    row_count = len(df)
                    col_count = len(df.columns)
                    
                    # Inject full or truncated CSV into evidence
                    injected_evidence += f"### {collection_name}\n\n"
                    injected_evidence += f"**Shape**: {row_count} rows, {col_count} columns\n\n"
                    
                    if row_count < 500:
                        # Inject full CSV as markdown table
                        injected_evidence += df.to_markdown(index=False) + "\n\n"
                    else:
                        # Inject summary only
                        injected_evidence += f"*Note: Dataset truncated (>{row_count} rows). Full data available in download.*\n\n"
                        injected_evidence += df.head(10).to_markdown(index=False) + "\n\n"
                    
                    # Generate download link - use basename for relative path
                    csv_filename = os.path.basename(csv_path)
                    html_snippet = f"<div class=\"icon-container-box-image\"><div class=\"icon-container-csv-image\"></div><div class=\"button-icon-menu\"><button class=\"icon-button\" onclick=\"viewCSV('download/{csv_path}')\"><i class=\"fa fa-eye\"></i></button><a href=\"/download/{csv_path}?attached=T\" class=\"icon-button\"><i class=\"fas fa-download\"></i></a></div></div>"
                    download_links.append(html_snippet)
                    
                except Exception as e:
                    # Error handling - return early and skip checker
                    error_message = f"Answer not valid. Collection '{collection_name}' does not exist or failed to generate. Reason: {str(e)}"
                    self.answer_attempts += 1
                    return error_message
            
            # Update evidence with injected data
            evidence = injected_evidence
            
            # Add download links to answer
            answer = answer + "\n\n**Data collections attached**: " + ", ".join(data_collection_names) + "\n\n" + "\n".join(download_links)
        
        # Run the standard goal checker (backward compatible)
        thoughts = reflect_on_evidence(self.goal, answer, evidence)
        self.answer_attempts += 1
        if thoughts == "":
            self.answer = answer
            self.evidence = evidence
            return "Goal achieved:\n" + answer + "\n\nEvidence:\n" + evidence
        return ("Goal not achieved. Here are some thoughts on why: " + thoughts + "\n\n" +
                "Consider refining your data collection request with this in mind, and trying again.")

    def answer_followup_question_schema(self):
        return {
            "type": "function",
            "function": {
                "strict": True,
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
                    "additionalProperties": False,
                    "required": ["answer", "evidence"]
                }
            }
        }

    def answer_followup_question(self, answer="", evidence=""):
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
                    "content": self.system_message
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
            arguments = {"messages": messages, "model": "o4-mini", 'reasoning_effort': 'medium', "tools": self.tools}
            chat_response = client.chat.completions.create(**arguments)
            new_history = use_tools(chat_response, arguments, function_dict=self.tool_callables)
            for call in new_history:
                self.db.add_analyst_context(self.name, call)
                messages.append(call)
            last_three_messages_exist_and_are_identical = len(messages) > 3 and messages[-1]["content"] == messages[-2]["content"] == messages[-3]["content"]
            last_three_messages_no_tools = messages[-1].get("tool_calls",None) is None and messages[-2].get("tool_calls",None) is None and messages[-3].get("tool_calls",None) is None
            if (self.answer_attempts > self.attempts and not self.answer or
                    len(messages) > 100 or last_three_messages_exist_and_are_identical or last_three_messages_no_tools):
                self.answer = ("The analyst has not been able to answer the question in the allotted attempts. "
                               "Refine the goal and make sure it is specific and longer to help the next analyst "
                               "succeed where this one failed. You should remind it that when it creates its "
                               "data collection requests it should include details on how to avoid those pitfalls.")
                reasons = [message["content"] for message in messages if message["role"] == "tool" and message["name"] == "complete_goal_by_answering_question_with_evidence"]
                self.evidence = ("Here are the reasons the analyst failed to reach its goal "
                                 "after ") + str(self.attempts) + " attempts:" + "\n\n" + "\n\n".join(reasons)
            last_two_messages_no_tools = messages[-1].get("tool_calls", None) is None and messages[-2].get("tool_calls",None) is None
            if last_two_messages_no_tools:
                messages += [{
                    "role": "system",
                    "content": "Make sure to use tool calls to attempt to collect data or complete your goal, do not just talk to yourself."
                }]
        self.db.add_analyst_metadata(self.name,
                                     {"goal_achieved": True, "answer": self.answer, "evidence": self.evidence})
