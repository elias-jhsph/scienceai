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


def process_metadata_field(metadata, field_name):
    """
    Process a single metadata field into a human-readable format.
    
    Args:
        metadata: The full metadata dict from Crossref
        field_name: The requested field name (e.g., 'authors', 'journal', 'year')
    
    Returns:
        Processed field value or None if not available
    """
    if not metadata:
        return None
    
    # Handle each field type
    if field_name == "authors":
        authors = metadata.get("author", [])
        if not authors:
            return "Not available"
        
        # Format authors
        formatted = []
        for author in authors:
            given = author.get("given", "")
            family = author.get("family", "")
            if given and family:
                formatted.append(f"{given} {family}")
            elif family:
                formatted.append(family)
        
        if not formatted:
            return "Not available"
        
        # Use et al. for long author lists
        if len(formatted) <= 5:
            return ", ".join(formatted)
        else:
            return ", ".join(formatted[:3]) + f", et al. ({len(formatted)} total)"
    
    elif field_name == "journal":
        container = metadata.get("container-title", [])
        return container[0] if container else "Not available"
    
    elif field_name == "year":
        # Try published first, then issued
        published = metadata.get("published", {})
        if published and "date-parts" in published and published["date-parts"]:
            return published["date-parts"][0][0] if published["date-parts"][0] else None
        
        issued = metadata.get("issued", {})
        if issued and "date-parts" in issued and issued["date-parts"]:
            return issued["date-parts"][0][0] if issued["date-parts"][0] else None
        
        return None
    
    elif field_name == "title":
        title = metadata.get("title", [])
        return title[0] if title else "Not available"
    
    elif field_name == "DOI":
        return metadata.get("DOI", "Not available")
    
    elif field_name == "citation_count":
        return metadata.get("is-referenced-by-count", 0)
    
    elif field_name == "publication_date":
        # Format as "Month Year" or just "Year"
        published = metadata.get("published", {})
        if not published:
            published = metadata.get("issued", {})
        
        if published and "date-parts" in published and published["date-parts"]:
            parts = published["date-parts"][0]
            if not parts:
                return "Not available"
            
            year = parts[0] if len(parts) > 0 else None
            month = parts[1] if len(parts) > 1 else None
            
            if year and month:
                month_names = ["January", "February", "March", "April", "May", "June",
                              "July", "August", "September", "October", "November", "December"]
                month_name = month_names[month - 1] if 1 <= month <= 12 else str(month)
                return f"{month_name} {year}"
            elif year:
                return str(year)
        
        return "Not available"
    
    elif field_name == "volume":
        return metadata.get("volume", "Not available")
    
    elif field_name == "issue":
        return metadata.get("issue", "Not available")
    
    elif field_name == "pages":
        return metadata.get("page", "Not available")
    
    elif field_name == "publisher":
        return metadata.get("publisher", "Not available")
    
    elif field_name == "URL":
        return metadata.get("URL", "Not available")
    
    elif field_name == "type":
        return metadata.get("type", "Not available")
    
    elif field_name == "ISSN":
        issn = metadata.get("ISSN", [])
        return issn[0] if issn else "Not available"
    
    elif field_name == "language":
        return metadata.get("language", "Not available")
    
    elif field_name == "reference_count":
        # Return count only, not the full reference list
        return metadata.get("reference-count", 0)
    
    # If field not recognized, return None
    return None


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
            "get_paper_metadata": self.get_paper_metadata,
            "create_data_collection_request": self.create_data_collection_request,
            "complete_goal_by_answering_question_with_evidence": self.complete_goal_by_answering_question_with_evidence
        }
        self.tools = [
            self.get_all_papers(return_tool=True),
            self.create_named_paper_list(None, None, return_tool=True),
            self.get_named_paper_list(None, return_tool=True),
            self.get_paper_metadata(return_tool=True),
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
            file_output_instruction = """

CRITICAL REQUIREMENT: You MUST provide downloadable file outputs.

**STEP 1: Check if metadata can satisfy the request**
First, check if ALL requested fields are available in metadata:
- authors, journal, year, title, DOI, citation_count, publication_date, volume, issue, pages, publisher, URL, type, ISSN, language, reference_count

If YES (e.g., publication years, author lists, journal names):
1. Use get_paper_metadata() to retrieve the data (100x faster!)
2. **CRITICAL**: Add `collection_name="YourCollectionName"` to the call (e.g., collection_name="PublicationYears")
3. This automatically generates the CSV file you need
4. When completing, use data_collection_names=["YourCollectionName"]

If NO (e.g., sample sizes, methods, results):
1. Use create_data_collection_request() for full-text extraction

**Why this matters:** Extracting publication years from paper content is SLOW and ERROR-PRONE. The metadata already has this information in structured form. Always check metadata first!

Do NOT complete without using data_collection_names parameter to attach files.
"""
        else:
            file_output_instruction = """

IMPORTANT FOR LARGE DATASETS: If the user requests large datasets or file outputs (e.g., sample sizes from 100+ papers), use the 'data_collection_names' parameter:
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

    def get_paper_metadata(self, paper_ids=None, metadata_fields=None, target_list=None, collection_name=None, return_tool=False):
        """
        Get specific metadata fields for papers.
        
        Args:
            paper_ids: List of short paper IDs to query (takes priority over target_list)
            metadata_fields: List of field names to retrieve. If empty, returns default essential fields.
            target_list: Name of a paper list or "ALL PAPERS" (used if paper_ids is empty)
            collection_name: Optional name to save results as a data collection for CSV export
        
        Returns:
            Dictionary mapping short paper IDs to metadata dictionaries
        """
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "strict": False,
                    "name": "get_paper_metadata",
                    "description": "Retrieve bibliographic metadata for papers (100x faster than full-text extraction). "
                                   "AVAILABLE FIELDS: authors, journal, year, title, DOI, citation_count, publication_date, "
                                   "volume, issue, pages, publisher, URL, type, ISSN, language, reference_count. "
                                   "USE THIS for: publication years, author names, journal names, DOIs, citation counts, dates. "
                                   "Query specific papers by ID, a named list, or all papers (default).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paper_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of specific paper IDs (short form) to query. If provided, this takes priority over target_list. Leave empty to use target_list instead."
                            },
                            "metadata_fields": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "authors", "journal", "year", "title", "DOI", 
                                        "citation_count", "publication_date", "volume", 
                                        "issue", "pages", "publisher", "URL", "type",
                                        "ISSN", "language", "reference_count"
                                    ]
                                },
                                "description": "Metadata fields to retrieve. Choose from: 'authors' (author names), 'journal' (venue name), "
                                               "'year' (publication year), 'title' (paper title), 'DOI', 'citation_count', "
                                               "'publication_date' (full date), 'volume', 'issue', 'pages', 'publisher', 'URL', "
                                               "'type' (article/conference), 'ISSN', 'language', 'reference_count'. "
                                               "Leave empty for defaults (authors, journal, year, DOI, citation_count)."
                            },
                            "target_list": {
                                "type": "string",
                                "description": "Optional name of a paper list to query, or 'ALL PAPERS' for all papers. Only used if paper_ids is empty. Defaults to 'ALL PAPERS' if neither paper_ids nor target_list is provided."
                            },
                            "collection_name": {
                                "type": "string",
                                "description": "OPTIONAL: Provide a name (e.g., 'PublicationYears') to automatically save these results as a data collection. "
                                               "REQUIRED if require_file_output=True. This generates the CSV file needed for your final answer."
                            }
                        },
                        "additionalProperties": False
                    }
                }
            }
        
        # Default fields if none specified
        if not metadata_fields:
            metadata_fields = ["authors", "journal", "year", "DOI", "citation_count"]
        
        # Determine which papers to query
        papers_to_query = []
        
        if paper_ids:
            # Use specific paper IDs (convert to full paper data)
            for short_paper_id in paper_ids:
                full_paper_id = short_id.get(short_paper_id)
                if full_paper_id:
                    papers_to_query.append({
                        "short_id": short_paper_id,
                        "full_id": full_paper_id
                    })
        else:
            # Use target_list or default to all papers
            if target_list == "ALL PAPERS" or target_list is None:
                target_list_name = None
            else:
                target_list_name = target_list
            
            try:
                papers_data = self.db.get_all_papers_data(analyst=self.name, named_list=target_list_name)
            except ValueError:
                return {"error": f"List '{target_list}' not found."}
            
            for paper in papers_data:
                full_paper_id = paper['database']['paper_id']
                short_paper_id = full_paper_id[:10]
                short_id[short_paper_id] = full_paper_id
                papers_to_query.append({
                    "short_id": short_paper_id,
                    "full_id": full_paper_id
                })
        
        # Query metadata for all selected papers
        output = {}
        
        for paper_info in papers_to_query:
            short_paper_id = paper_info["short_id"]
            full_paper_id = paper_info["full_id"]
            
            try:
                # Get paper data
                paper_data = self.db.get_paper_data(full_paper_id)
                metadata = paper_data.get("metadata", {})
                
                # Process requested fields
                paper_metadata = {}
                for field in metadata_fields:
                    processed_value = process_metadata_field(metadata, field)
                    paper_metadata[field] = processed_value
                
                output[short_paper_id] = paper_metadata
                
            except Exception as e:
                output[short_paper_id] = {"error": f"Failed to retrieve metadata: {str(e)}"}
        
        # If collection_name is provided, save as a data collection
        if collection_name:
            print(f"Saving metadata results to collection: {collection_name}")
            from datetime import datetime
            tracker = self.db.add_analyst_tool_tracker(self.name, collection_name, datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
            
            # Save each result to the tracker
            for short_paper_id, data in output.items():
                if "error" in data:
                    continue
                    
                # We need full_id for the tracker
                # Find it from our papers_to_query list
                full_id = next((p["full_id"] for p in papers_to_query if p["short_id"] == short_paper_id), None)
                
                if full_id:
                    self.db.update_analyst_tool_tracker(tracker, full_id, data)
            
            # Generate CSV
            self.db.convert_analyst_tool_tracker(self.name, collection_name)
            
            # Add note to output
            output["_system_note"] = f"Results saved to collection '{collection_name}'. You can now use data_collection_names=['{collection_name}'] in complete_goal."

        return output

    def create_data_collection_request(self, collection_name="", collection_goal="",
                                       target_list=None, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "strict": True,
                    "name": "create_data_collection_request",
                    "description": "Extract structured data from research papers using an AI-generated schema. "
                                   "This tool will: (1) Generate an extraction schema based on your goal, "
                                   "(2) Extract data from ALL papers in the target list CONCURRENTLY, "
                                   "(3) Save results to a data collection you can reference later. "
                                   "Use when you need to collect the SAME types of data points from multiple papers. "
                                   "NOTE: The schema is uniform across all papers—design a broad schema that captures variations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Unique name for this data collection (e.g., 'SampleSizeExtraction', 'MethodologyAnalysis'). "
                                               "Use descriptive names as you may reference this later."
                            },
                            "collection_goal": {
                                "type": "string",
                                "description": "Detailed description of what data to extract. BE SPECIFIC about: (1) Types of data points needed, "
                                               "(2) How many instances per paper (e.g., 'all genes mentioned' vs 'top 5 most important genes'), "
                                               "(3) Any context needed. Good: 'Collect all sample size information including total N, subgroup names, "
                                               "and subgroup N values, plus any exclusion criteria.' Bad: 'Get sample sizes.'"
                            },
                            "target_list": {
                                "type": "string",
                                "description": "Name of paper list to extract from, or 'ALL PAPERS' for entire database. "
                                               "Extraction runs on ALL papers in this list—there's no per-paper filtering in the schema."
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
        
        # Track progress with a counter
        completed_count = [0]  # Using list to allow modification in nested function
        total_papers = len(papers)
        
        # Emit initial progress
        try:
            from .__main__ import emit_progress
            emit_progress(0, total_papers, f"{collection_name}")
            # Small delay to ensure initial progress is received before concurrent processing
            time.sleep(0.1)
        except:
            pass  # If emit_progress not available, continue without progress updates
        
        async def extract_from_paper(paper):
            """Extract data from a single paper"""
            paper_id = paper["database"]["paper_id"]
            short_paper_id = paper_id[:10]
            print(f"*** Extracting from Paper: {short_paper_id}")
            
            # Run async extract_data
            result = await extract_data(tool, paper["cleaned_text"])
            
            # Update tracker immediately after extraction
            self.db.update_analyst_tool_tracker(tracker, paper_id, result)
            
            # Update progress
            completed_count[0] += 1
            try:
                from .__main__ import emit_progress
                emit_progress(completed_count[0], total_papers, f"{collection_name}")
            except:
                pass
            
            print(f"*** Completed extraction for Paper: {short_paper_id}")
            return short_paper_id, paper_id, result
        
        async def run_all_extractions():
            """Run all extractions concurrently"""
            tasks = [extract_from_paper(paper) for paper in papers]
            results = await asyncio.gather(*tasks)
            print(f"All {len(results)} extractions completed!")
            
            # Emit final progress (should already be at total, but ensure)
            try:
                from .__main__ import emit_progress
                emit_progress(total_papers, total_papers, f"Completed {collection_name}")
            except:
                pass
            
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

    def save_metadata_as_collection(self, collection_name, metadata_results, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "strict": True,
                    "name": "save_metadata_as_collection",
                    "description": "Save results from get_paper_metadata() as a data collection. "
                                   "REQUIRED when require_file_output=True and you used metadata instead of extraction. "
                                   "This generates the CSV file that allows you to complete the goal.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name for the collection (e.g., 'PublicationYears'). Use this same name in data_collection_names when completing."
                            },
                            "metadata_results": {
                                "type": "object",
                                "description": "The EXACT output dictionary you received from get_paper_metadata()."
                            }
                        },
                        "additionalProperties": False,
                        "required": ["collection_name", "metadata_results"]
                    }
                }
            }
            
        print(f"Saving metadata to collection: {collection_name}")
        
        # Initialize tracker
        from datetime import datetime
        tracker = self.db.add_analyst_tool_tracker(self.name, collection_name, datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
        
        # Build short_id -> full_id mapping
        # We need this because metadata_results uses short_ids but tracker needs full_ids
        all_papers = self.db.get_all_papers_data(analyst=self.name)
        short_to_full = {}
        for paper in all_papers:
            full_id = paper["database"]["paper_id"]
            short_id = full_id[:10]
            short_to_full[short_id] = full_id
            
        # Save each result
        count = 0
        for short_id, data in metadata_results.items():
            # Skip error entries
            if "error" in data:
                continue
                
            # Get full ID
            full_id = short_to_full.get(short_id)
            if not full_id:
                # Try to see if short_id is actually a full_id (unlikely but possible)
                if len(short_id) > 10:
                    full_id = short_id
                else:
                    print(f"Warning: Could not find full ID for {short_id}")
                    continue
            
            # Update tracker
            self.db.update_analyst_tool_tracker(tracker, full_id, data)
            count += 1
            
        # Finalize and generate CSV
        self.db.convert_analyst_tool_tracker(self.name, collection_name)
        
        return f"Successfully saved {count} metadata records to collection '{collection_name}'. You can now use data_collection_names=['{collection_name}'] in complete_goal."

    def complete_goal_by_answering_question_with_evidence_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "complete_goal_by_answering_question_with_evidence",
                "description": "Submit your final answer to the research question with supporting evidence. "
                               "This is the ONLY way to complete your task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Your complete answer to the research question. Be specific and comprehensive. "
                                           "If you created data collections, summarize key findings here—don't just say 'see attached file.' "
                                           "The user should understand your findings from reading this answer even without opening files."
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Specific data points that support your answer. For small datasets (<20 items), include the full list here. "
                                           "For large datasets, provide summary statistics and key examples. DO NOT just reference data you don't show—"
                                           "either display it here OR attach it as a data collection file. Example: 'Paper abc123: 150 participants; "
                                           "Paper xyz789: 200 participants' (showing actual data)."
                        },
                        "data_collection_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "OPTIONAL: List of data collection names to attach as CSV files. Use this when: (1) You have >20 data points, "
                                           "(2) User requested downloadable data, (3) require_file_output=True. When provided, the system automatically "
                                           "injects file contents into evidence and generates download buttons. Example: ['SampleSizeExtraction'] if you "
                                           "created that collection earlier."
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
