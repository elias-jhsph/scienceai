import asyncio
import hashlib
import logging
import os
import shutil
import time
from datetime import datetime

import dictdatabase as DDB
import pandas as pd
from pandas import json_normalize

logger = logging.getLogger(__name__)

lock_files = []


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def get_projects(storage_path):
    projects = []
    if os.path.basename(storage_path) != "scienceai_db":
        storage_path = os.path.join(storage_path, "scienceai_db")
    if not os.path.exists(storage_path):
        return projects
    for project in os.listdir(storage_path):
        if os.path.isdir(os.path.join(storage_path, project)) and project.find("_-checkpoint-_") == -1:
            projects.append(project)
    return projects


# Paper Manager
class DatabaseManager:
    def __init__(
        self,
        input_pdf_directory,
        processor,
        project_name,
        storage_path=None,
        auto_prune=False,
        read_only_mode=False,
        lock_timeout=60,
    ):
        if read_only_mode:
            self.auto_prune = False
        else:
            self.auto_prune = auto_prune
        self.lock_timeout = lock_timeout
        self.read_only_mode = read_only_mode
        self.processor = processor
        self.input_pdf_directory = input_pdf_directory
        if storage_path:
            if not os.path.exists(storage_path):
                os.makedirs(storage_path)
            self.storage_path = os.path.join(storage_path, "scienceai_db")
            if not os.path.exists(self.storage_path):
                os.makedirs(self.storage_path)
        else:
            self.storage_path = os.path.join(self.input_pdf_directory, "scienceai_db")
            if not os.path.exists(self.storage_path):
                os.makedirs(self.storage_path)
        self.project_path = os.path.join(self.storage_path, project_name)
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
        self.papers_pdf_path = os.path.join(self.project_path, "papers_pdf")
        if not os.path.exists(self.papers_pdf_path):
            os.makedirs(self.papers_pdf_path)
        self.db_path = os.path.join(self.project_path, "scienceai_ddb")
        DDB.config.storage_directory = self.db_path
        self.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not DDB.at("update_time").exists():
            DDB.at("update_time").create({})
        with DDB.at("update_time").session() as (session, update_time):
            update_time[project_name] = self.update_time
            session.write()
        self.default_schema = ["metadata", "papers", "pi_context"]
        self.project_name = project_name
        if not read_only_mode:
            self.__initialize_db__()

    @staticmethod
    def log_update(func):
        def wrapper(self, *args, **kwargs):
            if self.read_only_mode:
                raise ValueError("Database is in read only mode")
            self.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if not DDB.at("update_time").exists():
                DDB.at("update_time").create({self.project_name: self.update_time})
            else:
                with DDB.at("update_time").session() as (session, update_time):
                    update_time[self.project_name] = self.update_time
                    session.write()
            result = func(self, *args, **kwargs)
            return result

        return wrapper

    async def await_update(self, timeout=None):
        DDB.config.storage_directory = self.db_path
        start = time.time()
        original_update_time = self.update_time
        while timeout is None or time.time() - start < timeout:
            await asyncio.sleep(0.3)
            if not DDB.at("update_time").exists():
                DDB.at("update_time").create({self.project_name: original_update_time})
                update_time = original_update_time
            else:
                update_time = DDB.at("update_time").read().get(self.project_name, "")
            if len(update_time) > 11 and update_time != original_update_time:
                self.update_time = update_time
                break

    def update_update_time(self):
        data = DDB.at("update_time").read()
        if data:
            temp_update_time = data.get(self.project_name, "")
            if len(temp_update_time) > 11:
                self.update_time = temp_update_time

    def get_update_time(self):
        self.update_update_time()
        return self.update_time

    @log_update
    def __initialize_db__(self):
        DDB.config.storage_directory = self.db_path
        if not DDB.at("metadata").exists():
            DDB.at("metadata").create({})
        with DDB.at("metadata").session() as (session, metadata):
            metadata[self.project_name] = {"loaded": True}
            for project in os.listdir(self.storage_path):
                if project != self.project_name:
                    if os.path.isdir(os.path.join(self.storage_path, project)):
                        if project not in metadata:
                            metadata[project] = {"loaded": False}
                        else:
                            metadata[project]["loaded"] = False
            session.write()

    @log_update
    def remove_old_default_messages(self, default_messages):
        if not DDB.at("chat").exists():
            DDB.at("chat").create({})
        if default_messages and DDB.at("chat", key="messages").exists():
            current_messages = DDB.at("chat", key="messages").read()
            index = len(current_messages) - len(default_messages)
            last_message = current_messages[index:]
            last_content = [message["content"] for message in last_message]
            if last_content == default_messages:
                with DDB.at("chat", key="messages").session() as (session, _messages):
                    current_messages[:index]
                    session.write()

    @log_update
    def ingest_paper(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise ValueError(f"File {pdf_path} does not exist")
        paper_id = sha256sum(pdf_path)
        filename = paper_id + ".pdf"
        stored_pdf_path = os.path.join(self.papers_pdf_path, filename)
        if not os.path.exists(stored_pdf_path):
            shutil.copy(pdf_path, stored_pdf_path)
        if not DDB.at("papers").exists():
            DDB.at("papers").create({})
        with DDB.at("papers").session() as (session, papers):
            if paper_id not in papers:
                papers[paper_id] = {"pdf_path": stored_pdf_path, "paper_id": paper_id}
            else:
                papers[paper_id].update({"pdf_path": stored_pdf_path})
            session.write()
        return paper_id

    def ingest_papers(self):
        found_papers = []
        pdf_files = [f for f in os.listdir(self.input_pdf_directory) if f.endswith(".pdf")]
        total_papers = len(pdf_files)

        # Emit initial progress
        try:
            from .__main__ import emit_progress

            emit_progress(0, total_papers, "Ingesting papers")
        except Exception:  # nosec
            # Ignore errors during progress emission
            pass

        for idx, file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(self.input_pdf_directory, file)
            found_papers.append(self.ingest_paper(pdf_path))

            # Emit progress after each paper
            try:
                from .__main__ import emit_progress

                emit_progress(idx, total_papers, "Ingesting papers")
            except Exception:  # nosec
                # Ignore errors during progress emission
                pass

        if self.auto_prune and DDB.at("papers").exists():
            papers = DDB.at("papers").read()
            prune_papers = [paper["paper_id"] for paper in papers if paper["paper_id"] not in found_papers]
            for paper_id in prune_papers:
                self.prune_paper(paper_id)
        return found_papers

    @log_update
    def prune_paper(self, paper_id):
        if DDB.at("papers", key=paper_id).exists():
            with DDB.at("papers").session() as (session, papers):
                paths = papers[paper_id]
                if "pdf_path" in paths and os.path.exists(paths["pdf_path"]):
                    os.remove(paths["pdf_path"])
                if "json_path" in paths and os.path.exists(paths["json_path"]):
                    os.remove(paths["json_path"])
                del papers[paper_id]
                session.write()

    @log_update
    def update_paper(self, paper_id, paper_metadata):
        if DDB.at("papers", key=paper_id).exists():
            with DDB.at("papers").session() as (session, papers):
                papers[paper_id].update(paper_metadata)
                session.write()

    def get_paper(self, paper_id):
        if not DDB.at("papers", key=paper_id).exists():
            raise ValueError(f"Paper with id {paper_id} not found")
        return DDB.at("papers", key=paper_id).read()

    def get_paper_pdf(self, paper_id):
        paper = self.get_paper(paper_id)
        return paper.get("pdf_path", None)

    def get_paper_json(self, paper_id):
        paper = self.get_paper(paper_id)
        json_path = paper.get("json_path", None)
        if not json_path:
            # Construct default path
            json_path = os.path.join(self.db_path, paper_id + ".json")
        return json_path

    @log_update
    def store_paper_json(self, paper_id, dict_data):
        paper = self.get_paper(paper_id)
        if not DDB.at(paper_id).exists():
            DDB.at(paper_id).create(dict_data)
        with DDB.at("papers").session() as (session, papers):
            papers[paper_id] = paper
            if dict_data.get("metadata"):
                added = {}
                if dict_data["metadata"].get("title"):
                    added["Title"] = dict_data["metadata"]["title"][0]
                if dict_data["metadata"].get("created") and dict_data["metadata"]["created"].get("date-time"):
                    added["Date"] = dict_data["metadata"]["created"]["date-time"][:10]
                try:
                    author = dict_data["metadata"]["author"][0]
                    given = author.get("given", "")
                    family = author.get("family", author.get("name", "Unknown"))
                    added["Authors"] = f"{given} {family}".strip()
                except (KeyError, IndexError, TypeError) as e:
                    logger.warning(f"Could not extract author for paper {paper_id}: {e}")
                papers[paper_id].update(added)
            session.write()
        return True

    def _clear_paper_error_status(self, paper_id):
        """Clear error and status keys from a paper before reprocessing."""
        if DDB.at("papers", key=paper_id).exists():
            with DDB.at("papers").session() as (session, papers):
                if paper_id in papers:
                    papers[paper_id].pop("error", None)
                    papers[paper_id].pop("status", None)
                    session.write()

    async def process_paper(self, paper_id, semaphore):
        """Processes the paper asynchronously with a semaphore"""
        import traceback

        async with semaphore:
            pdf_path = self.get_paper_pdf(paper_id)
            if not DDB.at(paper_id).exists():
                # Clear any previous error/status before reprocessing
                self._clear_paper_error_status(paper_id)

                logger.debug(f"Processing paper {paper_id}...")
                try:
                    processed_paper = await self.processor(pdf_path)
                    self.store_paper_json(paper_id, processed_paper)
                    logger.debug(f"Finished paper {paper_id}")
                except Exception as e:
                    # Log full traceback for better debugging
                    full_traceback = traceback.format_exc()
                    logger.error(f"Failed to process paper {paper_id}: {e}\n{full_traceback}")
                    self.update_paper(paper_id, {"status": "failed", "error": str(e)})

    async def process_all_papers(self):
        """Processes all the papers in parallel"""
        logger.info("Processing all papers")
        paper_ids = list(DDB.at("papers").read().keys())
        total_papers = len(paper_ids)

        # Track completed papers
        completed_count = [0]

        # Emit initial progress
        try:
            from .__main__ import emit_progress

            emit_progress(0, total_papers, "Processing papers")
        except Exception:  # nosec
            # Ignore errors during progress emission
            pass

        # Limit concurrency to avoid hitting rate limits too hard
        # 5 concurrent papers * ~10 concurrent pages per paper = ~50 concurrent requests
        semaphore = asyncio.Semaphore(5)

        async def process_with_progress(paper_id):
            result = await self.process_paper(paper_id, semaphore)
            completed_count[0] += 1
            try:
                from .__main__ import emit_progress

                emit_progress(completed_count[0], total_papers, "Processing papers")
            except Exception:  # nosec
                # Ignore errors during progress emission
                pass
            return result

        tasks = [process_with_progress(paper_id) for paper_id in paper_ids]
        await asyncio.gather(*tasks)
        return True

    @log_update
    def add_chat(self, message):
        with DDB.at("chat").session() as (session, chat):
            if "messages" not in chat:
                chat["messages"] = []
            chat["messages"].append(message)
            session.write()
        return True

    @log_update
    def update_last_chat(self, status, full_update=None):
        with DDB.at("chat").session() as (session, chat):
            if "messages" in chat:
                if full_update:
                    chat["messages"][-1] = full_update
                chat["messages"][-1]["status"] = status
                session.write()
        return True

    @log_update
    def pop_last_chat(self):
        """Remove the last message from the chat history."""
        with DDB.at("chat").session() as (session, chat):
            if "messages" in chat and len(chat["messages"]) > 0:
                chat["messages"].pop()
                session.write()
        return True

    @log_update
    def set_all_chat_messages(self, messages):
        """Replace all chat messages with the provided list."""
        with DDB.at("chat").session() as (session, chat):
            chat["messages"] = messages
            session.write()
        return True

    @log_update
    def create_analyst(self, name, goal, other=None):
        if other is None:
            other = {}
        if not DDB.at("Analysts").exists():
            DDB.at("Analysts").create({})
        if not DDB.at(name).exists():
            DDB.at(name).create({})
        with DDB.at("Analysts").session() as (session, analysts):
            if name not in analysts:
                analysts[name] = {"goal": goal, **other}
            session.write()
        with DDB.at(name).session() as (session, analyst):
            analyst["context"] = []
            session.write()
        return True

    @log_update
    def add_analyst_context(self, name, analyst_context):
        if not DDB.at(name).exists():
            DDB.at(name).create({})
        if not DDB.at(name).exists():
            raise ValueError(f"Analyst {name} not found")
        with DDB.at(name).session() as (session, analyst):
            analyst["context"].append(analyst_context)
            session.write()
        return True

    @log_update
    def add_analyst_tool_tracker(self, analyst_name, tool_name, tool_time, json_data=None):
        if json_data is None:
            json_data = {}
        tool_fullname = analyst_name + "_/" + tool_name + "_" + tool_time
        if not DDB.at(tool_fullname).exists():
            DDB.at(tool_fullname).create(json_data)
        else:
            with DDB.at(tool_fullname).session() as (session, tool):
                tool.update(json_data)
                session.write()
        time.sleep(3)
        with DDB.at("Analysts").session() as (session, analysts):
            if analyst_name not in analysts:
                raise ValueError(f"Analyst {analyst_name} not found")
            if "tools" not in analysts[analyst_name]:
                analysts[analyst_name]["tools"] = []
            analysts[analyst_name]["tools"].append(
                {"tool_name": tool_name, "json_path": tool_fullname + ".json", "hidden": True}
            )
            session.write()
        return tool_fullname

    @log_update
    def revert_chat_from_index(self, index):
        """
        Reverts the chat history by removing all messages from the given index (inclusive) to the end.
        Used for undoing a user request and all its associated responses.
        """
        if not DDB.at("chat").exists():
            return False

        with DDB.at("chat").session() as (session, chat):
            if "messages" in chat and index < len(chat["messages"]):
                chat["messages"] = chat["messages"][:index]
            session.write()

        return True

    @log_update
    def convert_analyst_tool_tracker(self, analyst_name, tool_name):
        if not DDB.at("Analysts", key=analyst_name).exists():
            raise ValueError(f"Analyst {analyst_name} not found")
        data = None
        with DDB.at("Analysts", key=analyst_name).session() as (session, analyst):
            for i, tool in enumerate(analyst["tools"]):
                if tool["tool_name"] == tool_name:
                    data_path = tool["json_path"]
                    data = DDB.at(data_path.replace(".json", "")).read()

                    # Normalize data for CSV
                    # Handle both extraction results (with source_quote etc) and metadata results (flat key-value)
                    normalized_items = []
                    for k, v in data.items():
                        # Skip system keys (like _system_note)
                        if k.startswith("_"):
                            continue

                        item = {"id": k[:10]}
                        if isinstance(v, dict):
                            # Filter out error fields if present
                            if "error" in v:
                                continue
                            # Flatten the dictionary
                            for field_key, field_val in v.items():
                                # Special handling for derivation objects
                                if field_key.endswith("_derivation") and isinstance(field_val, dict):
                                    # Flatten derivation into multiple columns
                                    item[f"{field_key}_operation"] = field_val.get("operation", "")
                                    item[f"{field_key}_description"] = field_val.get("operation_description", "")
                                    item[f"{field_key}_computation"] = field_val.get("computation", "")

                                    # Concatenate source quotes for readability
                                    sources = field_val.get("sources", [])
                                    if sources:
                                        sources_text = " | ".join(
                                            [f"{s.get('location', '')}: {s.get('quote', '')}" for s in sources]
                                        )
                                        item[f"{field_key}_sources"] = sources_text
                                    else:
                                        item[f"{field_key}_sources"] = ""
                                else:
                                    item[field_key] = field_val
                        else:
                            # Handle simple value case (unlikely but possible)
                            item["value"] = v
                        normalized_items.append(item)

                    flat_data = json_normalize(normalized_items)

                    # Auto-add paper_title column from metadata if 'id' column exists
                    if "id" in flat_data.columns:
                        papers = DDB.at("papers").read()
                        # Create title lookup - match on beginning of paper ID (truncated in CSV)
                        title_map = {}
                        for pid, paper in papers.items():
                            truncated_id = pid[:10]
                            title_map[truncated_id] = paper.get("Title", "")
                        # Insert paper_title as the second column (after id)
                        flat_data.insert(1, "paper_title", flat_data["id"].map(title_map))

                    csv_path = data_path.replace(".json", ".csv")
                    csv_folder = os.path.join(self.project_path, "csv_files")
                    if not os.path.exists(csv_folder):
                        os.makedirs(csv_folder)
                    csv_path = os.path.join(csv_folder, os.path.basename(csv_path))
                    flat_data.to_csv(csv_path, index=False)
                    analyst["tools"][i]["csv_path"] = csv_path
            if not data:
                raise ValueError(f"Tool {tool_name} not found for analyst {analyst_name}")
            # dict comprehension to add an id key of the first 10 characters to each value
            session.write()
        return csv_path

    def combine_analyst_tool_trackers(self):
        if not DDB.at("Analysts").exists():
            df = pd.DataFrame(
                {
                    "Notes": [
                        "No Analysts have been created yet. After they are created, results of their"
                        " data extraction will be combined and accessible under the "
                        "'Extracted Data' tab."
                    ]
                }
            )
            df.to_csv(os.path.join(self.project_path, "merged_analyst_tools.csv"), index=False)
            return os.path.join(self.project_path, "merged_analyst_tools.csv")
        csv_paths = {}
        for analyst_name in DDB.at("Analysts").read():
            analyst = DDB.at("Analysts", key=analyst_name).read()
            if "tools" in analyst:
                for tool in analyst["tools"]:
                    if "csv_path" in tool:
                        csv_paths[tool["csv_path"]] = {"name": tool["tool_name"], "analyst": analyst_name}
        if not csv_paths:
            df = pd.DataFrame(
                {
                    "Notes": [
                        "No Analysts have extracted data yet. After they do, the results will be "
                        "combined and accessible under the 'Extracted Data' tab."
                    ]
                }
            )
            df.to_csv(os.path.join(self.project_path, "merged_analyst_tools.csv"), index=False)
            return os.path.join(self.project_path, "merged_analyst_tools.csv")

        bad_paths = []
        for csv in csv_paths:
            try:
                pd.read_csv(csv)
            except Exception as e:
                logger.error(f"Error reading csv {csv}: {e}")
                bad_paths.append(csv)

        for bad_path in bad_paths:
            del csv_paths[bad_path]

        merged_df = pd.DataFrame()

        for csv, meta in csv_paths.items():
            df = pd.read_csv(csv)
            # Always rename all non-id columns with a unique suffix to prevent any merge conflicts
            # This ensures no duplicate columns exist when merging
            suffix = f"_{meta['analyst']}_{meta['name']}"
            remap = {col: col + suffix for col in df.columns if col != "id"}
            df.rename(columns=remap, inplace=True)
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, how="outer", on="id")

        papers = self.get_database_papers()
        papers = [{k: v for k, v in paper.items() if "_path" not in k} for paper in papers]
        papers = [{k if k != "paper_id" else "id": v for k, v in paper.items()} for paper in papers]
        papers = [{k: (v[:10] if k == "id" else v) for k, v in paper.items()} for paper in papers]

        papers_df = pd.DataFrame(papers)

        merged_df = pd.merge(papers_df, merged_df, how="outer", on="id")

        merged_df.to_csv(os.path.join(self.project_path, "merged_analyst_tools.csv"), index=False)
        return os.path.join(self.project_path, "merged_analyst_tools.csv")

    @log_update
    def update_analyst_tool_tracker(self, json_path, key, update_data, overwrite_list=False):
        json_path = json_path.replace(".json", "")
        if not DDB.at(json_path).exists():
            raise ValueError(f"File {json_path} does not exist")
        with DDB.at(json_path).session() as (session, data):
            if key not in data:
                data[key] = update_data
            elif isinstance(data[key], dict):
                data[key].update(update_data)
            elif isinstance(data[key], list):
                if overwrite_list:
                    data[key] = update_data
                else:
                    if isinstance(update_data, list):
                        data[key].extend(update_data)
                    else:
                        data[key].append(update_data)
            session.write()
        return True

    @log_update
    def add_analyst_metadata(self, name, metadata):
        if not DDB.at("Analysts").exists():
            DDB.at("Analysts").create({})
        if not DDB.at("Analysts", key=name).exists():
            raise ValueError(f"Analyst {name} not found")
        with DDB.at("Analysts").session() as (session, analysts):
            analysts[name].update(metadata)
            session.write()
        return True

    @log_update
    def remove_analyst(self, name):
        """
        Removes all traces of an analyst by name.
        """
        if not DDB.at("Analysts").exists():
            return False
        if not DDB.at("Analysts", key=name).exists():
            return False

        # 1. Remove all lists created by this analyst
        self.remove_all_analyst_lists(name)

        # 2. Clean up tools and associated files
        try:
            analyst_data = DDB.at("Analysts", key=name).read()
            if "tools" in analyst_data:
                for tool in analyst_data["tools"]:
                    # Remove JSON file
                    json_filename = tool["json_path"]
                    json_db_name = json_filename.replace(".json", "")
                    if DDB.at(json_db_name).exists():
                        DDB.at(json_db_name).delete()

                    # Remove CSV file if it exists
                    if tool.get("csv_path"):
                        csv_path = tool["csv_path"]
                        if os.path.exists(csv_path):
                            try:
                                os.remove(csv_path)
                            except OSError as e:
                                logger.error(f"Error removing CSV {csv_path}: {e}")

        except Exception as e:
            logger.error(f"Error cleaning up analyst tools: {e}")

        # 3. Remove the analyst's individual context database
        if DDB.at(name).exists():
            DDB.at(name).delete()

        # 4. Remove from Analysts registry
        with DDB.at("Analysts").session() as (session, analysts):
            if name in analysts:
                del analysts[name]
            session.write()

        # 5. Remove analyst tracker directory (files with prefix analyst_name + "_/")
        tracker_dir_prefix = name + "_/"
        tracker_dir_path = os.path.join(self.db_path, tracker_dir_prefix)
        if os.path.exists(tracker_dir_path) and os.path.isdir(tracker_dir_path):
            try:
                import shutil

                shutil.rmtree(tracker_dir_path)
                logger.info(f"Removed tracker directory: {tracker_dir_path}")
            except Exception as e:
                logger.error(f"Error removing tracker directory {tracker_dir_path}: {e}")

        return True

    @log_update
    def add_paper_to_list(self, paper_id, analyst, name_of_list):
        if not DDB.at("papers", key=paper_id).exists():
            raise ValueError(f"Paper with id {paper_id} not found")
        if not DDB.at("Analysts").exists():
            DDB.at("Analysts").create({})
        if not DDB.at("Analysts", key=analyst).exists():
            raise ValueError(f"Analyst {analyst} not found")
        with DDB.at("papers").session() as (session, papers):
            if analyst not in papers[paper_id]:
                papers[paper_id][analyst] = [name_of_list]
            else:
                if name_of_list not in papers[paper_id][analyst]:
                    papers[paper_id][analyst].append(name_of_list)
            session.write()
        return True

    @log_update
    def remove_paper_from_list(self, paper_id, analyst, name_of_list=None):
        if not DDB.at("papers", key=paper_id).exists():
            raise ValueError(f"Paper with id {paper_id} not found")
        if not DDB.at("Analysts").exists():
            DDB.at("Analysts").create({})
        with DDB.at("papers").session() as (session, papers):
            if analyst in papers[paper_id]:
                if name_of_list is None:
                    papers[paper_id][analyst] = []
                elif name_of_list in papers[paper_id][analyst]:
                    papers[paper_id][analyst].remove(name_of_list)
            session.write()
        return True

    def get_all_tool_trackers_for_analyst(self, analyst_name):
        analysts = DDB.at("Analysts").read()
        if not analysts:
            return []
        if analyst_name not in analysts:
            raise ValueError(f"Analyst {analyst_name} not found")
        if "tools" not in analysts[analyst_name]:
            return []
        files = {}
        tools = {}
        for tool in analysts[analyst_name]["tools"]:
            csv_path = None
            if "csv_path" in tool:
                csv_path = tool["csv_path"]
            json_path = os.path.join(self.project_path, "scienceai_ddb", tool["json_path"])
            if tool["tool_name"][:-19] not in files:
                files[tool["tool_name"][:-19]] = json_path
                tools[json_path] = csv_path
            else:
                if tool["tool_name"][-19:] > files[tool["tool_name"][:-19]][-19:]:
                    files[tool["tool_name"][:-19]] = json_path
                    tools[json_path] = csv_path
        return tools

    def get_analyst_tool_tracker(self, json_path):
        data = DDB.at(json_path).read()
        return data

    @log_update
    def clear_analyst_context(self, name):
        if not DDB.at(name).exists():
            raise ValueError(f"Analyst {name} not found")
        with DDB.at("Analysts").session() as (session, analysts):
            analysts[name]["contexts"] = []
            session.write()
        return True

    def get_analyst_metadata(self, name):
        if not DDB.at("Analysts", key=name).exists():
            raise ValueError(f"Analyst {name} not found")
        with DDB.at("Analysts").session() as (_session, analysts):
            return analysts[name]

    def get_all_analysts(self):
        if not DDB.at("Analysts").exists():
            return []
        else:
            all = []
            for name in DDB.at("Analysts").read():
                temp = self.get_analyst_metadata(name).copy()
                temp["name"] = name
                all.append(temp)
            return all

    def get_all_papers(self, analyst=None, named_list=None):
        if named_list and not analyst:
            raise ValueError("If named_list is provided, analyst must also be provided")
        papers = DDB.at("papers").read()
        if analyst and named_list:
            result = [paper for paper in papers.values() if analyst in paper and named_list in paper[analyst]]
            return result
        return list(papers.values())

    @log_update
    def remove_all_analyst_lists(self, analyst):
        """Remove all papers associations for an analyst without nested sessions."""
        if not DDB.at("papers").exists():
            return True
        # Do the cleanup in a single session - don't call remove_paper_from_list
        # which would create a nested session on the same database
        with DDB.at("papers").session() as (session, papers):
            for paper_id in papers:
                if analyst in papers[paper_id]:
                    del papers[paper_id][analyst]  # Delete the key entirely
            session.write()
        return True

    def get_paper_data(self, paper_id):
        if not DDB.at("papers", key=paper_id).exists():
            raise ValueError(f"Paper with id {paper_id} not found")
        paper = DDB.at("papers", key=paper_id).read()
        data = {}
        if DDB.at(paper_id).exists():
            data = DDB.at(paper_id).read()
        data["database"] = paper
        return data

    def get_all_papers_data(self, analyst=None, named_list=None, selected_key=None):
        papers = self.get_all_papers(analyst=analyst, named_list=named_list)
        output = []
        for paper in papers:
            paper_id = paper["paper_id"]
            paper_data = self.get_paper_data(paper_id)
            if selected_key:
                paper_data = paper_data.get(selected_key)
            else:
                output.append(paper_data)
        return output

    def get_analyst_data_visual(self, path):
        analysts = DDB.at("Analysts").read()
        if not analysts:
            return {}
        else:
            analysts = list(analysts.keys())
        if path == "/":
            return {k: {} for k in analysts}
        path_parts = path.strip("/").split("/")
        if len(path_parts) == 1:
            add_ons = {"evidence_files": {}, "internal_memory": {}}
            analyst = DDB.at(path_parts[0]).read()
            analyst.update(add_ons)
            return analyst
        if len(path_parts) == 2:
            metadata = self.get_analyst_metadata(path_parts[1]).copy()
            if not metadata:
                return {}
            metadata["evidence_files"] = {}
            metadata["internal_memory"] = {}
            if "tools" in metadata:
                del metadata["tools"]
            return metadata
        if len(path_parts) > 2 and path_parts[2] == "evidence_files":
            results = self.get_all_tool_trackers_for_analyst(path_parts[1])
            if len(path_parts) == 3:
                if not results:
                    return {}
                return {os.path.basename(tool)[:-25]: {} for tool in results}
            if len(path_parts) == 4:
                for json_path, csv_path in results.items():
                    if os.path.basename(json_path)[:-25] == path_parts[3]:
                        temp = {"json_path": json_path}
                        if csv_path:
                            temp["csv_path"] = csv_path
                        return temp
        if path_parts[2] == "internal_memory":
            results = self.get_analyst_context(path_parts[1], include_hidden=True)
            if len(path_parts) == 3:
                return {f"memory - {i + 1}": {} for i in range(len(results))}
            if len(path_parts) > 3:
                result = results[int(path_parts[3].split(" - ")[1]) - 1]
                for part in path_parts[4:]:
                    result = result[part] if isinstance(result, dict) else result[int(part)]
                return result if isinstance(result, dict) else {str(i): v for i, v in enumerate(result)}

    def get_database_papers(self):
        full = DDB.at("papers").read()
        if not full:
            return []
        return [
            {**paper, **{"json_path": os.path.join(self.project_path, "scienceai_ddb", paper["paper_id"] + ".json")}}
            if "Title" in paper
            else paper
            for paper in full.values()
        ]

    def get_database_chat(self):
        full = DDB.at("chat", key="messages").read()
        if not full:
            return []
        return full

    def get_last_message(self):
        if not DDB.at("chat", key="messages").exists():
            return None
        full = DDB.at("chat", key="messages").read()
        if len(full) == 0:
            return None
        return full[-1]

    def get_analyst_context(self, name, include_hidden=False):
        if not DDB.at(name).exists():
            raise ValueError(f"Analyst {name} not found")
        if include_hidden:
            return DDB.at(name).read()["context"]
        return [context for context in DDB.at(name).read()["context"] if not context.get("hidden", False)]

    def get_last_save(self, path=False):
        existing_save = None
        for project in os.listdir(self.storage_path):
            if project.find(self.project_name + "_-checkpoint-_") > -1:
                existing_save = project
        if path and existing_save:
            return os.path.join(self.storage_path, existing_save)
        return existing_save

    def save_database(self):
        existing_save = self.get_last_save(path=True)
        new_save = self.project_name + "_-checkpoint-_" + self.update_time.replace(" ", "_").replace(":", "_")
        shutil.copytree(self.project_path, os.path.join(self.storage_path, new_save))
        if existing_save:
            shutil.rmtree(existing_save)

    def lock_project(self, timeout=5):
        """
        Attempt to acquire write locks on all critical database files.

        This is used before force-terminating threads to ensure no writes are in progress.
        If we can acquire locks on all critical files, it means no other process/thread
        is currently writing to them, and it's safe to force-terminate.

        Args:
            timeout: Maximum seconds to wait for each lock (default 5)

        Returns:
            True if all locks acquired successfully, False otherwise
        """
        import contextlib
        import threading

        critical_files = ["chat", "Analysts", "papers", "update_time"]
        acquired_sessions = []
        lock_failed = threading.Event()

        def try_acquire_lock(file_key):
            try:
                if DDB.at(file_key).exists():
                    # Open a session which acquires the lock
                    session_ctx = DDB.at(file_key).session()
                    session, _data = session_ctx.__enter__()
                    acquired_sessions.append((session_ctx, session))
                    logger.debug(f"Lock acquired on {file_key}")
            except Exception as e:
                logger.warning(f"Failed to acquire lock on {file_key}: {e}")
                lock_failed.set()

        # Try to acquire all locks with timeout
        threads = []
        for file_key in critical_files:
            t = threading.Thread(target=try_acquire_lock, args=(file_key,), daemon=True)
            t.start()
            threads.append(t)

        # Wait for all threads with timeout
        for t in threads:
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning("Timeout waiting for lock")
                lock_failed.set()

        # Release all acquired sessions (we don't write, just checking they're available)
        for session_ctx, _session in acquired_sessions:
            with contextlib.suppress(Exception):
                session_ctx.__exit__(None, None, None)

        if lock_failed.is_set():
            return False

        logger.info("All database locks acquired - safe to terminate")
        return True

    def get_project_setting(self, key: str, default=None):
        """
        Get a project-level setting value.

        Args:
            key: The setting key (e.g., 'n_parallel_calls')
            default: Default value if setting doesn't exist

        Returns:
            The setting value, or default if not found
        """
        if not DDB.at("settings").exists():
            return default
        settings = DDB.at("settings").read()
        return settings.get(key, default)

    @log_update
    def set_project_setting(self, key: str, value):
        """
        Set a project-level setting value.

        Args:
            key: The setting key (e.g., 'n_parallel_calls')
            value: The value to store
        """
        if not DDB.at("settings").exists():
            DDB.at("settings").create({})
        with DDB.at("settings").session() as (session, settings):
            settings[key] = value
            session.write()
        return True
