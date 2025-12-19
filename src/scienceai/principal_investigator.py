import concurrent.futures
import contextlib
import io
import json
import logging
import os
import threading
import traceback
from datetime import datetime
from time import sleep

from scienceai.analyst import Analyst

from .bundle_validator import validate_bundle, validate_bundle_tool_schema
from .database_manager import DatabaseManager
from .llm import MODEL_VISION, client, enc, get_config, get_model_for_role
from .llm import use_tools_sync as use_tools
from .reasoning import add_reasoning_to_context

logger = logging.getLogger(__name__)

path_to_app = os.path.dirname(os.path.abspath(__file__))


def _load_pi_system_prompt() -> str:
    """Load the PI system prompt with provider-specific prepend and append.

    Loads the base prompt and adds provider-specific instructions:
    - Prepend: Initial context and reminders at the start
    - Append: Critical rules at the end (leverages recency bias)
    """
    from .llm_providers import Provider, get_provider_type

    # Load base prompt
    with open(os.path.join(path_to_app, "principal_investigator_base_prompt.txt")) as file:
        base_prompt = file.read()

    prepend_content = ""
    append_content = ""

    # Determine provider and load corresponding prepend/append
    try:
        provider_type = get_provider_type()
        logger.info(f"Provider type: {provider_type}")

        if provider_type == Provider.ANTHROPIC:
            prepend_file = "pi_prepend_anthropic.txt"
            append_file = "pi_append_anthropic.txt"
        elif provider_type == Provider.OPENAI:
            prepend_file = "pi_prepend_openai.txt"
            append_file = "pi_append_openai.txt"
        elif provider_type == Provider.GOOGLE:
            prepend_file = "pi_prepend_google.txt"
            append_file = "pi_append_google.txt"
        else:
            prepend_file = None
            append_file = None

        if prepend_file:
            prepend_path = os.path.join(path_to_app, "prompts", prepend_file)
            if os.path.exists(prepend_path):
                with open(prepend_path) as f:
                    content = f.read().strip()
                if content:
                    prepend_content = content + "\n\n"
                    logger.info(f"Loaded PI prepend for provider: {provider_type.value}")

        if append_file:
            append_path = os.path.join(path_to_app, "prompts", append_file)
            if os.path.exists(append_path):
                with open(append_path) as f:
                    content = f.read().strip()
                if content:
                    append_content = "\n\n" + content
                    logger.info(f"Loaded PI append for provider: {provider_type.value}")

    except Exception as e:
        logger.warning(f"Could not load provider-specific prompt files: {e}")

    return prepend_content + base_prompt + append_content


# System message is now loaded dynamically in PrincipalInvestigator.__init__


class PrincipalInvestigator:
    def __init__(self, dbr: DatabaseManager):
        self.db = dbr
        self._lock = threading.Lock()
        # Load persistent setting, default to 1 if not set
        self.n_parallel_calls = dbr.get_project_setting("n_parallel_calls", default=1)
        self.analysts = []
        analysts_db = dbr.get_all_analysts()
        for analyst_dict in analysts_db:
            self.analysts.append(Analyst(dbr, analyst_dict=analyst_dict))
        self.tool_callables = {
            "delegate_research": self.delegate_research,
            "reflect_on_delegations": self.reflect_on_delegations,
            "get_analyst_data_link": self.get_analyst_data_link,
            "view_image": self.view_image,
            "validate_analytic_bundle": self.validate_analytic_bundle,
            "mark_analyst_result_analyzed": self.mark_analyst_result_analyzed,
        }
        self.tools = [
            self.delegate_research_schema(),
            self.reflect_on_delegations(return_tool=True),
            self.get_analyst_data_link(None, None, return_tool=True),
            self.run_python_code(None, return_tool=True),
            self.view_image(None, return_tool=True),
            validate_bundle_tool_schema(),
            self.mark_analyst_result_analyzed(None, None, return_tool=True),
        ]
        self.tool_callables["run_python_code"] = self.run_python_code
        self.system_message = _load_pi_system_prompt()

    async def initialize(self, ingest=True):
        chat_db = self.db.get_database_chat()
        first_message = (
            "Hello, I am ScienceAI. I first need to make sure all your papers are loaded into the system "
            "before I can help you. I will let you know when I am ready to answer your questions. "
            "This may take a long time if you uploaded many papers."
        )
        second_message_base = "All papers have been loaded into the system."
        # For matching old messages without paper count
        defaults = [first_message, second_message_base]
        self.db.remove_old_default_messages(defaults)

        def get_second_message_with_count():
            """Get the second message with paper count included."""
            paper_count = len(self.db.get_database_papers())
            return f"All {paper_count} papers have been loaded into the system."

        if len(chat_db) > 0:
            last_chat = chat_db[-1]
            if last_chat.get("content") == "CONTEXTLIMITREACHED":
                # Remove the context limit message so the user can continue or re-trigger it
                self.db.pop_last_chat()
                # Refresh chat_db after modification
                chat_db = self.db.get_database_chat()
                if len(chat_db) > 0:
                    last_chat = chat_db[-1]
                else:
                    last_chat = None

            if last_chat and last_chat["content"] == first_message:
                if ingest:
                    self.db.update_last_chat("Pending")
                    self.db.ingest_papers()
                    await self.db.process_all_papers()
                    self.db.update_last_chat("Processed")
                    second = {
                        "content": get_second_message_with_count(),
                        "role": "system",
                        "status": "Pending",
                        "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
                    }
                    self.db.add_chat(second)
                    self.db.update_last_chat("Processed")
            elif last_chat["content"] == second_message_base or (
                last_chat["content"].startswith("All ") and "papers have been loaded" in last_chat["content"]
            ):
                self.db.update_last_chat("Processed")
            else:
                if last_chat["status"] == "Pending":
                    if last_chat.get("tool_calls"):
                        await self.finish_tool_calls(last_chat)
                    elif last_chat["role"] == "user":
                        await self.process_message(
                            last_chat["content"],
                            last_chat["role"],
                            last_chat["status"],
                            last_chat["time"],
                            store_message=False,
                        )
        else:
            first = {
                "content": first_message,
                "role": "system",
                "status": "Pending",
                "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
            }
            self.db.add_chat(first)
            if ingest:
                self.db.ingest_papers()
                await self.db.process_all_papers()
                self.db.update_last_chat("Processed")
                second = {
                    "content": get_second_message_with_count(),
                    "role": "system",
                    "status": "Pending",
                    "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
                }
                self.db.add_chat(second)
                self.db.update_last_chat("Processed")
        self.db.update_last_chat("Processed")
        messages = self.db.get_database_chat()
        for msg in messages:
            if msg.get("status") == "Pending":
                msg["status"] = "Processed"
        # Update the chat in the database
        self.db.set_all_chat_messages(messages)

    def delegate_research_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "delegate_research",
                "description": "Delegate data extraction from research papers to a specialized Analyst Agent. "
                "CRITICAL: Each delegation must focus on ONE outcome type only. "
                "For multiple outcomes (e.g., nonunion + time to union + infections), create SEPARATE delegate_research calls. "
                "WRONG: One delegation for 'nonunion, time to union, and infection data'. "
                "RIGHT: Three delegate_research calls - one for nonunion, one for time to union, one for infections. "
                "The Analyst will collect structured data and optionally create CSV files. "
                "Use this when you need NEW information from papers that hasn't been collected yet. "
                "Returns answer and evidence from the Analyst after completion. "
                "NOTE: Some extraction failures (1-2 papers) are normal - don't re-delegate just for a 90%+ success rate. Unless you suspect data inconsistency, don't re-delegate.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Descriptive name for this analyst (e.g., 'Sample Size Analyst', 'Methods Analyst'). "
                            "Keep it concise but specific to the task. Used to track analyst's work.",
                        },
                        "question": {
                            "type": "string",
                            "description": "The research question or data extraction goal. Be specific about WHAT to collect, "
                            "not HOW to format it. **If you already know which specific papers to analyze** "
                            "(e.g., from previous analyst results or your own analysis), include their paper IDs "
                            "or titles directly in the question (e.g., 'For papers [abc123def4, xyz789ghi0], collect "
                            "sample sizes' or 'From papers titled X, Y, Z, collect methods'). This ensures consistency "
                            "and avoids forcing the analyst to rediscover your paper selection. Only use general "
                            "descriptions (e.g., 'papers using qualitative methods') when you need the analyst to "
                            "discover or filter papers themselves. Include what data points are needed and any "
                            "specific details (like lists vs fixed counts).",
                        },
                        "require_file_output": {
                            "type": "boolean",
                            "description": "Set to true when you need downloadable CSV files with structured data (typically "
                            "for 10+ papers or complex multi-field data extraction). Set to false for quick queries "
                            "or summary analyses. Default: false. When true, analyst MUST attach data extraction files.",
                        },
                    },
                    "required": ["name", "question"],
                },
            },
        }

    def _get_analyst_collections(self, analyst_name: str) -> list:
        """
        Get collection names for an analyst.

        Returns a list of collection names (without timestamp suffixes).
        """
        try:
            analysts = self.db.get_all_analysts()
            for analyst in analysts:
                if analyst.get("name") == analyst_name:
                    tools = analyst.get("tools", [])
                    # tool_name format: CollectionName_YYYY-MM-DD_HH_MM_SS (last 20 chars = _timestamp)
                    # Actually: tool_name is just the collection_name stored directly
                    return [tool.get("tool_name", "") for tool in tools if tool.get("tool_name")]
            return []
        except Exception as e:
            logger.warning(f"Could not get collections for analyst {analyst_name}: {e}")
            return []

    def _build_loading_instructions(self, analyst_name: str, collections: list) -> str:
        """
        Build clear loading instructions with pandas code for all collections.

        Args:
            analyst_name: Name of the analyst (could include 'copy N' suffix)
            collections: List of collection names

        Returns:
            Formatted instructions string with code examples
        """
        if not collections:
            return ""

        lines = [
            "\n\n---",
            "**📊 HOW TO LOAD THIS DATA (using pandas):**\n",
            "```python",
            "import pandas as pd",
            "",
        ]

        for collection in collections:
            lines.append(f"# Load '{collection}' collection")
            lines.append(f"filename = load_analyst_data('{analyst_name}', '{collection}')")
            lines.append(f"df_{collection.replace(' ', '_').replace('-', '_').lower()} = pd.read_csv(filename)")
            lines.append("")

        lines.append("```")
        lines.append("---\n")

        return "\n".join(lines)

    def delegate_research(self, name, question, require_file_output=False, n_parallel_calls=None):
        # Use instance variable if not explicitly provided (top-level call)
        if n_parallel_calls is None:
            n_parallel_calls = self.n_parallel_calls

            # Check for resumption of interrupted parallel task
            # If current setting is 1 but we find "Name copy 1", it implies previous intent was parallel
            if n_parallel_calls == 1:
                with self._lock:
                    has_copy_1 = any(a.name == f"{name} copy 1" for a in self.analysts)
                    if has_copy_1:
                        # Count how many copies exist
                        existing_count = 1
                        while True:
                            next_copy = f"{name} copy {existing_count + 1}"
                            if any(a.name == next_copy for a in self.analysts):
                                existing_count += 1
                            else:
                                break

                        n_parallel_calls = max(existing_count, 2)
                        logger.info(
                            f"Detected existing parallel analysts for '{name}'. Resuming with {n_parallel_calls} replicates."
                        )

        if n_parallel_calls > 1:
            header = f"Running {n_parallel_calls} parallel copies of analyst {name}..."
            logger.info(header)

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel_calls) as executor:
                futures = []
                for i in range(1, n_parallel_calls + 1):
                    analyst_name = f"{name} copy {i}"
                    futures.append(
                        executor.submit(
                            self.delegate_research,
                            analyst_name,
                            question,
                            require_file_output,
                            n_parallel_calls=1,
                        )
                    )

                results = []
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Parallel delegation error: {e}")
                        results.append(f"Error in parallel execution: {e}")

            # Build combined loading instructions for all parallel copies
            all_loading_instructions = []
            for i in range(1, n_parallel_calls + 1):
                copy_name = f"{name} copy {i}"
                collections = self._get_analyst_collections(copy_name)
                if collections:
                    all_loading_instructions.append(f"**{copy_name}:**")
                    all_loading_instructions.append("```python")
                    all_loading_instructions.append("import pandas as pd")
                    all_loading_instructions.append("")
                    for collection in collections:
                        all_loading_instructions.append(f"# Load '{collection}'")
                        all_loading_instructions.append(f"filename = load_analyst_data('{copy_name}', '{collection}')")
                        safe_name = collection.replace(" ", "_").replace("-", "_").lower()
                        all_loading_instructions.append(f"df_{safe_name} = pd.read_csv(filename)")
                        all_loading_instructions.append("")
                    all_loading_instructions.append("```")
                    all_loading_instructions.append("")

            loading_block = ""
            if all_loading_instructions:
                loading_block = (
                    "\n\n---\n"
                    "**📊 HOW TO LOAD DATA FROM ALL ANALYST COPIES (using pandas):**\n\n"
                    + "\n".join(all_loading_instructions)
                    + "---\n"
                )

            warning = (
                "\n\n### ⚠️ Multi-Analyst Validation\n"
                "These results were generated by independent analyst instances running in parallel. "
                "Please carefully scrutinize the evidence provided by each replicate for inconsistencies, "
                "hallucinations, or divergent interpretations of the data."
            )

            # Add loading instructions at TOP and BOTTOM
            combined_results = "\n\n---\n\n".join(results)
            return f"{header}{loading_block}\n\n{combined_results}{loading_block}{warning}"

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

        with self._lock:
            if len(self.analysts) > 0:
                for analyst in self.analysts:
                    if analyst.name == name and analyst.goal == question:
                        if analyst.answer is None:
                            new_analyst = analyst
                        else:
                            # Build loading instructions for existing analyst
                            collections = self._get_analyst_collections(analyst.name)
                            loading_instructions = self._build_loading_instructions(analyst.name, collections)

                            return (
                                loading_instructions
                                + "Response from "
                                + analyst.name
                                + ":\n"
                                + analyst.answer
                                + "\nEvidence provided by "
                                + analyst.name
                                + ":\n"
                                + analyst.evidence
                                + loading_instructions
                            )
            if not new_analyst:
                new_analyst = Analyst(self.db, name=name, goal=question, require_file_output=require_file_output)
                self.analysts.append(new_analyst)

        new_analyst.pursue_goal()

        # Build loading instructions for this analyst
        collections = self._get_analyst_collections(name)
        loading_instructions = self._build_loading_instructions(name, collections)

        return (
            loading_instructions
            + "Response from "
            + name
            + ":\n"
            + new_analyst.answer
            + "\nEvidence provided by "
            + name
            + ":\n"
            + new_analyst.evidence
            + loading_instructions
        )

    def get_analyst_data_link(self, analyst_name, data_collection_name, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "get_analyst_data_link",
                    "description": "Generate a download link for a data extraction file previously created by an Analyst.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analyst_name": {
                                "type": "string",
                                "description": "Name of the analyst who created the collection",
                            },
                            "data_collection_name": {"type": "string", "description": "Name of the data extraction"},
                        },
                        "additionalProperties": False,
                        "required": ["analyst_name", "data_collection_name"],
                    },
                },
            }
        try:
            csv_path = self.db.convert_analyst_tool_tracker(analyst_name, data_collection_name)
            html_snippet = f'<div class="icon-container-box-image"><div class="icon-container-csv-image"></div><div class="button-icon-menu"><button class="icon-button" onclick="viewCSV(\'download/{csv_path}\')">👁️</button><a href="/download/{csv_path}?attached=T" class="icon-button">📥</a></div></div>'
            return html_snippet
        except Exception as e:
            return f"Could not generate link for collection '{data_collection_name}' from analyst '{analyst_name}'. Reason: {e!s}"

    def run_python_code(self, code, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "run_python_code",
                    "description": "Executes Python code in a STATELESS environment. "
                    "CRITICAL: Each call starts completely fresh - variables, imports, and data do NOT persist between calls. "
                    "ALWAYS start with: import pandas as pd; import numpy as np. "
                    "To load analyst data: filename = load_analyst_data('Analyst Name', 'collection_name'); df = pd.read_csv(filename). "
                    "The load_analyst_data() helper is pre-defined - do NOT import it. "
                    "Use this for math, statistics, plotting, and creating files. "
                    "IMPORTANT: Your code runs in a dedicated workspace directory. Save files using RELATIVE paths only "
                    "(e.g., 'output.csv', 'plot.png'). Do NOT use 'pi_generated/' as a prefix - you are ALREADY in that directory. "
                    "Files created will be automatically detected and made available for download. "
                    "Standard output and errors are captured and returned.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "The Python code to execute."}},
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                },
            }

        # Setup workspace
        workspace_dir = os.path.join(self.db.project_path, "pi_generated")
        os.makedirs(workspace_dir, exist_ok=True)

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Track files before execution
        existing_files = set(os.listdir(workspace_dir))

        # Execution environment
        # We'll give it access to pandas, numpy, matplotlib, etc. if installed
        # and a way to save files to the workspace

        # Helper to show plots if matplotlib is used
        def show_plot():
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if plt.get_fignums():
                filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename)
                plt.close()
                print(f"Plot saved to {filename}")

        # Helper to load analyst data
        def load_analyst_data(analyst_name, collection_name):
            """
            Load CSV data collected by an analyst into the current workspace.

            This function retrieves data that was extracted by one of your analyst
            delegations and copies it to your working directory for analysis.

            Args:
                analyst_name: The name of the analyst who collected the data
                             (e.g., 'Outcome Scout Analyst', 'Sample Size Analyst copy 1')
                collection_name: The name of the data collection created by the analyst
                                (e.g., 'relevant_text', 'data_availability', 'extraction_data')

            Returns:
                str: The local filename of the CSV file (e.g., 'extraction_data.csv'),
                     which can then be loaded with pd.read_csv(filename).
                     Returns None if the data could not be found.

            Example:
                filename = load_analyst_data('Outcome Scout Analyst', 'data_availability')
                df = pd.read_csv(filename)
            """
            try:
                # Use existing DB logic to get the CSV path
                src_path = self.db.convert_analyst_tool_tracker(analyst_name, collection_name)
                if not src_path or not os.path.exists(src_path):
                    print(f"Error: Could not find data for {analyst_name} - {collection_name}")
                    return None

                # Copy to workspace
                filename = os.path.basename(src_path)
                dst_path = os.path.join(workspace_dir, filename)
                import shutil

                shutil.copy(src_path, dst_path)
                print(f"Loaded data to {filename}")
                return filename
            except Exception as e:  # nosec
                print(f"Error loading data: {e}")
                return None

        env = {
            "print": lambda *args, **kwargs: print(*args, file=stdout_capture, **kwargs),
            "show_plot": show_plot,
            "load_analyst_data": load_analyst_data,
        }

        # Execute code
        original_cwd = os.getcwd()
        try:
            os.chdir(workspace_dir)
            # Force Agg backend to avoid main thread issues on macOS
            import matplotlib

            matplotlib.use("Agg")

            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(code, env)  # nosec B102
        except Exception:
            traceback.print_exc(file=stderr_capture)
        finally:
            os.chdir(original_cwd)

        output = stdout_capture.getvalue()
        error = stderr_capture.getvalue()

        # Detect new files
        current_files = set(os.listdir(workspace_dir))
        new_files = current_files - existing_files

        result_msg = ""
        if output:
            result_msg += f"Output:\n{output}\n"
        if error:
            result_msg += f"Errors:\n{error}\n"

        if new_files:
            result_msg += "\n<div class='pi-generated-content'>\n"
            result_msg += "<h4>Generated Files:</h4>\n"
            for filename in new_files:
                file_path = os.path.join(workspace_dir, filename)
                # Generate HTML based on file type
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                    # Image - render it
                    result_msg += f"<div class='pi-image-container'><img src='/download/{file_path}' alt='{filename}'/><div class='pi-image-caption'>{filename}</div></div>\n"
                    result_msg += f"<a href='/download/{file_path}' class='pi-download-link' download>📥 Download {filename}</a>\n"
                elif filename.lower().endswith(".csv"):
                    # CSV - view/download buttons
                    result_msg += f"<div>Created CSV: {filename}</div>\n"
                    result_msg += f'<div class="icon-container-box-image"><div class="icon-container-csv-image"></div><div class="button-icon-menu"><button class="icon-button" onclick="viewCSV(\'download/{file_path}\')">👁️</button><a href="/download/{file_path}?attached=T" class="icon-button">📥</a></div></div>\n'
                elif filename.lower().endswith(".html"):
                    # HTML - link to view
                    result_msg += f"<div>Created HTML: {filename}</div>\n"
                    result_msg += f"<a href='/download/{file_path}' target='_blank' class='pi-download-link'>View {filename}</a>\n"
                else:
                    # Generic download link
                    result_msg += f"<div>Created file: {filename}</div>\n"
                    result_msg += f"<a href='/download/{file_path}' class='pi-download-link' download>📥 Download {filename}</a>\n"
            result_msg += "</div>\n"

        if not result_msg:
            result_msg = "Code executed successfully (no output)."

        return result_msg

    def view_image(self, filename, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "view_image",
                    "description": "View and analyze an image file that you generated. "
                    "MANDATORY: You MUST call this immediately after generating ANY plot or image. "
                    "This allows you to visually inspect the image quality, clarity, labels, colors, and overall presentation "
                    "before sharing it with the user. Do NOT share images without viewing them first.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "The name of the image file to view (e.g., 'plot_20231128_143022.png')",
                            }
                        },
                        "required": ["filename"],
                        "additionalProperties": False,
                    },
                },
            }

        # Locate the image file in the workspace
        workspace_dir = os.path.join(self.db.project_path, "pi_generated")
        image_path = os.path.join(workspace_dir, filename)

        if not os.path.exists(image_path):
            return f"Error: Image file '{filename}' not found in workspace. Available files: {', '.join(os.listdir(workspace_dir)) if os.path.exists(workspace_dir) else 'none'}"

        try:
            import base64

            from PIL import Image

            # Load the image and get metadata
            img = Image.open(image_path)
            width, height = img.size
            format_name = img.format
            file_size = os.path.getsize(image_path)

            # Encode image as base64 for vision model
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            # SIDE-CHANNEL ANALYSIS:
            # Instead of returning the base64 string to the main chat (which explodes context window),
            # we send it to the vision model here and return ONLY the text critique.

            logger.info(
                f"Analyzing image {filename} ({width}x{height}, {round(file_size / 1024)}KB) with vision model..."
            )

            vision_messages = [
                {
                    "role": "system",
                    "content": "You are an expert data visualization critic. Analyze the provided image for quality, clarity, label readability, color usage, and overall effectiveness. Be concise but critical. Identify any issues that need fixing (e.g., overlapping text, missing titles, poor contrast).",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Please analyze this generated image ({filename})."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{format_name.lower()};base64,{img_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ]

            # Call the vision model
            vision_response = client.chat.completions.create(
                model=get_model_for_role(MODEL_VISION), messages=vision_messages, max_tokens=500
            )

            critique = vision_response.choices[0].message.content

            # Return structured data with the CRITIQUE, not the image data
            response = f"Image: {filename}\n"
            response += f"Dimensions: {width}x{height}\n"
            response += f"Format: {format_name}\n"
            response += f"Size: {round(file_size / 1024, 2)} KB\n\n"
            response += "### Vision Model Analysis:\n"
            response += critique

            return response

        except Exception as e:
            return f"Error viewing image '{filename}': {e!s}"

    def reflect_on_delegations(self, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "reflect_on_delegations",
                    "description": "Reflect on the entire conversation history to identify issues with data, "
                    "suggest helpful calculations, or provide additional insights. "
                    "Use this when you want a second opinion on the analysis so far. "
                    "Takes no parameters.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False, "required": []},
                },
            }
        result = add_reasoning_to_context(self.db.get_database_chat())
        if result:
            return result
        return "Delegation reflected upon."

    def validate_analytic_bundle(self, zip_path):
        """
        Validate an analytic bundle before delivery.

        Runs an AI agent that checks for:
        - Failed extractions that aren't properly documented
        - Outcome directionality issues (inverted outcomes)
        - Data dictionary completeness
        - Code errors
        - README completeness
        - Image/figure quality

        Args:
            zip_path: Path to the bundle zip file

        Returns:
            Detailed validation feedback with pass/fail status
        """
        # Try to find the zip file - check both the given path and pi_generated
        if os.path.exists(zip_path):
            resolved_path = zip_path
        else:
            # Try pi_generated folder
            workspace_dir = os.path.join(self.db.project_path, "pi_generated")
            pi_generated_path = os.path.join(workspace_dir, zip_path)
            if os.path.exists(pi_generated_path):
                resolved_path = pi_generated_path
            else:
                # Neither exists - let validate_bundle handle the error
                resolved_path = zip_path

        # validate_bundle now returns a string directly (agent-based)
        return validate_bundle(resolved_path)

    def mark_analyst_result_analyzed(
        self,
        analyst_name: str,
        investigation_summary: str,
        normalization_applied: str = "",
        copies_reconciled: list | None = None,
        return_tool: bool = False,
    ):
        """
        Mark analyst results as analyzed and trigger auto-compaction.

        This tool should be called AFTER:
        1. Loading analyst data with run_python_code
        2. Investigating the data for quality/consistency
        3. Normalizing data (outcome direction, group ordering, units)
        4. Reconciling any parallel copies (if applicable)

        Args:
            analyst_name: Base name of the analyst (e.g., "Outcome Scout Analyst")
            investigation_summary: What was found and what was done with the results
            normalization_applied: Description of normalizations (e.g., "Inverted Paper X outcomes")
            copies_reconciled: List of copy names if parallel analysts were used

        Returns:
            Confirmation message; also triggers auto-compaction of related messages
        """
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "mark_analyst_result_analyzed",
                    "description": "Mark analyst results as analyzed after investigation. "
                    "Call ONCE per analyst delegation AFTER: (1) loading data with run_python_code, "
                    "(2) investigating data quality, (3) normalizing data, (4) reconciling parallel copies. "
                    "For parallel copies, use the BASE analyst name (without 'copy N') - all copies are handled together. "
                    "Triggers auto-compaction to free context. Do NOT call multiple times for the same analyst.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analyst_name": {
                                "type": "string",
                                "description": "Base analyst name. For parallel copies, use the base name without 'copy N' suffix.",
                            },
                            "investigation_summary": {
                                "type": "string",
                                "description": "What you found and how you used the results. This becomes the compressed record.",
                            },
                            "normalization_applied": {
                                "type": "string",
                                "description": "Data normalizations applied: outcome direction fixes, group swaps, unit conversions. "
                                "Leave empty if none needed.",
                            },
                            "copies_reconciled": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of copy analyst names if parallel analysts exist "
                                "(e.g., ['Analyst copy 1', 'Analyst copy 2']).",
                            },
                        },
                        "required": ["analyst_name", "investigation_summary"],
                        "additionalProperties": False,
                    },
                },
            }

        if copies_reconciled is None:
            copies_reconciled = []
        elif isinstance(copies_reconciled, str):
            # Handle case where model passes string instead of list
            import ast

            try:
                copies_reconciled = ast.literal_eval(copies_reconciled)
                if not isinstance(copies_reconciled, list):
                    copies_reconciled = [copies_reconciled]
            except (ValueError, SyntaxError):
                copies_reconciled = [copies_reconciled] if copies_reconciled else []

        # Auto-detect copies if not provided
        all_analyst_names = [analyst_name]
        if not copies_reconciled:
            # Check if parallel copies exist
            with self._lock:
                for analyst in self.analysts:
                    if analyst.name.startswith(f"{analyst_name} copy "):
                        copies_reconciled.append(analyst.name)
        all_analyst_names.extend(copies_reconciled)

        # Check if already analyzed - prevents duplicate calls
        messages = self.db.get_database_chat()
        for msg in messages:
            if msg.get("analyzed") and msg.get("analysis_notes", {}).get("analyst_name") == analyst_name:
                return (
                    f"❌ **ERROR**: Analyst '{analyst_name}' has ALREADY been marked as analyzed. "
                    "Do NOT call this tool again for the same analyst. "
                    "The delegation was already compacted and investigation notes were recorded."
                )

        # Find and compact the relevant messages in chat history
        from .backend import compact_analyst_interaction

        analysis_notes = {
            "analyst_name": analyst_name,
            "investigation_summary": investigation_summary,
            "normalization_applied": normalization_applied,
            "copies_reconciled": copies_reconciled,
            "all_analyst_names": all_analyst_names,
        }

        compacted = compact_analyst_interaction(self.db, analyst_name, analysis_notes)

        # Build result message
        result_parts = [f"✅ **Analyst results marked as analyzed**: {analyst_name}"]

        if copies_reconciled:
            result_parts.append(f"\n**Copies reconciled**: {', '.join(copies_reconciled)}")

        if normalization_applied:
            result_parts.append(f"\n**Normalizations applied**: {normalization_applied}")

        if compacted:
            result_parts.append("\n\n🗜️ Delegation and response messages have been compacted to free context space.")
        else:
            result_parts.append(
                "\n\n⚠️ Could not locate delegation messages to compact (they may already be compressed)."
            )

        result_parts.append(f"\n\n**Investigation summary recorded**:\n{investigation_summary}")

        return "".join(result_parts)

    def tool_callback(self, response, function_name=None):
        self.messages.append(response)
        self.db.update_last_chat("Processed")
        self.db.add_chat(
            response["content"],
            response["role"],
            "Pending",
            datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
            function_name=function_name,
        )

    @staticmethod
    def tool_error_callback(response):
        logger.error(f"Tool error: {response}")

    def _calculate_and_emit_context(self, messages):
        """Calculate token count for messages and emit context usage update."""
        try:
            # Calculate tokens for all messages
            total_tokens = 0
            for msg in messages:
                content = msg.get("content", "")
                if content:
                    total_tokens += len(enc.encode(str(content)))
                # Account for tool calls if present
                if msg.get("tool_calls"):
                    total_tokens += len(enc.encode(json.dumps(msg.get("tool_calls"))))

            # Add overhead for message structure (roughly 4 tokens per message)
            total_tokens += len(messages) * 4

            # Emit context update
            from .__main__ import emit_context

            emit_context(total_tokens)
        except Exception:  # nosec
            # Don't let context tracking break the main flow
            pass

    async def process_message(self, content, role, status, time, store_message=True):
        if status != "Pending":
            self.db.update_last_chat("Processed")
            return
        if role != "user":
            raise ValueError("Only new user messages can be processed")
        if store_message:
            chat_message = {"content": content, "role": role, "status": status, "time": time}
            self.db.add_chat(chat_message)
        called_tools = True
        first_tool_call = True  # Track if this is the first tool call in this process_message invocation
        loop_iteration = 0
        no_final_content = False
        while called_tools or no_final_content:
            no_final_content = True
            loop_iteration += 1

            # Check for pause request at the start of each iteration
            from .backend import clear_pause_request, is_pause_requested

            if is_pause_requested():
                # Mark all pending messages as processed
                messages = self.db.get_database_chat()
                for msg in messages:
                    if msg.get("status") == "Pending":
                        msg["status"] = "Processed"
                self.db.set_all_chat_messages(messages)

                # Emit pause complete event to frontend (no chat message - silent pause)
                from .__main__ import emit_pause_complete

                emit_pause_complete()

                clear_pause_request()
                logger.info("Processing paused by user request")
                return  # Exit processing loop

            called_tools = False
            # Get chat history, filtering out internal messages (like compression requests)
            chat_history = [m for m in self.db.get_database_chat() if not m.get("internal")]
            temp_messages = [{"content": self.system_message, "role": "system"}, *chat_history]

            # Calculate and emit context usage
            self._calculate_and_emit_context(temp_messages)

            arguments = {
                "messages": temp_messages,
                "model": get_config().default_model,
                "reasoning_effort": "high",
                "tools": self.tools,
                "parallel_tool_calls": False,
            }

            try:
                chat_response = client.chat.completions.create(**arguments)
            except Exception as e:
                error_str = str(e).lower()
                # Check for context length  or token limit errors across providers
                # OpenAI: "maximum context length"
                # Anthropic: "max_tokens", "prompt is too long"
                # Google: "invalid argument" (often for length), "resource exhausted"
                token_errors = [
                    "maximum context length",
                    "context_length",
                    "too long",
                    "max_tokens",
                    "budget_tokens",
                    "output blocked",  # Sometimes happens with length on Vertex
                    "400",  # Generic 400 often implies bad request due to length if not other things
                ]

                if "tool_use" in error_str and "tool_result" in error_str:
                    # Heal orphans and retry ONCE
                    logger.warning(f"Tool use mismatch detected (healing orphans): {e}")
                    temp_messages = self._heal_tool_mismatches(temp_messages)
                    # Retry carefully
                    try:
                        arguments["messages"] = temp_messages
                        chat_response = client.chat.completions.create(**arguments)
                    except Exception as retry_e:
                        logger.error(f"Retry failed after healing: {retry_e}")
                        raise retry_e
                elif any(err in error_str for err in token_errors):
                    logger.warning(f"Context limit or token error reached: {e}")
                    # Add a special message that the frontend will detect
                    context_limit_message = {
                        "content": "CONTEXTLIMITREACHED",
                        "role": "system",
                        "status": "Processed",
                        "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
                    }
                    self.db.add_chat(context_limit_message)
                    self.db.update_last_chat("Processed")
                    return  # Exit processing
                else:
                    raise

            if chat_response.choices[0].message.tool_calls:
                called_tools = True

            if chat_response.choices[0].message.content and not called_tools:
                chat_message = {
                    "content": chat_response.choices[0].message.content,
                    "role": "assistant",
                    "status": "Pending",
                    "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
                }
                if chat_response.choices[0].message.thinking:
                    chat_message["thinking"] = chat_response.choices[0].message.thinking
                self.db.add_chat(chat_message)
                no_final_content = False
            elif (chat_response.choices[0].message.content and called_tools) or (
                not chat_response.choices[0].message.content and not called_tools
            ):
                no_final_content = True

            if called_tools:
                call_new_history = use_tools(
                    chat_response, arguments, function_dict=self.tool_callables, pre_tool_call=True
                )
                # added_csv = False
                # called_tools = False # Allow looping!
                is_data_link_request = False

                # Check if this is a get_analyst_data_link call
                for call in call_new_history:
                    if call["role"] == "assistant" and call.get("tool_calls"):
                        if call["tool_calls"][0]["function"]["name"] == "get_analyst_data_link":
                            is_data_link_request = True
                            break

                # Handle get_analyst_data_link specially - just show the button
                if is_data_link_request:
                    self.db.update_last_chat("Processed")
                    new_history = use_tools(chat_response, arguments, function_dict=self.tool_callables)
                    for call in new_history:
                        if call["role"] == "tool":
                            # Just add the button HTML as a simple assistant message
                            button_message = {
                                "content": call["content"],
                                "role": "assistant",
                                "status": "Pending",
                                "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
                            }
                            self.db.add_chat(button_message)
                    continue

                # Normal tool call handling for other tools
                for call in call_new_history:
                    if call["role"] == "assistant":
                        if first_tool_call:
                            # Mark user's message as processed when we start working
                            self.db.update_last_chat("Processed")
                            first_tool_call = False
                        call["status"] = "Pending"
                        call["time"] = datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z")
                        if not call["content"]:
                            call["content"] = "Working on that now..."
                        self.db.add_chat(call)

                new_history = use_tools(chat_response, arguments, function_dict=self.tool_callables)

                for call in new_history:
                    if call["role"] != "assistant":
                        call["status"] = "Pending"
                        call["time"] = datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z")
                        self.db.add_chat(call)

                        # Skip continuing the loop for run_python_code to allow auto-fixing
                        if call.get("name") == "run_python_code":
                            continue

                        # For all other tools, the loop will naturally continue
                        # The PI will keep calling tools until it responds with text (no tools)
        # Mark all pending messages as processed now that PI is done
        sleep(0.025)
        messages = self.db.get_database_chat()
        for msg in messages:
            if msg.get("status") == "Pending":
                msg["status"] = "Processed"
        # Update the chat in the database
        self.db.set_all_chat_messages(messages)

    async def finish_tool_calls(self, last_chat):
        new_history = use_tools(
            last_chat,
            {
                "messages": self.db.get_database_chat(),
                "model": get_config().default_model,
                "reasoning_effort": "medium",
                "tools": self.tools,
            },
            function_dict=self.tool_callables,
        )
        for call in new_history:
            if call["role"] != "assistant":
                self.db.update_last_chat("Processed")
                call["status"] = "Pending"
                call["time"] = datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z")
                self.db.add_chat(call)
        self.db.update_last_chat("Processed")

    def _heal_tool_mismatches(self, messages):
        """
        Scans the message history and removes 'assistant' messages that have tool_calls
        but are missing their corresponding 'tool' result messages (orphans).
        This is necessary because some LLM providers (e.g. Anthropic) enforce strict
        alternation of tool_use and tool_result.
        """
        healed_messages = []
        skip_indices = set()

        for i, msg in enumerate(messages):
            if i in skip_indices:
                continue

            # If it's an assistant message with tool calls
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                # Look ahead for corresponding tool results
                # We expect the NEXT N messages to be tool results for these calls
                # or interspersed if parallel, but typically they follow immediately.

                # Simple heuristic: Check if IMMEDIATE next messages cover these IDs.
                # If we have 3 tool calls, we expect 3 tool messages eventually?
                # Strict strict check: The sequence must be complete.

                required_ids = {tc["id"] for tc in tool_calls}
                found_ids = set()

                # Look ahead until we hit a non-tool message or run out
                offset = 1
                possible_result_indices = []

                while i + offset < len(messages):
                    next_msg = messages[i + offset]
                    if next_msg.get("role") == "tool":
                        possible_result_indices.append(i + offset)
                        # Extract tool_call_id. Standard format variants:
                        tid = next_msg.get("tool_call_id")
                        if not tid and "tool_use_id" in str(next_msg):
                            # Fallback for weird stored formats if any
                            pass

                        if tid:
                            found_ids.add(tid)
                    else:
                        break  # End of tool result block

                    offset += 1

                # Check coverage
                if not required_ids.issubset(found_ids):
                    logger.warning(
                        f"Found orphaned tool calls in message {i}. Expected {required_ids}, found {found_ids}. Removing this interaction."
                    )
                    # Mark this assistant message AND the partial tool results for skipping
                    skip_indices.add(i)
                    for idx in possible_result_indices:
                        skip_indices.add(idx)
                    continue

            healed_messages.append(msg)

        return healed_messages
