from datetime import datetime
from scienceai.analyst import Analyst
from .database_manager import DatabaseManager
from .llm import client, use_tools_sync as use_tools, enc
from .reasoning import add_reasoning_to_context
from .bundle_validator import validate_bundle, validate_bundle_tool_schema
import openai
import os
import io
import contextlib
import traceback
import sys
import shutil
import json
from time import sleep



path_to_app = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(path_to_app, "principal_investigator_base_prompt.txt"), "r") as file:
    system_message = file.read()


# PI Module
class PrincipalInvestigator:
    def __init__(self, dbr: DatabaseManager):
        self.db = dbr
        self.analysts = []
        analysts_db = dbr.get_all_analysts()
        for analyst_dict in analysts_db:
            self.analysts.append(Analyst(dbr, analyst_dict=analyst_dict))
        self.tool_callables = {
            "delegate_research": self.delegate_research,
            "reflect_on_delegations": self.reflect_on_delegations,
            "create_arbitrary_csv": self.create_arbitrary_csv,
            "get_analyst_data_link": self.get_analyst_data_link,
            "view_image": self.view_image,
            "validate_analytic_bundle": self.validate_analytic_bundle
        }
        self.tools = [
            self.delegate_research_schema(), 
            self.reflect_on_delegations(return_tool=True),
            self.create_arbitrary_csv(None, None, return_tool=True),
            self.get_analyst_data_link(None, None, return_tool=True),
            self.run_python_code(None, return_tool=True),
            self.view_image(None, return_tool=True),
            validate_bundle_tool_schema()
        ]
        self.tool_callables["run_python_code"] = self.run_python_code
        self.system_message = system_message

    async def initialize(self, ingest=True):
        chat_db = self.db.get_database_chat()
        first_message = ("Hello, I am ScienceAI. I first need to make sure all your papers are loaded into the system "
                         "before I can help you. I will let you know when I am ready to answer your questions. "
                         "This may take a long time if you uploaded many papers.")
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
            if last_chat["content"] == first_message:
                if ingest:
                    self.db.update_last_chat("Pending")
                    self.db.ingest_papers()
                    await self.db.process_all_papers()
                    self.db.update_last_chat("Processed")
                    second = {"content": get_second_message_with_count(), "role": "system", "status": "Pending",
                              "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')}
                    self.db.add_chat(second)
                    self.db.update_last_chat("Processed")
            elif last_chat["content"] == second_message_base or last_chat["content"].startswith("All ") and "papers have been loaded" in last_chat["content"]:
                self.db.update_last_chat("Processed")
            else:
                if last_chat["status"] == "Pending":
                    if last_chat.get("tool_calls"):
                        await self.finish_tool_calls(last_chat)
                    elif last_chat["role"] == "user":
                        await self.process_message(last_chat["content"], last_chat["role"], last_chat["status"], last_chat["time"],
                                             store_message=False)
        else:
            first = {"content": first_message, "role": "system", "status": "Pending",
                     "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')}
            self.db.add_chat(first)
            if ingest:
                self.db.ingest_papers()
                await self.db.process_all_papers()
                self.db.update_last_chat("Processed")
                second = {"content": get_second_message_with_count(), "role": "system", "status": "Pending",
                          "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')}
                self.db.add_chat(second)
                self.db.update_last_chat("Processed")
        self.db.update_last_chat("Processed")
        messages = self.db.get_database_chat()
        for msg in messages:
            if msg.get("status") == "Pending":
                msg["status"] = "Processed"
        # Update the chat in the database
        from .database_manager import DDB
        with DDB.at("chat").session() as (session, chat):
            chat["messages"] = messages
            session.write()

    def delegate_research_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "delegate_research",
                "description": "Delegate data extraction from research papers to a specialized Analyst Agent. "
                               "The Analyst will extract structured data and optionally create CSV files. "
                               "Use this when you need NEW information from papers that hasn't been collected yet. "
                               "Returns answer and evidence from the Analyst after completion.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Descriptive name for this analyst (e.g., 'Sample Size Analyst', 'Methods Analyst'). "
                                           "Keep it concise but specific to the task. Used to track analyst's work."
                        },
                        "question": {
                        "type": "string",
                        "description": "The research question or data extraction goal. Be specific about WHAT to extract, "
                                       "not HOW to format it. **If you already know which specific papers to analyze** "
                                       "(e.g., from previous analyst results or your own analysis), include their paper IDs "
                                       "or titles directly in the question (e.g., 'For papers [abc123def4, xyz789ghi0], extract "
                                       "sample sizes' or 'From papers titled X, Y, Z, extract methods'). This ensures consistency "
                                       "and avoids forcing the analyst to rediscover your paper selection. Only use general "
                                       "descriptions (e.g., 'papers using qualitative methods') when you need the analyst to "
                                       "discover or filter papers themselves. Include what data points are needed and any "
                                       "specific details (like lists vs fixed counts)."
                    },
                        "require_file_output": {
                            "type": "boolean",
                            "description": "Set to true when you need downloadable CSV files with structured data (typically "
                                           "for 10+ papers or complex multi-field extractions). Set to false for quick queries "
                                           "or summary analyses. Default: false. When true, analyst MUST attach data collection files."
                        }
                    },
                    "required": ["name", "question"]
                }
            }
        }

    def delegate_research(self, name, question, require_file_output=False):
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
            new_analyst = Analyst(self.db, name=name, goal=question, require_file_output=require_file_output)
            self.analysts.append(new_analyst)
        new_analyst.pursue_goal()
        return ("Response from " + name + ":\n" + new_analyst.answer +
                "\nEvidence provided by " + name + ":\n" + new_analyst.evidence)
    
    def create_arbitrary_csv(self, csv_name, csv_str, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "create_arbitrary_csv",
                    "description": "Creates a CSV file with the given name and data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "csv_name": {"type": "string", "description": "The name of the CSV file"},
                            "csv_str": {"type": "string", "description": "The data to be written to the CSV file"}
                        },
                        "additionalProperties": False,
                        "required": ["csv_name", "csv_str"]
                    }
                }
            }
        self.db.create_pi_arbitrary_csv(csv_name, csv_str)
        csv_path = self.db.get_pi_arbitrary_csv(csv_name)
        return f"CSV file {csv_name} created successfully.  <button type='submit' onclick='window.open(\"download/{csv_path}\")'>Download CSV</button>"

    def get_analyst_data_link(self, analyst_name, data_collection_name, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "get_analyst_data_link",
                    "description": "Generate a download link for a data collection file previously created by an Analyst.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analyst_name": {"type": "string", "description": "Name of the analyst who created the collection"},
                            "data_collection_name": {"type": "string", "description": "Name of the data collection"}
                        },
                        "additionalProperties": False,
                        "required": ["analyst_name", "data_collection_name"]
                    }
                }
            }
        try:
            csv_path = self.db.convert_analyst_tool_tracker(analyst_name, data_collection_name)
            html_snippet = f"<div class=\"icon-container-box-image\"><div class=\"icon-container-csv-image\"></div><div class=\"button-icon-menu\"><button class=\"icon-button\" onclick=\"viewCSV('download/{csv_path}')\"><i class=\"fa fa-eye\"></i></button><a href=\"/download/{csv_path}?attached=T\" class=\"icon-button\"><i class=\"fas fa-download\"></i></a></div></div>"
            return html_snippet
        except Exception as e:
            return f"Could not generate link for collection '{data_collection_name}' from analyst '{analyst_name}'. Reason: {str(e)}"

    def run_python_code(self, code, return_tool=False):
        if return_tool:
            return {
                "type": "function",
                "function": {
                    "name": "run_python_code",
                    "description": "Executes Python code in a restricted environment. Use this for math, statistics, plotting, and creating files. "
                                   "Files created in the current directory will be automatically detected and made available for download. "
                                   "Standard output and errors are captured and returned.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute."
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False
                    }
                }
            }
        
        # Setup workspace
        workspace_dir = os.path.join(self.db.project_path, "pi_generated")
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)
            
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
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            if plt.get_fignums():
                filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename)
                plt.close()
                print(f"Plot saved to {filename}")

        # Helper to load analyst data
        def load_analyst_data(analyst_name, collection_name):
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
            except Exception as e:
                print(f"Error loading data: {e}")
                return None

        env = {
            "print": lambda *args, **kwargs: print(*args, file=stdout_capture, **kwargs),
            "show_plot": show_plot,
            "load_analyst_data": load_analyst_data
        }
        
        # Execute code
        original_cwd = os.getcwd()
        try:
            os.chdir(workspace_dir)
            # Force Agg backend to avoid main thread issues on macOS
            import matplotlib
            matplotlib.use('Agg')
            
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(code, env)
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
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    # Image - render it
                    result_msg += f"<div class='pi-image-container'><img src='/download/{file_path}' alt='{filename}'/><div class='pi-image-caption'>{filename}</div></div>\n"
                    result_msg += f"<a href='/download/{file_path}' class='pi-download-link' download><i class='fas fa-download'></i> Download {filename}</a>\n"
                elif filename.lower().endswith('.csv'):
                    # CSV - view/download buttons
                    result_msg += f"<div>Created CSV: {filename}</div>\n"
                    result_msg += f"<div class=\"icon-container-box-image\"><div class=\"icon-container-csv-image\"></div><div class=\"button-icon-menu\"><button class=\"icon-button\" onclick=\"viewCSV('download/{file_path}')\"><i class=\"fa fa-eye\"></i></button><a href=\"/download/{file_path}?attached=T\" class=\"icon-button\"><i class=\"fas fa-download\"></i></a></div></div>\n"
                elif filename.lower().endswith('.html'):
                    # HTML - link to view
                    result_msg += f"<div>Created HTML: {filename}</div>\n"
                    result_msg += f"<a href='/download/{file_path}' target='_blank' class='pi-download-link'>View {filename}</a>\n"
                else:
                    # Generic download link
                    result_msg += f"<div>Created file: {filename}</div>\n"
                    result_msg += f"<a href='/download/{file_path}' class='pi-download-link' download><i class='fas fa-download'></i> Download {filename}</a>\n"
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
                    "description": "View and analyze an image file that you generated. This allows you to visually inspect "
                                   "the image quality, clarity, labels, colors, and overall presentation before sharing it "
                                   "with the user. Use this IMMEDIATELY after generating any image to ensure it meets quality standards.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "The name of the image file to view (e.g., 'plot_20231128_143022.png')"
                            }
                        },
                        "required": ["filename"],
                        "additionalProperties": False
                    }
                }
            }
        
        # Locate the image file in the workspace
        workspace_dir = os.path.join(self.db.project_path, "pi_generated")
        image_path = os.path.join(workspace_dir, filename)
        
        if not os.path.exists(image_path):
            return f"Error: Image file '{filename}' not found in workspace. Available files: {', '.join(os.listdir(workspace_dir)) if os.path.exists(workspace_dir) else 'none'}"
        
        try:
            from PIL import Image
            import base64
            
            # Load the image and get metadata
            img = Image.open(image_path)
            width, height = img.size
            format_name = img.format
            file_size = os.path.getsize(image_path)
            
            # Encode image as base64 for vision model
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # SIDE-CHANNEL ANALYSIS:
            # Instead of returning the base64 string to the main chat (which explodes context window),
            # we send it to the vision model here and return ONLY the text critique.
            
            print(f"Analyzing image {filename} ({width}x{height}, {round(file_size/1024)}KB) with vision model...")
            
            vision_messages = [
                {
                    "role": "system",
                    "content": "You are an expert data visualization critic. Analyze the provided image for quality, clarity, label readability, color usage, and overall effectiveness. Be concise but critical. Identify any issues that need fixing (e.g., overlapping text, missing titles, poor contrast)."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Please analyze this generated image ({filename})."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{format_name.lower()};base64,{img_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Call the vision model (using gpt-4o or compatible vision model)
            vision_response = client.chat.completions.create(
                model="gpt-4o",
                messages=vision_messages,
                max_tokens=500
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
            return f"Error viewing image '{filename}': {str(e)}"

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
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                        "required": []
                    }
                }
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
        except Exception as e:
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
            called_tools = False
            # Get chat history, filtering out internal messages (like compression requests)
            chat_history = [m for m in self.db.get_database_chat() if not m.get("internal")]
            temp_messages = [{"content": self.system_message, "role": "system"}] + chat_history
            
            # Calculate and emit context usage
            self._calculate_and_emit_context(temp_messages)
            
            arguments = {"messages": temp_messages, "model": "gpt-5.1", 'reasoning_effort': 'high', "tools": self.tools, "parallel_tool_calls": False}
            
            # Try to make API call, catch context limit errors
            try:
                chat_response = client.chat.completions.create(**arguments)
            except openai.BadRequestError as e:
                error_str = str(e)
                # Check for context length errors
                if 'maximum context length' in error_str.lower():
                    # Add a special message that the frontend will detect
                    context_limit_message = {
                        "content": "CONTEXT_LIMIT_REACHED",
                        "role": "system",
                        "status": "Processed",
                        "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z'),
                        "context_limit_exceeded": True
                    }
                    self.db.add_chat(context_limit_message)
                    self.db.update_last_chat("Processed")
                    return  # Exit processing
                else:
                    raise

            if chat_response.choices[0].message.tool_calls:
                called_tools = True

            if chat_response.choices[0].message.content and not called_tools:
                chat_message = {"content": chat_response.choices[0].message.content, "role": "assistant",
                                "status": "Pending", "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')}
                self.db.add_chat(chat_message)
                no_final_content = False
            elif chat_response.choices[0].message.content and called_tools:
                no_final_content = True
            elif not chat_response.choices[0].message.content and not called_tools:
                no_final_content = True
                
            if called_tools:
                call_new_history = use_tools(chat_response, arguments, function_dict=self.tool_callables, pre_tool_call=True)
                added_csv = False
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
                                "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
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
                        call["time"] = datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
                        
                        if call["tool_calls"][0]["function"]["name"] == "create_arbitrary_csv":
                            added_csv = True
                        if not call["content"]:
                            call["content"] = "Working on that now..."
                        self.db.add_chat(call)

                new_history = use_tools(chat_response, arguments, function_dict=self.tool_callables)

                last_csv = None

                for call in new_history:
                    if call["role"] != "assistant":
                        call["status"] = "Pending"
                        call["time"] = datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
                        self.db.add_chat(call)
                        
                        if added_csv:
                            # Store the CSV content for potential use
                            last_csv = call["content"]
                        
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
        from .database_manager import DDB
        with DDB.at("chat").session() as (session, chat):
            chat["messages"] = messages
            session.write()

    async def finish_tool_calls(self, last_chat):
        new_history = use_tools(last_chat, {"messages": self.db.get_database_chat(), "model": "gpt-5.1", 'reasoning_effort': 'medium',
                                            "tools": self.tools}, function_dict=self.tool_callables)
        for call in new_history:
            if call["role"] != "assistant":
                self.db.update_last_chat("Processed")
                call["status"] = "Pending"
                call["time"] = datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
                self.db.add_chat(call)
        self.db.update_last_chat("Processed")



