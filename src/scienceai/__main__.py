import asyncio
import atexit
import builtins
import contextlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib
import uuid
import zipfile
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path

import markdown2
from flask import Flask, abort, after_this_request, render_template
from flask_sock import Sock

from .backend import run_backend
from .database_manager import DatabaseManager, get_projects

# Compiled regex pattern for common HTML tags
COMMON_HTML_TAG_PATTERN = re.compile(
    r"^</?(?:a|abbr|address|area|article|aside|audio|b|base|bdi|bdo|blockquote|body|br|button|"
    r"canvas|caption|cite|code|col|colgroup|data|datalist|dd|del|details|dfn|dialog|div|dl|dt|"
    r"em|embed|fieldset|figcaption|figure|footer|form|h1|h2|h3|h4|h5|h6|head|header|hgroup|hr|"
    r"html|i|iframe|img|input|ins|kbd|label|legend|li|link|main|map|mark|meta|meter|nav|noscript|"
    r"object|ol|optgroup|option|output|p|param|picture|pre|progress|q|rp|rt|ruby|s|samp|script|"
    r"section|select|small|source|span|strong|style|sub|summary|sup|svg|table|tbody|td|template|"
    r"textarea|tfoot|th|thead|time|title|tr|track|u|ul|var|video|wbr)(?:\s|>|/)",
    re.IGNORECASE,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
sock = Sock(app)
logger = logging.getLogger(__name__)

own_pid = os.getpid()
database = None
stop_event = None
message_queue = None
thread = None
original_save = None

# Progress tracking globals
progress_connections = []  # WebSocket connections for progress updates
progress_lock = threading.Lock()
current_progress = None  # Store current progress state for new connections

# Context usage tracking globals
context_connections = []  # WebSocket connections for context updates
context_lock = threading.Lock()
current_context = None  # Store current context state for new connections


def safe_db_read(read_func):
    """
    Safely execute a database read function.
    Catches transient errors from dictdatabase lock race conditions and returns None.
    The caller should handle None gracefully (skip this iteration).
    NOTE: We don't retry because DDB's thread-based locking doesn't allow
    re-acquiring locks on the same thread.
    """
    try:
        return read_func()
    except FileNotFoundError as e:
        logger.warning(f"Database read failed (lock file race): {e}")
        return None
    except RuntimeError as e:
        # "Thread already has a read lock" - DDB locking issue
        logger.warning(f"Database read failed (lock state): {e}")
        return None
    except Exception as e:
        logger.error(f"Database read failed: {e}")
        return None


db_folder = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
if not os.path.exists(db_folder):
    os.makedirs(db_folder)
path_to_app = os.path.dirname(os.path.abspath(__file__))
path_to_python = sys.executable
script_to_return_to_menu = "<script>window.location.href = '/menu';</script>"


def emit_progress(current, total, description="Processing", analyst_name=None):
    """Emit progress update to all connected WebSocket clients"""
    global progress_connections, current_progress
    message_data = {"current": current, "total": total, "description": description}
    if analyst_name:
        message_data["analyst_name"] = analyst_name

    message = json.dumps(message_data)
    logger.debug(message)

    # Store current progress state for new connections
    with progress_lock:
        current_progress = message
        for ws in progress_connections[:]:  # Copy list to avoid modification during iteration
            try:
                ws.send(message)
            except Exception:
                # Suppress errors during WebSocket send to avoid crashing the loop
                progress_connections.remove(ws)

    # Clear progress state when complete
    if current >= total and total > 0:
        with progress_lock:
            current_progress = None


def emit_context(tokens_used, tokens_limit=None, can_compress=True):
    """Emit context usage update to all connected WebSocket clients"""
    global context_connections, current_context
    if tokens_limit is None:
        from .llm_providers import get_context_limit

        tokens_limit = get_context_limit()

    # Guard against division by zero (can happen during provider switch or cold start)
    if not tokens_limit or tokens_limit <= 0:
        percentage = 0
    else:
        percentage = min(100, round((tokens_used / tokens_limit) * 100, 1))
    message_data = {
        "type": "context",
        "tokens_used": tokens_used,
        "tokens_limit": tokens_limit,
        "percentage": percentage,
        "can_compress": can_compress,
    }

    message = json.dumps(message_data)

    # Store current context state for new connections
    with context_lock:
        current_context = message
        for ws in context_connections[:]:  # Copy list to avoid modification during iteration
            try:
                ws.send(message)
            except Exception:
                context_connections.remove(ws)


def emit_pause_complete():
    """Emit pause complete event to all connected WebSocket clients"""
    global context_connections
    message_data = {"type": "pause_complete"}
    message = json.dumps(message_data)

    with context_lock:
        for ws in context_connections[:]:
            try:
                ws.send(message)
            except Exception:
                context_connections.remove(ws)


def calculate_and_emit_context_from_messages(messages):
    """Calculate token count from chat messages and emit context usage. Returns percentage."""
    try:
        from .llm_providers import get_context_limit, get_provider

        provider = get_provider()
        total_tokens = provider.count_tokens(messages)

        # Skip emission if token counting failed (returns None)
        if total_tokens is None:
            logger.debug("Token counting returned None, skipping context emission")
            return None  # Return None to indicate no update

        # Check if compression is possible
        can_compress = can_compress_context(messages)

        emit_context(total_tokens, can_compress=can_compress)

        # Return the percentage for template use
        limit = get_context_limit()
        if limit and limit > 0:
            return min(100, round((total_tokens / limit) * 100, 1))
        return None  # Return None if limit is invalid
    except Exception as e:
        logger.warning(f"Context calculation failed: {e}")
        # Don't let context tracking break the main flow
        return None


def can_compress_context(messages):
    """Check if there are any uncompressed tool messages that can be compressed."""
    for msg in messages:
        # Skip already compressed messages
        if msg.get("compressed"):
            continue
        if msg.get("role") == "tool" or msg.get("tool_calls"):
            return True
    return False


def is_undo_blocked(messages):
    """
    Check if undoing the last user request is blocked because run_python_code was used.
    Returns True if undo is blocked, False otherwise.
    """
    # Find the last user message index
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return True  # No user message, nothing to undo

    # Check if run_python_code was used in any message after the last user message
    messages_after_user = messages[last_user_idx + 1 :]
    for msg in messages_after_user:
        tool_calls = msg.get("tool_calls", [])
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            func_name = func.get("name", "")
            if func_name == "run_python_code":
                return True

    return False


def close():
    global thread
    global stop_event
    global message_queue
    global database
    global original_save

    logger.info("close() called")

    if thread:
        logger.info(f"Thread exists (alive={thread.is_alive()}), sending TERMINATE...")
        message_queue.put({"TERMINATE": True})
        stop_event.set()
        logger.info("Waiting for graceful shutdown (10s timeout)...")
        thread.join(timeout=10)  # Increased from 3 to 10 seconds for graceful shutdown

        if thread.is_alive():
            # Thread didn't terminate gracefully
            # Try raising SystemExit in the thread - this allows context managers to cleanup
            import ctypes

            logger.info("Thread still alive after 10s, raising SystemExit in backend thread...")
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), ctypes.py_object(SystemExit))
            thread.join(timeout=5)  # Give it time to handle the exception

        if thread.is_alive():
            # SystemExit didn't work - wait for DB locks to confirm no writes in progress
            logger.warning("Thread still alive after SystemExit, checking DB locks...")
            if database and database.lock_project(timeout=10):
                # All DB locks acquired - no writes in progress, thread just stuck on HTTP
                logger.info("DB locks acquired, thread is stuck but DB is safe. Detaching thread...")
            else:
                # Couldn't get locks - wait a bit more
                logger.warning("Couldn't acquire all DB locks, waiting longer...")
            thread.join(timeout=100)

            if thread.is_alive():
                # Can't stop the thread - just detach it and let it die on its own
                logger.error(
                    "ScienceAI Project Thread is still alive after all attempts to stop it. Killing entire process because we can't stop it, restarting ScienceAI."
                )
                os._exit(1)
                # The thread will eventually finish when HTTP calls complete
        else:
            logger.info("Thread terminated successfully")
    else:
        logger.info("No thread to close (thread was None)")

    thread = None
    stop_event = None
    message_queue = None
    database = None
    original_save = None
    io_dir = os.path.join(path_to_app, "io")
    if os.path.exists(io_dir):
        for file in os.listdir(io_dir):
            os.remove(os.path.join(io_dir, file))

    logger.info("close() completed")


def sanitize_for_id(value):
    """Replace special characters that are problematic in CSS selectors with dashes."""
    # Replace all problematic characters with dashes
    value = re.sub(r'[/\s.&()\[\]{}<>:;,?!@#$%^*+=|\\~`\'"]+', "-", value)
    # Remove leading/trailing dashes and convert to lowercase
    return re.sub(r"^-+|-+$", "", value).lower()


app.jinja_env.filters["sanitize_for_id"] = sanitize_for_id
app.jinja_env.filters["quote_url"] = lambda u: urllib.parse.quote(u)


def convert_markdown(messages):
    # messages is a list of dictionaries with keys one of which is content - convert content from markdown to html
    cp_text_dict = []
    for message in messages:
        try:
            content = message["content"]

            # PASS 0: Pre-escape bare < and > that are clearly NOT HTML tags
            # This prevents the tag regex from matching huge spans like "< 0.0001 ... text ... >"
            # Escape < followed by space, digit, or = (math comparisons like "p < 0.0001")
            content = re.sub(r"<(?=[\s\d=])", "&lt;", content)
            # Escape > preceded by space or digit (math comparisons like "x > 5")
            content = re.sub(r"(?<=[\s\d])>", "&gt;", content)

            # PASS 1: Escape any < or > that are NOT part of valid HTML tags
            def escape_if_invalid(match):
                """Escape the tag if it's not a common HTML tag."""
                tag = match.group(0)
                # Check if it's a common HTML tag
                if COMMON_HTML_TAG_PATTERN.match(tag):
                    return tag  # Keep valid HTML tags as-is
                # Also check if it contains our file link classes (pi-file-link, pi-file-name, etc.)
                # This ensures HTML generated by replace_pi_generated_file_links is preserved
                if re.search(r"class=['\"]pi-file-(link|name|buttons)", tag, re.IGNORECASE):
                    return tag  # Keep our file link HTML as-is
                else:
                    return tag.replace("<", "&lt;").replace(">", "&gt;")  # Escape invalid tags

            # Find all <...> patterns and escape invalid ones
            content = re.sub(r"<[^>]+>", escape_if_invalid, content)

            # PASS 2: Store valid HTML tags with placeholders to protect from markdown processing
            # This prevents markdown from interpreting underscores in HTML attributes as emphasis
            html_blocks = []
            import uuid

            placeholder_prefix = str(uuid.uuid4()).replace("-", "")

            def store_html(match, html_blocks=html_blocks, placeholder_prefix=placeholder_prefix):
                """Store valid HTML tag and return placeholder."""
                tag = match.group(0)
                # At this point, only valid HTML tags remain (invalid ones are escaped)
                idx = len(html_blocks)
                html_blocks.append(tag)
                return f"HTMLBLOCK{placeholder_prefix}{idx}PLACEHOLDER"

            # Store all remaining HTML tags (these are all valid)
            content = re.sub(r"<[^>]+>", store_html, content)

            # Now process markdown (valid HTML is protected, invalid tags are already escaped)
            # Enable extras for proper list handling and code blocks
            content = markdown2.markdown(
                content,
                extras=[
                    "fenced-code-blocks",  # Support ```code``` blocks
                    "tables",  # Support markdown tables
                    "cuddled-lists",  # Allow lists without blank line before them
                    # NOTE: Do NOT use "break-on-newline" - it breaks nested list parsing
                ],
            )

            # Restore HTML blocks
            for i, html_block in enumerate(html_blocks):
                placeholder = f"HTMLBLOCK{placeholder_prefix}{i}PLACEHOLDER"
                content = re.sub(re.escape(placeholder), html_block, content)

            # Final cleanup: Remove any unmatched tags
            # Parse all tags and track which ones are matched
            def remove_unmatched_tags(html):
                """Remove unmatched opening or closing tags."""
                # Find all tags
                tag_pattern = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)[^>]*?(/?)>")
                tags = []

                for match in tag_pattern.finditer(html):
                    is_closing = bool(match.group(1))  # Is it </tag>?
                    tag_name = match.group(2).lower()
                    is_self_closing = bool(match.group(3)) or tag_name in {
                        "area",
                        "base",
                        "br",
                        "col",
                        "embed",
                        "hr",
                        "img",
                        "input",
                        "link",
                        "meta",
                        "param",
                        "source",
                        "track",
                        "wbr",
                    }

                    tags.append(
                        {
                            "match": match,
                            "is_closing": is_closing,
                            "is_self_closing": is_self_closing,
                            "tag_name": tag_name,
                            "start": match.start(),
                            "end": match.end(),
                            "matched": is_self_closing,  # Self-closing tags are always "matched"
                        }
                    )

                # Match opening and closing tags
                stack = []
                for tag in tags:
                    if tag["is_self_closing"]:
                        continue
                    elif not tag["is_closing"]:
                        # Opening tag
                        stack.append(tag)
                    else:
                        # Closing tag - find matching opening tag
                        # Look backwards in stack for the nearest matching opening tag
                        # This handles cases like <div><span>Text</div></span> where </div> matches <div>
                        # and <span> is left unmatched inside
                        for i in range(len(stack) - 1, -1, -1):
                            if stack[i]["tag_name"] == tag["tag_name"]:
                                # Found a match!
                                stack[i]["matched"] = True
                                tag["matched"] = True
                                # Everything between the matched opening tag and this closing tag
                                # that is still on the stack is effectively unmatched (improperly nested)
                                # But we leave them on the stack for now, they just won't be marked matched
                                stack.pop(i)
                                # We also need to remove everything that was AFTER the matched tag on the stack
                                # because those are now "orphaned" by this closure
                                # e.g. <div><span></div> -> <span> is orphaned
                                del stack[i:]
                                break
                        # If no match found, tag remains unmatched (matched=False from init)

                # Remove unmatched tags
                # Build result by removing unmatched tag positions
                unmatched_ranges = [(tag["start"], tag["end"]) for tag in tags if not tag["matched"]]

                if not unmatched_ranges:
                    return html

                # Sort by position and remove
                unmatched_ranges.sort(reverse=True)  # Remove from end to start to preserve positions
                result = html
                for start, end in unmatched_ranges:
                    result = result[:start] + result[end:]

                return result

            content = remove_unmatched_tags(content)

            message["content"] = content
        except Exception:  # nosec
            pass
        cp_text_dict.append(message)
    return cp_text_dict


def filter_intermediate_messages(messages):
    """
    Hide all but the last intermediate status messages like 'Working on that now...'
    and 'Reflecting on work now...' to keep the chat clean.
    """
    intermediate_phrases = ["Working on that now...", "Reflecting on work now..."]

    # Find indices of all intermediate messages
    intermediate_indices = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("content") in intermediate_phrases:
            intermediate_indices.append(i)

    # If there are intermediate messages, hide all but the last one (if it's still pending)
    if intermediate_indices:
        last_index = intermediate_indices[-1]
        last_msg = messages[last_index]

        # Only keep the last one visible if it's still pending
        for idx in intermediate_indices:
            if idx != last_index or last_msg.get("status") != "Pending":
                messages[idx]["hidden"] = True

    # Filter out hidden messages
    return [msg for msg in messages if not msg.get("hidden", False)]


def replace_pi_generated_file_links(messages, project_path):
    """
    Replace exact file path references to pi_generated folder files with clickable download/view links.

    Detects patterns like:
    - /full/path/to/project/pi_generated/filename.ext
    - pi_generated/filename.ext
    - Just the filename if it exists in pi_generated
    """
    if not project_path:
        return messages

    pi_generated_path = os.path.join(project_path, "pi_generated")

    if not os.path.exists(pi_generated_path):
        return messages

    # Get all files in pi_generated
    try:
        pi_files = os.listdir(pi_generated_path)
    except Exception:
        return messages

    if not pi_files:
        return messages

    # Build a mapping of all full paths to filenames using pathlib.Path no paths to dirs all paths to files but at all depths
    # this is wrong because it will not include the files in the subdirectories
    file_map = {}
    for path in Path(pi_generated_path).glob("**/*"):
        if path.is_file():
            raw_path = str(path)
            file_map[raw_path.replace((pi_generated_path + "/").replace("//", "/"), "")] = raw_path

    if not file_map:
        return messages

    # Process each message
    for message in messages:
        content = message.get("content", "")
        if not content or not isinstance(content, str):
            continue

        # Skip messages that already have file links (run_python_code output)
        if "pi-generated-content" in content:
            continue

        # Replace file references
        for filename, full_path in file_map.items():
            # Patterns to match:
            # 1. Full absolute path
            # 2. Relative path with pi_generated/
            # 3. Just the filename (word boundary)

            # Build replacement HTML based on file type
            ext = os.path.splitext(filename)[1].lower()

            if ext in [".csv"]:
                # CSV: view + download buttons
                replacement = (
                    f"<span class='pi-file-link'>"
                    f"<span class='pi-file-name'>{filename}</span>"
                    f"<span class='pi-file-buttons'>"
                    f"<button class='icon-button' onclick='viewCSV(\"download/{full_path}\")'>👁️</button>"
                    f"<a href='/download/{full_path}?attached=T' class='icon-button'>📥</a>"
                    f"</span></span>"
                )
            elif ext in [".json"]:
                # JSON: view + download buttons
                replacement = (
                    f"<span class='pi-file-link'>"
                    f"<span class='pi-file-name'>{filename}</span>"
                    f"<span class='pi-file-buttons'>"
                    f"<button class='icon-button' onclick='viewJSON(\"download/{full_path}\")'>👁️</button>"
                    f"<a href='/download/{full_path}?attached=T' class='icon-button'>📥</a>"
                    f"</span></span>"
                )
            elif ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                # Images: view inline + download
                replacement = (
                    f"<span class='pi-file-link'>"
                    f"<span class='pi-file-name'>{filename}</span>"
                    f"<span class='pi-file-buttons'>"
                    f"<a href='/download/{full_path}' target='_blank' class='icon-button'>👁️</a>"
                    f"<a href='/download/{full_path}?attached=T' class='icon-button'>📥</a>"
                    f"</span></span>"
                )
            else:
                # Other files: just download
                replacement = (
                    f"<span class='pi-file-link'>"
                    f"<span class='pi-file-name'>{filename}</span>"
                    f"<span class='pi-file-buttons'>"
                    f"<a href='/download/{full_path}?attached=T' class='icon-button'>📥</a>"
                    f"</span></span>"
                )

            # Replace full path references
            content = content.replace(full_path, replacement)

            # Replace relative path references (pi_generated/filename)
            relative_path = f"pi_generated/{filename}"
            content = content.replace(relative_path, replacement)

            # Replace just the filename (with word boundaries to avoid partial matches)
            # Use regex for word boundary matching
            # Only replace if the filename appears as a standalone reference (not already replaced)
            if replacement not in content:
                # Match filename surrounded by whitespace, punctuation, or HTML tags
                # But NOT if it's part of a URL or already in an HTML attribute
                # Make underscores optional to match text where underscores may have been omitted
                escaped_filename = re.escape(filename)
                # Replace escaped underscores with optional underscore/space pattern
                flexible_filename = escaped_filename.replace(r"\_", r"[_ ]?")
                pattern = re.compile(
                    r'(?<![a-zA-Z0-9_/"\'])' + flexible_filename + r'(?![a-zA-Z0-9_/"\'])', re.IGNORECASE
                )
                content = pattern.sub(replacement, content)

        message["content"] = content

    return messages


def load_project(project):
    global stop_event
    global database
    global message_queue
    global thread
    global original_save
    for file in os.listdir(os.path.join(path_to_app, "io")):
        os.remove(os.path.join(path_to_app, "io", file))
    if database:
        close()
        return False
    stop_event = threading.Event()
    message_queue = Queue()
    ingest_folder = os.path.join(db_folder, "scienceai_db", project, project.replace(" ", "_") + "_ingest_folder")
    if not os.path.exists(ingest_folder):
        return False
    thread = threading.Thread(target=run_backend, args=(ingest_folder, project, db_folder, message_queue, stop_event))
    thread.start()
    time.sleep(1)
    database = DatabaseManager(ingest_folder, None, project, storage_path=db_folder, read_only_mode=True)
    original_save = database.get_last_save()
    return True


@app.route("/", methods=["GET", "POST"])
@app.route("/menu", methods=["GET", "POST"])
def menu():
    from flask import redirect, request

    from .llm_providers import get_available_providers, get_current_provider_name

    # Redirect to app if already loaded
    if database:
        return redirect("/app")
    projects = get_projects(db_folder)

    # Get provider info for the UI
    available_providers = get_available_providers()
    current_provider = get_current_provider_name()

    if request.method == "POST" and "project" in request.form:
        project = request.form["project"]
        if project in projects:
            close()
            result = load_project(project)
            if result:
                return redirect("/app")
            return redirect("/menu?error=Folder%20not%20found")
        else:
            return redirect("/create?project=" + project)
    if request.args.get("error"):
        error = request.args.get("error")
        error = urllib.parse.unquote(error)
        return render_template(
            "menu.html",
            projects=projects,
            error=error,
            available_providers=available_providers,
            current_provider=current_provider,
        )
    return render_template(
        "menu.html",
        projects=projects,
        available_providers=available_providers,
        current_provider=current_provider,
    )


@app.route("/create", methods=["GET", "POST"])
def create_project():
    from flask import redirect, request

    if request.method == "GET" and "project" in request.args:
        project = request.args["project"]
        return render_template("create.html", project=project)
    if request.method == "POST":
        if "project" in request.form:
            project = request.form["project"]
            # download the files if they exist in the form under files
            files = request.files.getlist("files")
            # write the files to the ingest
            full_db_folder = os.path.join(db_folder, "scienceai_db")
            os.makedirs(full_db_folder, exist_ok=True)
            project_folder = os.path.join(full_db_folder, project)
            os.makedirs(project_folder, exist_ok=True)
            ingest_folder = os.path.join(project_folder, project.replace(" ", "_") + "_ingest_folder")
            os.makedirs(ingest_folder, exist_ok=True)
            atleast_one_file = False
            for file in files:
                if file.filename == "":
                    continue
                atleast_one_file = True
                file.save(os.path.join(ingest_folder, str(uuid.uuid4()) + ".pdf"))
            # unzip the files in the zip form if they exist
            zips = request.files.getlist("zips")
            for file in zips:
                if file.filename == "":
                    continue
                zip_name = str(uuid.uuid4()) + ".zip"
                os.makedirs(os.path.join(ingest_folder, "zip"), exist_ok=True)
                file.save(os.path.join(ingest_folder, "zip", zip_name))
                # using python to unzip the files
                with zipfile.ZipFile(os.path.join(ingest_folder, "zip", zip_name), "r") as zip_ref:
                    zip_ref.extractall(os.path.join(ingest_folder, "zip"))
                # then delete any non-pdfs or subfolders
                for root, dirs, files in os.walk(os.path.join(ingest_folder, "zip")):
                    for file in files:
                        if file.endswith(".pdf"):
                            if len(dirs) == 0:
                                shutil.move(os.path.join(root, file), os.path.join(ingest_folder, file))
                            else:
                                shutil.move(
                                    os.path.join(root, os.path.join(*dirs), file), os.path.join(ingest_folder, file)
                                )
                            atleast_one_file = True
                shutil.rmtree(os.path.join(ingest_folder, "zip"))
            if not atleast_one_file:
                return redirect("/menu?error=No%20files%20uploaded")
            result = load_project(project)
            if result:
                return redirect("/app")
            return redirect("/menu?error=Failed%20to%20create%20project")
    return redirect("/menu?error=Failed%20to%20Create%20Project")


@app.route("/add_papers_to_existing_project", methods=["POST"])
def add_papers_to_existing_project():
    from flask import jsonify, request

    if not database or not message_queue:
        return jsonify({"error": "No project loaded"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    # 1. Prepare Messages
    num_files = len(files)
    uploading_msg = {
        "content": f"I am uploading {num_files} new papers...",
        "role": "user",
        "status": "Pending",
        "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
    }

    completion_msg = {
        "content": f"Uploaded {num_files} papers. \n\n**Warning:** These papers have been added to the database, but previous analyses have NOT been updated to include them. You must explicitly ask me to re-run any analysis if you want these new papers included.",
        "role": "user",
        "status": "Processed",
        "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
    }

    # 2. File Handling (Save to ingest folder)
    ingest_folder = os.path.join(database.project_path, database.project_name.replace(" ", "_") + "_ingest_folder")
    os.makedirs(ingest_folder, exist_ok=True)

    for file in files:
        if file.filename == "":
            continue
        filename = str(uuid.uuid4()) + ".pdf"
        file_path = os.path.join(ingest_folder, filename)
        file.save(file_path)

    # 3. Send Command to Backend
    message_queue.put({"ADD_PAPERS": True, "uploading_msg": uploading_msg, "completion_msg": completion_msg})

    return jsonify({"status": "success", "message": "Upload started"}), 200


@app.route("/app")
def app_endpoint():
    from flask import redirect

    if database:
        return render_template("app.html")
    return redirect("/menu?error=No%20project%20loaded")


@app.route("/start-database")
def db():
    if not database:
        return script_to_return_to_menu
    db_snippet = database.get_analyst_data_visual("/")
    html_snippet = render_template("db_element.html", data_dict=db_snippet, basepath="Analysts")
    return render_template("db.html", html_snippet=html_snippet)


@app.route("/Analysts", defaults={"path": "/Analysts"})
@app.route("/Analysts/<path:path>")
def update_data(path):
    from urllib.parse import unquote

    if not database:
        return script_to_return_to_menu
    path = unquote(path)
    if path == "/Analysts":
        data_to_return = database.get_analyst_data_visual("/")
    else:
        path = "/Analysts/" + path
        data_to_return = database.get_analyst_data_visual(path)
    # Always render template for dicts (including empty ones)
    # Only return empty for None or other non-dict types
    if isinstance(data_to_return, dict):
        return render_template("db_element.html", data_dict=data_to_return, basepath=path)
    return abort(404, description="Resource not found")


@app.route("/download/<path:filepath>")
def download(filepath):
    from flask import request, send_from_directory

    filepath = urllib.parse.unquote(filepath)
    if filepath[0] != "/" and not sys.platform.startswith("win"):
        filepath = "/" + filepath

    # Check if the file exists before attempting to copy
    if not os.path.isfile(filepath):
        return abort(404, description=f"File not found: {os.path.basename(filepath)}")

    target = os.path.join(path_to_app, "io", os.path.basename(filepath))
    shutil.copyfile(filepath, target)

    @after_this_request
    def remove_file(response):
        if not sys.platform.startswith("win"):
            os.remove(target)
        return response

    dir_path = os.path.dirname(target)
    path = os.path.basename(filepath)
    if request.args.get("attached"):
        return send_from_directory(directory=dir_path, path=path, download_name=path, as_attachment=True)  # type: ignore
    else:
        return send_from_directory(directory=dir_path, path=path, download_name=path, as_attachment=False)  # type: ignore


@sock.route("/discussion")
def discussion(ws):
    # Wait for the backend thread to be ready
    while not database:
        time.sleep(1)

    messages = safe_db_read(database.get_database_chat)
    if messages is None:
        messages = []  # Fallback to empty if read failed

    # Emit initial context usage on connect and get percentage
    context_percentage = calculate_and_emit_context_from_messages(messages)
    can_compress = can_compress_context(messages)
    undo_blocked = is_undo_blocked(messages)

    if len(messages) == 0:
        current = str(uuid.uuid4())
        # Always send initial content even if empty to prevent 5-second refresh fallback
        try:
            ws.send(
                render_template(
                    "chat.html",
                    messages=[],
                    context_percentage=context_percentage,
                    can_compress=can_compress,
                    undo_blocked=undo_blocked,
                )
            )
        except (BrokenPipeError, OSError):
            return  # Client disconnected, exit gracefully
    else:
        current = str(hash(str(messages)))
        filtered_messages = filter_intermediate_messages(messages.copy())
        processed_messages = convert_markdown(replace_pi_generated_file_links(filtered_messages, database.project_path))
        try:
            ws.send(
                render_template(
                    "chat.html",
                    messages=processed_messages,
                    context_percentage=context_percentage,
                    can_compress=can_compress,
                    undo_blocked=undo_blocked,
                )
            )
        except (BrokenPipeError, OSError):
            return  # Client disconnected, exit gracefully
    while True:
        asyncio.run(database.await_update(timeout=20))
        if not database:
            break
        # All database reads are now after the pause check
        messages = safe_db_read(database.get_database_chat)
        if messages is None:
            # Read failed, skip this iteration
            continue
        new = str(hash(str(messages)))
        if new != current:
            current = new
            # Update context usage when chat changes
            context_percentage = calculate_and_emit_context_from_messages(messages)
            can_compress = can_compress_context(messages)
            undo_blocked = is_undo_blocked(messages)
            filtered_messages = filter_intermediate_messages(messages.copy())
            processed_messages = convert_markdown(
                replace_pi_generated_file_links(filtered_messages, database.project_path)
            )
            try:
                ws.send(
                    render_template(
                        "chat.html",
                        messages=processed_messages,
                        context_percentage=context_percentage,
                        can_compress=can_compress,
                        undo_blocked=undo_blocked,
                    )
                )
            except (BrokenPipeError, OSError):
                break  # Client disconnected, exit loop gracefully


@app.route("/send_message", methods=["POST"])
def send_message():
    from flask import render_template, request

    if not database or not message_queue:
        return script_to_return_to_menu
    message = request.form["text"]
    new = {
        "content": message,
        "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
        "role": "user",
        "status": "Pending",
    }
    message_queue.put(new)
    return render_template("chat_update.html")


@app.route("/compress_context", methods=["POST"])
def compress_context():
    from flask import jsonify

    from .backend import _compression_running

    if not database or not message_queue:
        return jsonify({"success": False, "error": "No project loaded"}), 400

    # Check if compression is already running - reject immediately, don't queue
    if _compression_running:
        return jsonify({"success": False, "error": "Compression already in progress", "already_running": True}), 409

    # Send compress command to the backend thread
    message_queue.put({"COMPRESS_CONTEXT": True})
    return jsonify({"success": True, "message": "Compression started"})


@app.route("/pause_processing", methods=["POST"])
def pause_processing():
    """
    Request a pause of the PI's message processing loop.
    The loop will be paused at the next safe point (after current tool call completes).
    """
    from flask import jsonify

    from .backend import set_pause_requested

    if not database or not message_queue:
        return jsonify({"success": False, "error": "No project loaded"}), 400

    set_pause_requested(True)
    return jsonify({"success": True, "message": "Pause requested"})


@app.route("/cancel_pause", methods=["POST"])
def cancel_pause():
    """Cancel a pending pause request."""
    from flask import jsonify

    from .backend import set_pause_requested

    if not database or not message_queue:
        return jsonify({"success": False, "error": "No project loaded"}), 400

    set_pause_requested(False)
    return jsonify({"success": True, "message": "Pause cancelled"})


@app.route("/undo_last_request", methods=["POST"])
def undo_last_request():
    """
    Undo the last user request by removing the message and all subsequent responses.
    Also cleans up any analysts created by delegate_research.
    This is a 'smart undo' that reverts associated side effects.
    """
    from flask import jsonify

    if not database or not message_queue:
        return jsonify({"success": False, "error": "No project loaded"}), 400

    messages = database.get_database_chat()
    if not messages:
        return jsonify({"success": False, "error": "No messages to undo"}), 400

    # Find the last user message index
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return jsonify({"success": False, "error": "No user message found"}), 400

    # Check if run_python_code was used in any message after the last user message
    messages_after_user = messages[last_user_idx + 1 :]
    run_python_code_used = False
    analysts_to_delete = []

    for msg in messages_after_user:
        tool_calls = msg.get("tool_calls", [])
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            func_name = func.get("name", "")
            func_args = func.get("arguments", "{}")

            if func_name == "run_python_code":
                run_python_code_used = True

            if func_name == "delegate_research":
                try:
                    args = json.loads(func_args) if isinstance(func_args, str) else func_args
                    analyst_name = args.get("name")
                    if analyst_name:
                        analysts_to_delete.append(analyst_name)
                        # Also attempt to delete potential parallel replicates (up to 5 to be safe)
                        for i in range(1, 6):
                            analysts_to_delete.append(f"{analyst_name} copy {i}")
                except (json.JSONDecodeError, TypeError):
                    pass

    if run_python_code_used:
        return jsonify(
            {
                "success": False,
                "error": "Cannot undo: run_python_code was used. The AI executed code that may have created files or made changes that cannot be automatically reverted.",
                "blocked": True,
            }
        )

    if not database or not message_queue:
        return jsonify({"success": False, "error": "No project loaded"}), 400

    # Send undo command to the backend thread
    message_queue.put(
        {"UNDO_LAST_REQUEST": True, "last_user_idx": last_user_idx, "analysts_to_delete": analysts_to_delete}
    )

    return jsonify({"success": True, "message": "Undo request started"})


@app.route("/reset_conversation", methods=["POST"])
def reset_conversation():
    """
    Reset the entire conversation by removing all analysts, clearing pi_generated,
    and clearing the chat history while keeping papers intact.
    """
    from flask import jsonify

    if not database or not message_queue:
        return jsonify({"success": False, "error": "No project loaded"}), 400

    # Send reset command to the backend thread
    message_queue.put({"RESET_CONVERSATION": True})

    return jsonify({"success": True, "message": "Reset started. Papers are preserved."})


@app.route("/switch_provider", methods=["POST"])
def switch_provider_endpoint():
    from flask import jsonify, request

    from .llm_providers import get_available_providers, get_current_provider_name, switch_provider

    data = request.get_json() or {}
    provider_name = data.get("provider", "").lower()

    if provider_name not in ["openai", "anthropic", "google"]:
        return jsonify({"success": False, "error": "Invalid provider name"}), 400

    available = get_available_providers()
    if not available.get(provider_name, False):
        return jsonify({"success": False, "error": "API key not configured for this provider"}), 400

    success = switch_provider(provider_name)
    if success:
        # Reset cached context to prevent stale values from flashing
        # The new context will be calculated when the discussion WebSocket reconnects
        global current_context
        with context_lock:
            current_context = None
        return jsonify(
            {"success": True, "provider": get_current_provider_name(), "message": f"Switched to {provider_name}"}
        )
    else:
        return jsonify({"success": False, "error": "Failed to switch provider"}), 500


@app.route("/set_parallel_calls", methods=["POST"])
def set_parallel_calls():
    from flask import jsonify, request

    if not database or not message_queue:
        return jsonify({"success": False, "error": "No project loaded"}), 400

    data = request.get_json() or {}
    count = data.get("count", 1)

    message_queue.put({"SET_PARALLEL_CALLS": True, "count": count})
    return jsonify({"success": True, "message": f"Parallel calls set to {count}"})


@app.route("/get_parallel_calls", methods=["GET"])
def get_parallel_calls():
    """Get the current parallel calls setting for the project."""
    from flask import jsonify

    if not database:
        return jsonify({"success": False, "error": "No project loaded"}), 400

    count = database.get_project_setting("n_parallel_calls", default=1)
    return jsonify({"success": True, "count": count})


@app.route("/get_provider_status", methods=["GET"])
def get_provider_status():
    from flask import jsonify

    from .llm_providers import get_available_providers, get_current_provider_name

    return jsonify({"current_provider": get_current_provider_name(), "available_providers": get_available_providers()})


@sock.route("/papers")
def papers(ws):
    if not database:
        return script_to_return_to_menu

    # Wait for the backend thread to be ready
    while not database:
        time.sleep(1)

    papers_dict = safe_db_read(database.get_database_papers)
    if papers_dict is None:
        papers_dict = []  # Fallback to empty if read failed
    if len(papers_dict) == 0:
        current = str(uuid.uuid4())
        # Always send initial content even if empty to prevent 5-second refresh fallback
        try:
            ws.send(render_template("papers.html", papers=[]))
        except BrokenPipeError:
            return  # Client disconnected, exit gracefully
    else:
        current = str(hash(str(papers_dict)))
        try:
            ws.send(render_template("papers.html", papers=papers_dict))
        except (BrokenPipeError, OSError):
            return  # Client disconnected, exit gracefully
    while True:
        asyncio.run(database.await_update(timeout=20))
        if not database:
            break
        # All database reads are now after the pause check
        papers_dict = safe_db_read(database.get_database_papers)
        if papers_dict is None:
            # Read failed, skip this iteration
            continue
        new = str(hash(str(papers_dict)))
        if new != current:
            current = new
            try:
                ws.send(render_template("papers.html", papers=papers_dict))
            except (BrokenPipeError, OSError):
                break  # Client disconnected, exit gracefully


@sock.route("/progress")
def progress(ws):
    """WebSocket endpoint for progress updates"""
    global progress_connections, current_progress
    with progress_lock:
        progress_connections.append(ws)
        # Send current progress immediately if an operation is in progress
        if current_progress:
            with contextlib.suppress(builtins.BaseException):
                ws.send(current_progress)
    try:
        while True:
            # Keep connection alive, actual updates sent via emit_progress()
            time.sleep(1)
    finally:
        with progress_lock:
            if ws in progress_connections:
                progress_connections.remove(ws)


@sock.route("/context")
def context(ws):
    """WebSocket endpoint for context usage updates"""
    global context_connections, current_context
    with context_lock:
        context_connections.append(ws)
        # Send current context immediately if available
        if current_context:
            with contextlib.suppress(builtins.BaseException):
                ws.send(current_context)
    try:
        while True:
            # Keep connection alive, actual updates sent via emit_context()
            time.sleep(1)
    finally:
        with context_lock:
            if ws in context_connections:
                context_connections.remove(ws)


@app.route("/close_project")
def close_project():
    from flask import redirect, render_template, request

    if request.args.get("confirm"):
        global database
        global message_queue
        if database:
            close()
        database = None
        message_queue = None
        return redirect("/menu")
    last_save = None
    if database:
        last_save = database.get_last_save()
        if not last_save:
            ready = False
        else:
            update_time = database.get_update_time().replace(" ", "_").replace(":", "_")
            ready = last_save.find(update_time) > -1 or last_save == original_save
        messages = database.get_database_chat()
        option = False
        if len(messages) > 0:
            if not ready and messages[-1]["status"] == "Processed":
                option = True
            elif messages[-1]["status"] != "Processed":
                ready = False
        if last_save:
            save_time = datetime.strptime(last_save[-19:], "%Y-%m-%d_%H_%M_%S")
            pretty_time = save_time.strftime("%B %d, %Y %I:%M:%S %p %Z")
        else:
            pretty_time = None
        return render_template("close.html", last_save=pretty_time, ready=ready, option=option)


@app.route("/export_papers")
def export_papers():
    from urllib.parse import unquote

    from flask import request, send_from_directory

    if not database:
        return script_to_return_to_menu
    analystName = request.args.get("analyst", "")
    listName = request.args.get("list", "")
    if len(analystName) + len(listName) == 0:
        listName = None
        analystName = None
    try:
        papers = database.get_all_papers(analyst=analystName, named_list=listName)
    except ValueError:
        return abort(404, description="Resource not found")
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "scienceai_paper_export_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(temp_path, exist_ok=True)
    selected_fields = unquote(request.args.get("fields")).split(",")
    sep = unquote(request.args.get("seperator", "_"))
    user_defined_tag = unquote(request.args.get("userDefinedTag", ""))
    inj = "INJECT-TITLE"
    names = []
    for paper in papers:
        title = "NA"
        data = database.get_paper_data(paper.get("paper_id"))
        metadata = data.get("metadata", {})
        name = ""
        for field in selected_fields:
            if field == "User Defined Tag":
                name += user_defined_tag + sep
            elif field == "ScienceAI List":
                if listName:
                    name += listName + sep
                else:
                    name += "NA" + sep
            elif field == "DOI":
                if "DOI" in metadata:
                    name += metadata["DOI"] + sep
                else:
                    name += "NA" + sep
            elif field == "Date of Publication":
                if "created" in metadata:
                    name += metadata["created"]["date-time"][:10] + sep
                else:
                    name += "NA" + sep
            elif field == "First Author":
                if "author" in metadata:
                    name += metadata["author"][0]["given"] + "-" + metadata["author"][0]["family"] + sep
                else:
                    name += "NA" + sep
            elif field == "Title":
                if "title" in metadata:
                    name += inj + sep
                    title = metadata["title"][0]
                else:
                    name += "NA" + sep
            elif field == "Journal":
                if "container-title" in metadata:
                    name += metadata["container-title"][0] + sep
                else:
                    name += "NA" + sep
        name = name[:-1]
        revert = name
        replacements = {
            "<": "_lt_",
            ">": "_gt_",
            ":": "_colon_",
            '"': "_quote_",
            "'": "_quote_",
            "/": "_slash_",
            "\\": "_backslash_",
            "|": "_pipe_",
            "?": "_question_",
            "*": "_asterisk_",
            ".": "_period_",
        }

        # Replace invalid characters
        for invalid_char, replacement in replacements.items():
            name = name.replace(invalid_char, replacement)
            title = title.replace(invalid_char, replacement)

        # Replace any remaining invalid characters with an underscore
        name = re.sub(r"[^\w\-_\. ]", "", name)
        title = re.sub(r"[^\w\-_\. ]", "", title)

        name = name.replace(inj, title) + ".pdf"
        if len(name) > 255:
            # chop off the end of the title
            short_title = title[: 255 - len(name)]
            name = revert.replace(inj, short_title) + ".pdf"
        if len(name) > 255:
            # chop off the end of the title
            short_title = title[: 255 - len(name)]
            name = revert.replace(inj, short_title)[:251] + ".pdf"

        if name in names:
            name = name[:251] + "_" + str(names.count(name) + 1) + ".pdf"

        names.append(name)

        shutil.copyfile(database.get_paper_pdf(paper.get("paper_id")), os.path.join(temp_path, name))
    source = temp_path
    zip_name = "scienceai_paper_export_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".zip"
    destination = os.path.join(path_to_app, "io", zip_name)
    base = os.path.basename(destination)
    name = base.split(".")[0]
    format = base.split(".")[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move(f"{name}.{format}", destination)
    shutil.rmtree(temp_dir)

    dir_path = os.path.dirname(destination)
    path = os.path.basename(destination)

    @after_this_request
    def remove_file(response):
        if not sys.platform.startswith("win"):
            os.remove(destination)
        return response

    return send_from_directory(directory=dir_path, path=path, download_name=path, as_attachment=True)  # type: ignore


@app.route("/save")
def save():
    from flask import redirect

    if not database:
        return script_to_return_to_menu
    database.save_database()
    return redirect("/app")


@app.route("/save_project")
def save_project():
    from flask import render_template

    if not database:
        return script_to_return_to_menu
    last_save = database.get_last_save()
    if last_save:
        update_time = database.get_update_time().replace(" ", "_").replace(":", "_")
        ready = last_save.find(update_time) > -1 or last_save == original_save
    else:
        ready = False
    messages = database.get_database_chat()
    option = False
    if len(messages) > 0 and not ready and messages[-1]["status"] == "Processed":
        option = True
    if last_save:
        save_time = datetime.strptime(last_save[-19:], "%Y-%m-%d_%H_%M_%S")
        pretty_time = save_time.strftime("%B %d, %Y %I:%M:%S %p %Z")
    else:
        pretty_time = None
    return render_template("save.html", last_save=pretty_time, option=option)


@app.route("/download_save")
def download_save():
    from flask import send_from_directory

    if not database:
        return script_to_return_to_menu
    save_path = database.get_last_save(path=True)
    temp_dir = tempfile.mkdtemp()
    project = os.path.basename(database.project_path)
    source = os.path.join(temp_dir, project)
    shutil.copytree(save_path, source)
    zip_name = project.replace(" ", "_") + "_scienceai_save_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".zip"
    destination = os.path.join(path_to_app, "io", zip_name)
    base = os.path.basename(destination)
    name = base.split(".")[0]
    format = base.split(".")[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move(f"{name}.{format}", destination)

    dir_path = os.path.dirname(destination)
    path = os.path.basename(destination)

    @after_this_request
    def remove_file(response):
        if not sys.platform.startswith("win"):
            os.remove(destination)
        return response

    return send_from_directory(directory=dir_path, path=path, download_name=path, as_attachment=True)  # type: ignore


@app.route("/download_analysis")
def download_analysis():
    from flask import send_from_directory

    if not database:
        return script_to_return_to_menu
    analysis_path = database.combine_analyst_tool_trackers()
    project = os.path.basename(database.project_path)
    destination = os.path.join(
        path_to_app,
        "io",
        project.replace(" ", "_") + "_scienceai_analysis_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv",
    )
    shutil.move(analysis_path, destination)
    dir_path = os.path.dirname(destination)
    path = os.path.basename(destination)

    @after_this_request
    def remove_file(response):
        if not sys.platform.startswith("win"):
            os.remove(destination)
        return response

    return send_from_directory(directory=dir_path, path=path, download_name=path, as_attachment=True)  # type: ignore


@app.route("/load_checkpoint", methods=["POST"])
def load_save():
    from flask import redirect, request

    if database:
        close_project()
        return script_to_return_to_menu
    save_file = request.files["checkpoint"]
    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    project = request.form["project"]
    save_path = os.path.join(temp_dir, "save.zip")
    # Extract to a temp extraction folder
    extract_folder = os.path.join(temp_dir, "extracted")
    save_file.save(save_path)
    shutil.unpack_archive(save_path, extract_folder)

    # Find the actual project folder inside the extracted archive
    # There should be exactly one folder at the top level
    extracted_items = os.listdir(extract_folder)
    if len(extracted_items) != 1 or not os.path.isdir(os.path.join(extract_folder, extracted_items[0])):
        shutil.rmtree(temp_dir)
        return redirect("/menu?error=Invalid%20checkpoint%20file%20format")

    found_project_name = extracted_items[0]
    source_project_folder = os.path.join(extract_folder, found_project_name)

    # Rename the ingest folder inside to match the new project name
    # The ingest folder should be named {project_name}_ingest_folder
    old_ingest_name = found_project_name.replace(" ", "_") + "_ingest_folder"
    new_ingest_name = project.replace(" ", "_") + "_ingest_folder"
    old_ingest_path = os.path.join(source_project_folder, old_ingest_name)
    new_ingest_path = os.path.join(source_project_folder, new_ingest_name)

    if os.path.exists(old_ingest_path) and old_ingest_name != new_ingest_name:
        shutil.move(old_ingest_path, new_ingest_path)

    # Update the project name in the database
    # The database stores project names in the update_time dict
    db_path = os.path.join(source_project_folder, "scienceai_ddb")
    if os.path.exists(db_path) and found_project_name != project:
        import dictdatabase as DDB

        original_storage = DDB.config.storage_directory
        try:
            DDB.config.storage_directory = db_path
            if DDB.at("update_time").exists():
                update_time_data = DDB.at("update_time").read()
                if found_project_name in update_time_data:
                    # Rename the key from old project name to new project name
                    update_time_data[project] = update_time_data.pop(found_project_name)
                    DDB.at("update_time").create(update_time_data, force_overwrite=True)

            # Also update metadata if it exists
            if DDB.at("metadata").exists():
                metadata = DDB.at("metadata").read()
                if found_project_name in metadata:
                    metadata[project] = metadata.pop(found_project_name)
                    DDB.at("metadata").create(metadata, force_overwrite=True)
        finally:
            DDB.config.storage_directory = original_storage

    # Prepare the final destination
    projects_folder = os.path.join(db_folder, "scienceai_db")
    if not os.path.exists(projects_folder):
        os.makedirs(projects_folder)
    project_path = os.path.join(projects_folder, project)

    # Check if project already exists
    if os.path.exists(project_path):
        if request.form.get("overwrite"):
            # Use a more robust deletion method for macOS
            def remove_readonly(func, path, excinfo):
                """Error handler for shutil.rmtree to handle read-only files."""
                os.chmod(path, 0o777)  # nosec B103
                func(path)

            try:
                shutil.rmtree(project_path, onerror=remove_readonly)
            except OSError:
                # If still fails, try with subprocess (more aggressive)
                import subprocess  # nosec B404

                subprocess.run(["rm", "-rf", project_path], check=False)  # nosec B603, B607
        else:
            shutil.rmtree(temp_dir)
            return redirect("/menu?error=Project%20already%20exists")

    # Move the extracted project to the final location with the new name
    shutil.move(source_project_folder, project_path)
    shutil.rmtree(temp_dir)
    result = load_project(project)
    if result:
        return redirect("/app")
    return redirect("/menu?error=Failed%20to%20load%20project")


@app.route("/delete_project", methods=["POST"])
def delete_project():
    from flask import redirect, request

    if database:
        return redirect("/app")
    project = request.form["project"]
    project_path = os.path.join(db_folder, "scienceai_db")
    checkpoints = []
    for dir in os.listdir(project_path):
        if dir.find(project + "_-checkpoint-_") > -1:
            checkpoints.append(os.path.join(project_path, dir))
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)
    shutil.rmtree(os.path.join(project_path, project))
    return redirect("/menu")


@app.route("/shutdown", methods=["GET", "POST"])
def shutdown():
    from flask import redirect

    if database:
        return redirect("/app")
    global own_pid  # Make sure to use the global variable
    os.kill(own_pid, 9)


atexit.register(close)


def main():
    import argparse

    from .llm_providers import (
        get_available_providers,
        get_current_provider_name,
        save_api_key,
        setup_api_keys_interactive,
        switch_provider,
        validate_all_configured_keys,
        validate_api_key,
    )

    parser = argparse.ArgumentParser(
        description="ScienceAI - AI-powered scientific literature analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  scienceai                           # Start the web server
  scienceai --setup-keys              # Interactive API key setup
  scienceai --set-key openai sk-...   # Set a specific API key
  scienceai --validate-keys           # Validate all configured keys
  scienceai --provider anthropic      # Start with a specific provider
  scienceai -v                        # Start with verbose (INFO) logging
  scienceai --debug                   # Start with debug logging
  scienceai --log-level WARNING       # Set specific log level
        """,
    )
    parser.add_argument(
        "--setup-keys",
        action="store_true",
        help="Interactive setup for API keys",
    )
    parser.add_argument(
        "--set-key",
        nargs=2,
        metavar=("PROVIDER", "KEY"),
        help="Set an API key for a provider (openai, anthropic, google)",
    )
    parser.add_argument(
        "--validate-keys",
        action="store_true",
        help="Validate all configured API keys",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google"],
        help="Set the default LLM provider",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip API key validation on startup",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4242,
        help="Port to run the server on (default: 4242)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (DEBUG level, more verbose than -v)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level explicitly (overrides -v and --debug)",
    )
    parser.add_argument(
        "--gcp-service-account",
        metavar="PATH",
        help="Path to GCP service account JSON file (for Gemini and/or Claude on Vertex AI)",
    )
    parser.add_argument(
        "--remove-gcp-config",
        action="store_true",
        help="Remove GCP service account configuration (reverts to API key)",
    )

    args = parser.parse_args()

    # Handle --setup-keys
    if args.setup_keys:
        setup_api_keys_interactive()
        return

    # Handle --set-key
    if args.set_key:
        provider, key = args.set_key
        if provider.lower() not in ["openai", "anthropic", "google"]:
            print(f"✗ Invalid provider: {provider}")
            print("  Valid providers: openai, anthropic, google")
            sys.exit(1)

        if save_api_key(provider, key):
            print(f"✓ Saved {provider} API key")
            # Validate the key
            print("  Validating...")
            is_valid, message = validate_api_key(provider)
            if is_valid:
                print(f"  ✓ Key is valid: {message}")
            else:
                print(f"  ✗ Key validation failed: {message}")
        else:
            print(f"✗ Failed to save {provider} API key")
            sys.exit(1)
        return

    # Handle --gcp-service-account
    if args.gcp_service_account:
        from .llm_providers import save_gcp_config

        sa_path = args.gcp_service_account

        # Validate file exists
        if not os.path.exists(sa_path):
            print(f"✗ Service account file not found: {sa_path}")
            sys.exit(1)

        # Load and validate JSON
        try:
            with open(sa_path) as f:
                sa_data = json.load(f)
            project_id = sa_data.get("project_id")
            if not project_id:
                print("✗ Invalid service account JSON (missing project_id)")
                sys.exit(1)
        except (json.JSONDecodeError, OSError) as e:
            print(f"✗ Invalid service account JSON: {e}")
            sys.exit(1)

        print(f"\n✓ Valid service account file for project: {project_id}")
        print("  This service account can be used for:")
        print("    1. Google Gemini (native GCP models)")
        print("    2. Claude on Vertex AI (Anthropic partner models)")
        print()

        # Ask about Claude on Vertex
        claude_response = input("Use this service account for Claude on Vertex AI? (y/n): ").strip().lower()
        use_for_claude = claude_response in ["y", "yes"]

        # Ask for region if using for Claude (or always)
        if use_for_claude:
            print("\nCommon Vertex AI regions:")
            print("  - us-east5 (US East)")
            print("  - us-central1 (US Central)")
            print("  - europe-west1 (Europe West)")
            region = input("Enter Vertex AI region (default: us-east5): ").strip()
            if not region:
                region = "us-east5"
        else:
            region = "us-east5"  # Default for Gemini

        # Save configuration (always for Gemini, optionally for Claude)
        if save_gcp_config(
            service_account_path=sa_path,
            project_id=project_id,
            region=region,
            use_for_gemini=True,  # Always configure for Gemini
            use_for_claude=use_for_claude,
        ):
            print("\n✓ Configured GCP service account:")
            print(f"  Project: {project_id}")
            print(f"  Region: {region}")
            print("  Gemini: ✓")
            print(f"  Claude on Vertex: {'✓' if use_for_claude else '✗'}")
        else:
            print("\n✗ Failed to save configuration")
            sys.exit(1)
        return

    # Handle --remove-gcp-config
    if args.remove_gcp_config:
        from .llm_providers import load_gcp_config, remove_gcp_config

        gemini_config = load_gcp_config("google_gcp")
        claude_config = load_gcp_config("anthropic_vertex")

        if not gemini_config and not claude_config:
            print("\n✗ No GCP service account configuration found")
            sys.exit(1)

        print("\nCurrent GCP configuration:")
        if gemini_config:
            print(f"  ✓ Gemini: {gemini_config['project_id']} ({gemini_config.get('region', 'us-east5')})")
        if claude_config:
            print(f"  ✓ Claude on Vertex: {claude_config['project_id']} ({claude_config.get('region', 'us-east5')})")
        print()

        remove_gemini = False
        remove_claude = False

        if gemini_config:
            response = input("Remove Gemini GCP config? (y/n): ").strip().lower()
            remove_gemini = response in ["y", "yes"]

        if claude_config:
            response = input("Remove Claude Vertex config? (y/n): ").strip().lower()
            remove_claude = response in ["y", "yes"]

        if not remove_gemini and not remove_claude:
            print("\n✗ No configurations removed")
            return

        if remove_gcp_config(remove_gemini=remove_gemini, remove_claude=remove_claude):
            print("\n✓ Removed GCP configuration:")
            if remove_gemini:
                print("  - Gemini (will use API key if configured)")
            if remove_claude:
                print("  - Claude on Vertex (use direct Anthropic API instead)")
        else:
            print("\n✗ Failed to remove configuration")
            sys.exit(1)
        return

    # Handle --validate-keys
    if args.validate_keys:
        print("\nValidating configured API keys...\n")
        results = validate_all_configured_keys()

        any_valid = False
        for provider_name, (is_valid, message) in results.items():
            status = "✓" if is_valid else "✗"
            print(f"  {status} {provider_name}: {message}")
            if is_valid:
                any_valid = True

        if not any_valid:
            print("\n⚠ No valid API keys configured.")
            sys.exit(1)
        else:
            print("\n✓ All validations complete.")
        return

    # Handle --provider
    if args.provider:
        if not switch_provider(args.provider):
            available = get_available_providers()
            if not available.get(args.provider, False):
                print(f"✗ Cannot use {args.provider}: No API key configured")
                print("  Use --setup-keys to configure API keys")
                sys.exit(1)
            else:
                print(f"✗ Failed to switch to {args.provider}")
                sys.exit(1)

    # Check that at least one provider is available
    available = get_available_providers()
    if not any(available.values()):
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║  ⚠ No API keys configured!                               ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print("\nScienceAI requires at least one LLM provider API key.")
        print("Run 'scienceai --setup-keys' to configure your API keys.\n")
        print("Or set one directly:")
        print("  scienceai --set-key openai YOUR_OPENAI_KEY")
        print("  scienceai --set-key anthropic YOUR_ANTHROPIC_KEY")
        print("  scienceai --set-key google YOUR_GOOGLE_KEY\n")
        sys.exit(1)

    # Validate keys on startup (unless skipped)
    if not args.skip_validation:
        print("Validating API keys...")
        results = validate_all_configured_keys()

        valid_providers = []
        for provider_name, (is_valid, message) in results.items():
            if is_valid:
                valid_providers.append(provider_name)
                print(f"  ✓ {provider_name}: {message}")
            elif available.get(provider_name, False):
                # Key was configured but is invalid
                print(f"  ✗ {provider_name}: {message}")

        if not valid_providers:
            print("\n⚠ No valid API keys found!")
            print("Please check your API keys with: scienceai --validate-keys")
            sys.exit(1)

        # Make sure current provider is valid
        current = get_current_provider_name()
        if current not in valid_providers:
            # Switch to a valid provider
            new_provider = valid_providers[0]
            switch_provider(new_provider)
            print(f"\n  Switched to {new_provider} (previous provider not available)")

    # Configure logging based on CLI arguments
    if args.log_level:
        log_level = getattr(logging, args.log_level)
    elif args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING  # Default: only warnings and above

    # Configure root logger with format
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure Werkzeug (Flask's HTTP logger) - suppress unless in debug mode
    werkzeug_log = logging.getLogger("werkzeug")
    if log_level <= logging.DEBUG:
        werkzeug_log.setLevel(logging.DEBUG)
    else:
        werkzeug_log.setLevel(logging.ERROR)

    # Print startup info
    logger = logging.getLogger("scienceai")
    current_provider = get_current_provider_name()
    logger.info(f"Using {current_provider.upper()} as LLM provider")
    logger.info("ScienceAI is running")

    url = f"http://localhost:{args.port}"
    logger.info(f"Access the web interface at: {url}")

    # Also print the clickable link to stdout for convenience if running interactively
    if sys.stdout.isatty():
        if not sys.platform.startswith("win"):
            print(f"\033]8;;{url}\a{url}\033]8;;\a")
        else:
            print(url)

    app.run(host="localhost", port=args.port, debug=False)


if __name__ == "__main__":
    main()
