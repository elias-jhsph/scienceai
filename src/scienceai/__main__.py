import asyncio
import re
import os
import shutil
import sys
import threading
import time
import urllib
import uuid
import zipfile
import tempfile
import markdown2
import json


from flask_sock import Sock
from flask import Flask, render_template, abort, after_this_request
from .database_manager import DatabaseManager, get_projects
from .backend import run_backend
from multiprocessing import Queue
import atexit
from datetime import datetime
from .llm import client

# Compiled regex pattern for common HTML tags
COMMON_HTML_TAG_PATTERN = re.compile(
    r'^</?(?:a|abbr|address|area|article|aside|audio|b|base|bdi|bdo|blockquote|body|br|button|'
    r'canvas|caption|cite|code|col|colgroup|data|datalist|dd|del|details|dfn|dialog|div|dl|dt|'
    r'em|embed|fieldset|figcaption|figure|footer|form|h1|h2|h3|h4|h5|h6|head|header|hgroup|hr|'
    r'html|i|iframe|img|input|ins|kbd|label|legend|li|link|main|map|mark|meta|meter|nav|noscript|'
    r'object|ol|optgroup|option|output|p|param|picture|pre|progress|q|rp|rt|ruby|s|samp|script|'
    r'section|select|small|source|span|strong|style|sub|summary|sup|svg|table|tbody|td|template|'
    r'textarea|tfoot|th|thead|time|title|tr|track|u|ul|var|video|wbr)(?:\s|>|/)',
    re.IGNORECASE
)

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
sock = Sock(app)

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
CONTEXT_LIMIT = 400000  # GPT-5.1 context limit (400,000 tokens)

db_folder = os.path.join(os.path.expanduser('~'), 'Documents', "ScienceAI")
if not os.path.exists(db_folder):
    os.makedirs(db_folder)
path_to_app = os.path.dirname(os.path.abspath(__file__))
path_to_python = sys.executable
script_to_return_to_menu = "<script>window.location.href = '/menu';</script>"


def emit_progress(current, total, description="Processing", analyst_name=None):
    """Emit progress update to all connected WebSocket clients"""
    global progress_connections, current_progress
    message_data = {
        "current": current,
        "total": total,
        "description": description
    }
    if analyst_name:
        message_data["analyst_name"] = analyst_name
    
    message = json.dumps(message_data)
    print(message)
    
    # Store current progress state for new connections
    with progress_lock:
        current_progress = message
        for ws in progress_connections[:]:  # Copy list to avoid modification during iteration
            try:
                ws.send(message)
            except:
                progress_connections.remove(ws)
    
    # Clear progress state when complete
    if current >= total and total > 0:
        with progress_lock:
            current_progress = None


def emit_context(tokens_used, tokens_limit=None, can_compress=True):
    """Emit context usage update to all connected WebSocket clients"""
    global context_connections, current_context
    if tokens_limit is None:
        tokens_limit = CONTEXT_LIMIT
    percentage = min(100, round((tokens_used / tokens_limit) * 100, 1))
    message_data = {
        "type": "context",
        "tokens_used": tokens_used,
        "tokens_limit": tokens_limit,
        "percentage": percentage,
        "can_compress": can_compress
    }
    
    message = json.dumps(message_data)
    
    # Store current context state for new connections
    with context_lock:
        current_context = message
        for ws in context_connections[:]:  # Copy list to avoid modification during iteration
            try:
                ws.send(message)
            except:
                context_connections.remove(ws)


def calculate_and_emit_context_from_messages(messages):
    """Calculate token count from chat messages and emit context usage. Returns percentage."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        
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
        
        # Add estimate for system message (~2000 tokens typically)
        total_tokens += 2000
        
        # Check if compression is possible
        can_compress = can_compress_context(messages)
        
        emit_context(total_tokens, can_compress=can_compress)
        
        # Return the percentage for template use
        return min(100, round((total_tokens / CONTEXT_LIMIT) * 100, 1))
    except Exception as e:
        # Don't let context tracking break the main flow
        return 0


def can_compress_context(messages):
    """Check if there are any uncompressed tool messages that can be compressed."""
    for msg in messages:
        # Skip already compressed messages
        if msg.get("compressed"):
            continue
        if msg.get("role") == "tool" or msg.get("tool_calls"):
            return True
    return False


def close():
    global thread
    global stop_event
    global message_queue
    global database
    global original_save
    if thread:
        message_queue.put({"TERMINATE": True})
        stop_event.set()
        thread.join()
    thread = None
    stop_event = None
    message_queue = None
    database = None
    original_save = None
    for file in os.listdir(os.path.join(path_to_app, "io")):
        os.remove(os.path.join(path_to_app, "io", file))


def sanitize_for_id(value):
    """Replace special characters that are problematic in CSS selectors with dashes."""
    # Replace all problematic characters with dashes
    value = re.sub(r'[/\s.&()\[\]{}<>:;,?!@#$%^*+=|\\~`\'"]+', '-', value)
    # Remove leading/trailing dashes and convert to lowercase
    return re.sub(r'^-+|-+$', '', value).lower()


app.jinja_env.filters['sanitize_for_id'] = sanitize_for_id
app.jinja_env.filters['quote_url'] = lambda u: urllib.parse.quote(u)


def convert_markdown(messages):
    # messages is a list of dictionaries with keys one of which is content - convert content from markdown to html
    cp_text_dict = []
    for message in messages:
        try:
            content = message["content"]
            
            # PASS 1: Escape any < or > that are NOT part of valid HTML tags
            def escape_if_invalid(match):
                """Escape the tag if it's not a common HTML tag."""
                tag = match.group(0)
                if COMMON_HTML_TAG_PATTERN.match(tag):
                    return tag  # Keep valid HTML tags as-is
                else:
                    return tag.replace('<', '&lt;').replace('>', '&gt;')  # Escape invalid tags
            
            # Find all <...> patterns and escape invalid ones
            content = re.sub(r'<[^>]+>', escape_if_invalid, content)
            
            # PASS 2: Store valid HTML tags with placeholders to protect from markdown processing
            # This prevents markdown from interpreting underscores in HTML attributes as emphasis
            html_blocks = []
            import uuid
            placeholder_prefix = str(uuid.uuid4()).replace('-', '')
            
            def store_html(match):
                """Store valid HTML tag and return placeholder."""
                tag = match.group(0)
                # At this point, only valid HTML tags remain (invalid ones are escaped)
                idx = len(html_blocks)
                html_blocks.append(tag)
                return f"HTMLBLOCK{placeholder_prefix}{idx}PLACEHOLDER"
            
            # Store all remaining HTML tags (these are all valid)
            content = re.sub(r'<[^>]+>', store_html, content)
            
            # Now process markdown (valid HTML is protected, invalid tags are already escaped)
            content = markdown2.markdown(content)
            
            # Restore HTML blocks
            for i, html_block in enumerate(html_blocks):
                placeholder = f"HTMLBLOCK{placeholder_prefix}{i}PLACEHOLDER"
                content = re.sub(re.escape(placeholder), html_block, content)
            
            # Final cleanup: Remove any unmatched tags
            # Parse all tags and track which ones are matched
            def remove_unmatched_tags(html):
                """Remove unmatched opening or closing tags."""
                # Find all tags
                tag_pattern = re.compile(r'<(/?)([a-zA-Z][a-zA-Z0-9]*)[^>]*?(/?)>')
                tags = []
                
                for match in tag_pattern.finditer(html):
                    is_closing = bool(match.group(1))  # Is it </tag>?
                    tag_name = match.group(2).lower()
                    is_self_closing = bool(match.group(3)) or tag_name in {
                        'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 
                        'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'
                    }
                    
                    tags.append({
                        'match': match,
                        'is_closing': is_closing,
                        'is_self_closing': is_self_closing,
                        'tag_name': tag_name,
                        'start': match.start(),
                        'end': match.end(),
                        'matched': is_self_closing  # Self-closing tags are always "matched"
                    })
                
                # Match opening and closing tags
                stack = []
                for tag in tags:
                    if tag['is_self_closing']:
                        continue
                    elif not tag['is_closing']:
                        # Opening tag
                        stack.append(tag)
                    else:
                        # Closing tag - find matching opening tag
                        found = False
                        # Look backwards in stack for the nearest matching opening tag
                        # This handles cases like <div><span>Text</div></span> where </div> matches <div>
                        # and <span> is left unmatched inside
                        for i in range(len(stack) - 1, -1, -1):
                            if stack[i]['tag_name'] == tag['tag_name']:
                                # Found a match!
                                stack[i]['matched'] = True
                                tag['matched'] = True
                                # Everything between the matched opening tag and this closing tag
                                # that is still on the stack is effectively unmatched (improperly nested)
                                # But we leave them on the stack for now, they just won't be marked matched
                                stack.pop(i) 
                                # We also need to remove everything that was AFTER the matched tag on the stack
                                # because those are now "orphaned" by this closure
                                # e.g. <div><span></div> -> <span> is orphaned
                                del stack[i:] 
                                found = True
                                break
                        # If no match found, tag remains unmatched (matched=False from init)
                
                # Remove unmatched tags
                # Build result by removing unmatched tag positions
                unmatched_ranges = [(tag['start'], tag['end']) for tag in tags if not tag['matched']]
                
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
        except Exception as e:
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
    
    # Build a mapping of filenames to full paths
    file_map = {}
    for filename in pi_files:
        full_path = os.path.join(pi_generated_path, filename)
        if os.path.isfile(full_path):
            file_map[filename] = full_path
    
    if not file_map:
        return messages
    
    # Process each message
    for message in messages:
        content = message.get("content", "")
        if not content or not isinstance(content, str):
            continue
        
        # Skip messages that already have file links (run_python_code output)
        if 'pi-generated-content' in content:
            continue
        
        # Replace file references
        for filename, full_path in file_map.items():
            # Patterns to match:
            # 1. Full absolute path
            # 2. Relative path with pi_generated/
            # 3. Just the filename (word boundary)
            
            # Build replacement HTML based on file type
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in ['.csv']:
                # CSV: view + download buttons
                replacement = (
                    f'<span class="pi-file-link">'
                    f'<span class="pi-file-name">{filename}</span>'
                    f'<span class="pi-file-buttons">'
                    f'<button class="icon-button" onclick="viewCSV(\'download/{full_path}\')"><i class="fa fa-eye"></i></button>'
                    f'<a href="/download/{full_path}?attached=T" class="icon-button"><i class="fas fa-download"></i></a>'
                    f'</span></span>'
                )
            elif ext in ['.json']:
                # JSON: view + download buttons
                replacement = (
                    f'<span class="pi-file-link">'
                    f'<span class="pi-file-name">{filename}</span>'
                    f'<span class="pi-file-buttons">'
                    f'<button class="icon-button" onclick="viewJSON(\'download/{full_path}\')"><i class="fa fa-eye"></i></button>'
                    f'<a href="/download/{full_path}?attached=T" class="icon-button"><i class="fas fa-download"></i></a>'
                    f'</span></span>'
                )
            elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                # Images: view inline + download
                replacement = (
                    f'<span class="pi-file-link">'
                    f'<span class="pi-file-name">{filename}</span>'
                    f'<span class="pi-file-buttons">'
                    f'<a href="/download/{full_path}" target="_blank" class="icon-button"><i class="fa fa-eye"></i></a>'
                    f'<a href="/download/{full_path}?attached=T" class="icon-button"><i class="fas fa-download"></i></a>'
                    f'</span></span>'
                )
            else:
                # Other files: just download
                replacement = (
                    f'<span class="pi-file-link">'
                    f'<span class="pi-file-name">{filename}</span>'
                    f'<span class="pi-file-buttons">'
                    f'<a href="/download/{full_path}?attached=T" class="icon-button"><i class="fas fa-download"></i></a>'
                    f'</span></span>'
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
                flexible_filename = escaped_filename.replace(r'\_', r'[_ ]?')
                pattern = re.compile(
                    r'(?<![a-zA-Z0-9_/"\'])' + flexible_filename + r'(?![a-zA-Z0-9_/"\'])',
                    re.IGNORECASE
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
    ingest_folder = os.path.join(db_folder, "scienceai_db", project, project.replace(" ", "_")+"_ingest_folder")
    if not os.path.exists(ingest_folder):
        return False
    thread = threading.Thread(target=run_backend, args=(ingest_folder, project, db_folder, message_queue, stop_event))
    thread.start()
    time.sleep(1)
    database = DatabaseManager(ingest_folder, None, project, storage_path=db_folder, read_only_mode=True)
    original_save = database.get_last_save()
    return True


@app.route('/', methods=['GET', 'POST'])
@app.route('/menu', methods=['GET', 'POST'])
def menu():
    from flask import redirect, request
    # Redirect to app if already loaded
    if database:
        return redirect('/app')
    projects = get_projects(db_folder)
    if request.method == 'POST':
        if "project" in request.form:
            project = request.form["project"]
            if project in projects:
                close()
                result = load_project(project)
                if result:
                    return redirect('/app')
                return redirect('/menu?error=Folder%20not%20found')
            else:
                return redirect('/create?project='+project)
    if request.args.get("error"):
        error = request.args.get("error")
        error = urllib.parse.unquote(error)
        return render_template('menu.html', projects=projects, error=error)
    return render_template('menu.html', projects=projects)


@app.route('/create', methods=['GET', 'POST'])
def create_project():
    from flask import redirect, request
    if request.method == 'GET':
        if "project" in request.args:
            project = request.args["project"]
            return render_template('create.html', project=project)
    if request.method == 'POST':
        if "project" in request.form:
            project = request.form["project"]
            # download the files if they exist in the form under files
            files = request.files.getlist("files")
            # write the files to the ingest
            full_db_folder = os.path.join(db_folder, "scienceai_db")
            os.makedirs(full_db_folder, exist_ok=True)
            project_folder = os.path.join(full_db_folder, project)
            os.makedirs(project_folder, exist_ok=True)
            ingest_folder = os.path.join(project_folder, project.replace(" ", "_")+"_ingest_folder")
            os.makedirs(ingest_folder, exist_ok=True)
            atleast_one_file = False
            for file in files:
                if file.filename == "":
                    continue
                atleast_one_file = True
                file.save(os.path.join(ingest_folder, str(uuid.uuid4())+".pdf"))
            # unzip the files in the zip form if they exist
            zips = request.files.getlist("zips")
            for file in zips:
                if file.filename == "":
                    continue
                zip_name = str(uuid.uuid4())+".zip"
                os.makedirs(os.path.join(ingest_folder, "zip"), exist_ok=True)
                file.save(os.path.join(ingest_folder, "zip", zip_name))
                # using python to unzip the files
                with zipfile.ZipFile(os.path.join(ingest_folder, "zip", zip_name), 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(ingest_folder, "zip"))
                # then delete any non-pdfs or subfolders
                for root, dirs, files in os.walk(os.path.join(ingest_folder, "zip")):
                    for file in files:
                        if file.endswith(".pdf"):
                            if len(dirs) == 0:
                                shutil.move(os.path.join(root, file), os.path.join(ingest_folder, file))
                            else:
                                shutil.move(os.path.join(root, os.path.join(*dirs), file), os.path.join(ingest_folder, file))
                            atleast_one_file = True
                shutil.rmtree(os.path.join(ingest_folder, "zip"))
            if not atleast_one_file:
                return redirect('/menu?error=No%20files%20uploaded')
            result = load_project(project)
            if result:
                return redirect('/app')
            return redirect('/menu?error=Failed%20to%20create%20project')
    return redirect('/menu?error=Failed%20to%20Create%20Project')
@app.route('/add_papers_to_existing_project', methods=['POST'])
def add_papers_to_existing_project():
    from flask import request, jsonify
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
        "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
    }
    
    completion_msg = {
        "content": f"Uploaded {num_files} papers. \n\n**Warning:** These papers have been added to the database, but previous analyses have NOT been updated to include them. You must explicitly ask me to re-run any analysis if you want these new papers included.",
        "role": "user",
        "status": "Processed",
        "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
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
    message_queue.put({
        "ADD_PAPERS": True,
        "uploading_msg": uploading_msg,
        "completion_msg": completion_msg
    })

    return jsonify({"status": "success", "message": "Upload started"}), 200

@app.route('/app')
def app_endpoint():
    from flask import redirect
    if database:
        return render_template('app.html')
    return redirect('/menu?error=No%20project%20loaded')


@app.route('/start-database')
def db():
    if not database:
        return script_to_return_to_menu
    db_snippet = database.get_analyst_data_visual("/")
    html_snippet = render_template('db_element.html', data_dict=db_snippet, basepath="Analysts")
    return render_template('db.html', html_snippet=html_snippet)


@app.route('/Analysts', defaults={'path': '/Analysts'})
@app.route('/Analysts/<path:path>')
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
        return render_template('db_element.html', data_dict=data_to_return, basepath=path)
    return abort(404, description="Resource not found")


@app.route('/download/<path:filepath>')
def download(filepath):
    from flask import send_from_directory, request
    filepath = urllib.parse.unquote(filepath)
    if filepath[0] != '/' and not sys.platform.startswith("win"):
        filepath = "/"+filepath
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
        return send_from_directory(directory=dir_path, path=path, as_attachment=True)
    else:
        return send_from_directory(directory=dir_path, path=path, as_attachment=False)


@sock.route('/discussion')
def discussion(ws):
    if not database:
        return script_to_return_to_menu
    messages = database.get_database_chat()
    
    # Emit initial context usage on connect and get percentage
    context_percentage = calculate_and_emit_context_from_messages(messages)
    can_compress = can_compress_context(messages)
    
    if len(messages) == 0:
        current = str(uuid.uuid4())
    else:
        current = str(hash(str(database.get_database_chat())))
        filtered_messages = filter_intermediate_messages(messages.copy())
        processed_messages = convert_markdown(replace_pi_generated_file_links(filtered_messages, database.project_path))
        ws.send(render_template('chat.html', messages=processed_messages, context_percentage=context_percentage, can_compress=can_compress))
    while True:
        asyncio.run(database.await_update(timeout=20))
        if not database:
            break
        messages = database.get_database_chat()
        new = str(hash(str(database.get_database_chat())))
        if new != current:
            current = new
            # Update context usage when chat changes
            context_percentage = calculate_and_emit_context_from_messages(messages)
            can_compress = can_compress_context(messages)
            filtered_messages = filter_intermediate_messages(messages.copy())
            processed_messages = convert_markdown(replace_pi_generated_file_links(filtered_messages, database.project_path))
            ws.send(render_template('chat.html', messages=processed_messages, context_percentage=context_percentage, can_compress=can_compress))


@app.route('/send_message', methods=['POST'])
def send_message():
    from flask import request, render_template
    if not database or not message_queue:
        return script_to_return_to_menu
    message = request.form['text']
    new = {"content": message, "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z'), "role": "user",
                       "status": "Pending"}
    message_queue.put(new)
    return render_template('chat_update.html')


@app.route('/compress_context', methods=['POST'])
def compress_context():
    from flask import jsonify
    if not database or not message_queue:
        return jsonify({"success": False, "error": "No project loaded"}), 400
    
    # Send compress command to the backend thread
    message_queue.put({"COMPRESS_CONTEXT": True})
    return jsonify({"success": True, "message": "Compression started"})


@sock.route('/papers')
def papers(ws):
    if not database:
        return script_to_return_to_menu
    papers_dict = database.get_database_papers()
    if len(papers_dict) == 0:
        current = str(uuid.uuid4())
    else:
        current = str(hash(str(database.get_database_papers())))
        ws.send(render_template('papers.html', papers=papers_dict))
    while True:
        asyncio.run(database.await_update(timeout=20))
        if not database:
            break
        new = str(hash(str(database.get_database_papers())))
        if new != current:
            papers_dict = database.get_database_papers()
            current = new
            ws.send(render_template('papers.html', papers=papers_dict))


@sock.route('/progress')
def progress(ws):
    """WebSocket endpoint for progress updates"""
    global progress_connections, current_progress
    with progress_lock:
        progress_connections.append(ws)
        # Send current progress immediately if an operation is in progress
        if current_progress:
            try:
                ws.send(current_progress)
            except:
                pass
    try:
        while True:
            # Keep connection alive, actual updates sent via emit_progress()
            time.sleep(1)
    finally:
        with progress_lock:
            if ws in progress_connections:
                progress_connections.remove(ws)


@sock.route('/context')
def context(ws):
    """WebSocket endpoint for context usage updates"""
    global context_connections, current_context
    with context_lock:
        context_connections.append(ws)
        # Send current context immediately if available
        if current_context:
            try:
                ws.send(current_context)
            except:
                pass
    try:
        while True:
            # Keep connection alive, actual updates sent via emit_context()
            time.sleep(1)
    finally:
        with context_lock:
            if ws in context_connections:
                context_connections.remove(ws)


@app.route('/close_project')
def close_project():
    from flask import redirect, render_template, request
    if request.args.get("confirm"):
        global database
        global message_queue
        if database:
            close()
        database = None
        message_queue = None
        return redirect('/menu')
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
            pretty_time = save_time.strftime('%B %d, %Y %I:%M:%S %p %Z')
        else:
            pretty_time = None
        return render_template('close.html', last_save=pretty_time, ready=ready, option=option)


@app.route('/export_papers')
def export_papers():
    from flask import send_from_directory, request
    from urllib.parse import unquote, quote
    if not database:
        return script_to_return_to_menu
    analystName = request.args.get("analyst", "")
    listName = request.args.get("list", "")
    if len(analystName)+len(listName) == 0:
        listName = None
        analystName = None
    try:
        papers = database.get_all_papers(analyst=analystName, named_list=listName)
    except ValueError:
        return abort(404, description="Resource not found")
    temp_dir = tempfile.mktemp()
    temp_path = os.path.join(temp_dir, "scienceai_paper_export_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
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
            if field == 'User Defined Tag':
                name += user_defined_tag + sep
            elif field == 'ScienceAI List':
                if listName:
                    name += listName + sep
                else:
                    name += "NA" + sep
            elif field == 'DOI':
                if "DOI" in metadata:
                    name += metadata["DOI"] + sep
                else:
                    name += "NA" + sep
            elif field == 'Date of Publication':
                if "created" in metadata:
                    name += metadata["created"]["date-time"][:10] + sep
                else:
                    name += "NA" + sep
            elif field == 'First Author':
                if "author" in metadata:
                    name += metadata["author"][0]["given"] + "-" + metadata["author"][0]["family"] + sep
                else:
                    name += "NA" + sep
            elif field == 'Title':
                if "title" in metadata:
                    name += inj + sep
                    title = metadata["title"][0]
                else:
                    name += "NA" + sep
            elif field == 'Journal':
                if "container-title" in metadata:
                    name += metadata["container-title"][0] + sep
                else:
                    name += "NA" + sep
        name = name[:-1]
        revert = name
        replacements = {
            '<': '_lt_',
            '>': '_gt_',
            ':': '_colon_',
            '"': '_quote_',
            "'": '_quote_',
            '/': '_slash_',
            '\\': '_backslash_',
            '|': '_pipe_',
            '?': '_question_',
            '*': '_asterisk_',
            '.': '_period_',
        }

        # Replace invalid characters
        for invalid_char, replacement in replacements.items():
            name = name.replace(invalid_char, replacement)
            title = title.replace(invalid_char, replacement)

        # Replace any remaining invalid characters with an underscore
        name = re.sub(r'[^\w\-_\. ]', '', name)
        title = re.sub(r'[^\w\-_\. ]', '', title)

        name = name.replace(inj, title)+".pdf"
        if len(name) > 255:
            # chop off the end of the title
            short_title = title[:255-len(name)]
            name = revert.replace(inj, short_title)+".pdf"
        if len(name) > 255:
            # chop off the end of the title
            short_title = title[:255-len(name)]
            name = revert.replace(inj, short_title)[:251]+".pdf"

        if name in names:
            name = name[:251]+"_"+str(names.count(name)+1)+".pdf"

        names.append(name)

        shutil.copyfile(database.get_paper_pdf(paper.get("paper_id")), os.path.join(temp_path, name))
    source = temp_path
    zip_name = "scienceai_paper_export_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".zip"
    destination = os.path.join(path_to_app, "io", zip_name)
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s' % (name, format), destination)
    shutil.rmtree(temp_dir)

    dir_path = os.path.dirname(destination)
    path = os.path.basename(destination)

    @after_this_request
    def remove_file(response):
        if not sys.platform.startswith("win"):
            os.remove(destination)
        return response
    return send_from_directory(directory=dir_path, path=path, as_attachment=True)


@app.route('/save')
def save():
    from flask import redirect
    if not database:
        return script_to_return_to_menu
    database.save_database()
    return redirect('/app')


@app.route('/save_project')
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
    if len(messages) > 0:
        if not ready and messages[-1]["status"] == "Processed":
            option = True
    if last_save:
        save_time = datetime.strptime(last_save[-19:], "%Y-%m-%d_%H_%M_%S")
        pretty_time = save_time.strftime('%B %d, %Y %I:%M:%S %p %Z')
    else:
        pretty_time = None
    return render_template('save.html', last_save=pretty_time, option=option)


@app.route('/download_save')
def download_save():
    from flask import send_from_directory
    if not database:
        return script_to_return_to_menu
    save_path = database.get_last_save(path=True)
    temp_dir = tempfile.mktemp()
    project = os.path.basename(database.project_path)
    source = os.path.join(temp_dir, project)
    shutil.copytree(save_path, source)
    zip_name = project.replace(" ", "_")+"_scienceai_save_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".zip"
    destination = os.path.join(path_to_app, "io", zip_name)
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s' % (name, format), destination)

    dir_path = os.path.dirname(destination)
    path = os.path.basename(destination)

    @after_this_request
    def remove_file(response):
        if not sys.platform.startswith("win"):
            os.remove(destination)
        return response
    return send_from_directory(directory=dir_path, path=path, as_attachment=True)


@app.route('/download_analysis')
def download_analysis():
    from flask import send_from_directory
    if not database:
        return script_to_return_to_menu
    analysis_path = database.combine_analyst_tool_trackers()
    project = os.path.basename(database.project_path)
    destination = os.path.join(path_to_app, "io", project.replace(" ", "_")+"_scienceai_analysis_"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".csv")
    shutil.move(analysis_path, destination)
    dir_path = os.path.dirname(destination)
    path = os.path.basename(destination)

    @after_this_request
    def remove_file(response):
        if not sys.platform.startswith("win"):
            os.remove(destination)
        return response
    return send_from_directory(directory=dir_path, path=path, as_attachment=True)


@app.route('/load_checkpoint', methods=['POST'])
def load_save():
    from flask import request, redirect
    if database:
        close_project()
        return script_to_return_to_menu
    save_file = request.files["checkpoint"]
    temp_dir = tempfile.mktemp()
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
        return redirect('/menu?error=Invalid%20checkpoint%20file%20format')
    
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
            shutil.rmtree(project_path)
        else:
            shutil.rmtree(temp_dir)
            return redirect('/menu?error=Project%20already%20exists')
    
    # Move the extracted project to the final location with the new name
    shutil.move(source_project_folder, project_path)
    shutil.rmtree(temp_dir)
    result = load_project(project)
    if result:
        return redirect('/app')
    return redirect('/menu?error=Failed%20to%20load%20project')


@app.route('/delete_project', methods=['POST'])
def delete_project():
    from flask import redirect, request
    if database:
        return redirect('/app')
    project = request.form["project"]
    project_path = os.path.join(db_folder, "scienceai_db")
    checkpoints = []
    for dir in os.listdir(project_path):
        if dir.find(project+"_-checkpoint-_") > -1:
            checkpoints.append(os.path.join(project_path, dir))
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)
    shutil.rmtree(os.path.join(project_path, project))
    return redirect('/menu')


@app.route('/shutdown', methods=['GET', 'POST'])
def shutdown():
    from flask import redirect
    if database:
        return redirect('/app')
    global own_pid  # Make sure to use the global variable
    os.kill(own_pid, 9)


atexit.register(close)


def main():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    # print a clickable link to the user to open the app by navigating to the link
    print("ScienceAI is running. Please open the following link in your browser to access the application:")
    if sys.platform.startswith("win"):
        print("http://localhost:4242")
    else:
        print("\033]8;;http://localhost:4242\ahttp://localhost:4242\033]8;;\a")
    app.run(host='localhost', port=4242, debug=False)


if __name__ == '__main__':
    main()
