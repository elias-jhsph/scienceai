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

from flask_sock import Sock
from flask import Flask, render_template, abort, after_this_request
from .database_manager import DatabaseManager, get_projects
from .backend import run_backend
from multiprocessing import Queue
import atexit
from datetime import datetime
from .llm import client

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
sock = Sock(app)


database = None
stop_event = None
message_queue = None
thread = None
original_save = None
db_folder = os.path.join(os.path.expanduser('~'), 'Documents', "ScienceAI")
if not os.path.exists(db_folder):
    os.makedirs(db_folder)
path_to_app = os.path.dirname(os.path.abspath(__file__))
path_to_python = sys.executable
script_to_return_to_menu = "<script>window.location.href = '/menu';</script>"


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
    """Replace slashes and spaces with dashes."""
    return re.sub("^-|-$", "",
                  value.replace('/', '-').replace(' ', '-').replace('.', '-').replace('--', '-').lower())


app.jinja_env.filters['sanitize_for_id'] = sanitize_for_id
app.jinja_env.filters['quote_url'] = lambda u: urllib.parse.quote(u)


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
    if data_to_return:
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
    if len(messages) == 0:
        current = str(uuid.uuid4())
    else:
        current = str(hash(str(database.get_database_chat())))
        ws.send(render_template('chat.html', messages=messages))
    while True:
        asyncio.run(database.await_update(timeout=60))
        messages = database.get_database_chat()
        new = str(hash(str(database.get_database_chat())))
        if new != current:
            current = new
            ws.send(render_template('chat.html', messages=messages))


@app.route('/send_message', methods=['POST'])
def send_message():
    from flask import request, render_template
    message = request.form['text']
    new = {"content": message, "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z'), "role": "user",
                       "status": "Pending"}
    message_queue.put(new)
    return render_template('chat_update.html')


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
        asyncio.run(database.await_update(timeout=60))
        new = str(hash(str(database.get_database_papers())))
        if new != current:
            papers_dict = database.get_database_papers()
            current = new
            ws.send(render_template('papers.html', papers=papers_dict))


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
    unzip_folder_path = os.path.join(temp_dir, project)
    folder_path = os.path.join(unzip_folder_path, project)
    save_file.save(save_path)
    shutil.unpack_archive(save_path, unzip_folder_path)
    found_project_name = None
    for dir in os.listdir(temp_dir):
        found_project_name = os.path.basename(dir)
    unzip_folder_path = os.path.join(unzip_folder_path, found_project_name)
    shutil.move(unzip_folder_path, folder_path)
    projects_folder = os.path.join(db_folder, "scienceai_db")
    if not os.path.exists(projects_folder):
        os.makedirs(projects_folder)
    project_path = os.path.join(projects_folder, project)
    if os.path.exists(project_path):
        if request.form.get("overwrite"):
            shutil.rmtree(project_path)
        else:
            return redirect('/menu?error=Project%20already%20exists')
    shutil.move(folder_path, project_path)
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
