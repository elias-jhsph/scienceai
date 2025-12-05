import os
import shutil
import threading
import time
import uuid
from datetime import datetime
from queue import Queue

from .backend import run_backend
from .database_manager import DatabaseManager


class ScienceAI:
    def __init__(self, project_name=None, storage_path=None, n_workers=5):
        import tempfile
        import warnings
        from datetime import datetime

        if project_name is None:
            self.project_name = f"Project Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            if storage_path is None:
                warnings.warn(
                    "No project name or storage path provided. Using temporary storage. Data will NOT be persisted after this session.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            self.project_name = project_name

        if storage_path is None:
            if project_name is None:
                self.storage_path = tempfile.mkdtemp()
            else:
                self.storage_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
        else:
            self.storage_path = storage_path

        self.n_workers = n_workers
        self.message_queue = Queue()
        self.error_queue = Queue()
        self.stop_event = threading.Event()
        self.thread = None
        self.database = None

        # State flags
        self.papers_uploaded = False
        self.preprocessing_started = False
        self.last_action_time = None

        # Ensure storage path exists
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        self.ingest_folder = os.path.join(
            self.storage_path, "scienceai_db", self.project_name, self.project_name.replace(" ", "_") + "_ingest_folder"
        )
        if not os.path.exists(self.ingest_folder):
            os.makedirs(self.ingest_folder)

        # Start the backend thread automatically
        self.thread = threading.Thread(
            target=run_backend,
            args=(self.ingest_folder, self.project_name, self.storage_path, self.message_queue, self.stop_event),
            kwargs={"ingest": False, "error_queue": self.error_queue},  # Do not ingest on load
        )
        self.thread.start()

        # Initialize database manager for read access
        # We wait a bit to ensure backend has initialized the DB structure if it's new
        time.sleep(1)
        self.database = DatabaseManager(
            self.ingest_folder, None, self.project_name, storage_path=self.storage_path, read_only_mode=True
        )

    def preprocess(self):
        """
        Blocking version of preprocess.
        """
        self.preprocess_background()
        self.wait()

    def preprocess_background(self):
        """
        Non-blocking preprocess. Triggers ingestion in backend.
        """
        self.message_queue.put({"INGEST": True})
        self.preprocessing_started = True
        self.last_action_time = datetime.now()

    def chat(self, message):
        """
        Blocking chat.
        """
        self.chat_background(message)
        return self.wait()

    def chat_background(self, message):
        """
        Non-blocking chat.
        """
        if not self.papers_uploaded:
            raise RuntimeError("No papers uploaded. Please upload papers first.")
        if not self.preprocessing_started:
            raise RuntimeError("Papers not preprocessed. Please call preprocess() first.")

        from datetime import datetime

        new_msg = {
            "content": message,
            "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
            "role": "user",
            "status": "Pending",
        }
        self.message_queue.put(new_msg)
        self.last_action_time = datetime.now()

    def upload_papers(self, file_paths, trigger_preprocess=True):
        """
        Blocking upload.
        """
        self.upload_papers_background(file_paths, trigger_preprocess=trigger_preprocess)
        if trigger_preprocess:
            self.wait()

    def upload_papers_background(self, file_paths, trigger_preprocess=True):
        """
        Non-blocking upload.
        """
        for file_path in file_paths:
            if os.path.exists(file_path) and file_path.endswith(".pdf"):
                filename = str(uuid.uuid4()) + ".pdf"
                shutil.copy(file_path, os.path.join(self.ingest_folder, filename))

        self.papers_uploaded = True

        if trigger_preprocess:
            self.preprocess_background()

    def wait(self, timeout=None):
        """
        Blocks until the last message status is "Processed".
        Returns the content of the last assistant message.
        """
        start_time = time.time()
        while True:
            # Check for background errors
            if not self.error_queue.empty():
                error = self.error_queue.get()
                raise error

            if timeout and (time.time() - start_time > timeout):
                # We do not stop the backend, just raise timeout
                raise TimeoutError("Timed out waiting for response")

            # Wait for message queue to be empty (backend picked up request)
            if not self.message_queue.empty():
                time.sleep(0.1)
                continue

            # Refresh database view
            self.database.update_update_time()
            messages = self.database.get_database_chat()

            if not messages:
                time.sleep(0.5)
                continue

            last_msg = messages[-1]

            # Check if PI is done
            if last_msg["status"] == "Processed":
                # Ensure no earlier messages are pending
                if not any(m["status"] == "Pending" for m in messages):
                    # Check if the last message is newer than our last action
                    if self.last_action_time:
                        try:
                            # Parse message time
                            # Format: '%B %d, %Y %I:%M:%S %p %Z'
                            # Note: %Z might be tricky, but let's try.
                            # If parsing fails, we might fallback or assume it's new.
                            # Actually, let's just compare if it's the *same* message as before?
                            # No, we need time.
                            msg_time_str = last_msg["time"]
                            # Remove timezone for simpler parsing if needed, but let's try full parse first
                            # Assuming standard format from backend
                            msg_time = datetime.strptime(msg_time_str, "%B %d, %Y %I:%M:%S %p %Z")

                            # We need to handle timezone awareness.
                            # datetime.now() is local. msg_time is local string.
                            # So direct comparison should work if both are naive or both aware.
                            # strptime returns naive by default usually unless %Z is handled specifically.

                            # If msg_time is older than last_action_time, we are seeing old state.
                            # Add a small buffer for clock skew/execution time?
                            # last_action_time was set BEFORE queue put.
                            # msg_time is set by backend AFTER queue get.
                            # So msg_time > last_action_time should hold.

                            # If msg_time is older than last_action_time, we are seeing old state.
                            # We ignore microseconds for this comparison because the message time format
                            # does not include them, which can cause race conditions for fast operations.
                            if msg_time < self.last_action_time.replace(microsecond=0):
                                time.sleep(0.5)
                                continue
                        except ValueError:
                            # If parsing fails, we can't verify time.
                            # Warn and proceed? Or wait?
                            # Let's assume it's fine if we can't parse, to avoid deadlock.
                            pass

                    if last_msg["role"] == "assistant":
                        return last_msg["content"]
                    return last_msg["content"]

            time.sleep(0.5)

    def poll(self):
        """
        Returns None if background is running (status="Pending").
        Returns last message content if status="Processed".
        """
        # Check for background errors
        if not self.error_queue.empty():
            error = self.error_queue.get()
            raise error

        if not self.database:
            return None

        self.database.update_update_time()
        messages = self.database.get_database_chat()

        if not messages:
            return None

        if any(m["status"] == "Pending" for m in messages):
            return None

        return messages[-1]["content"]

    def history(self):
        """
        Returns the full chat history.
        """
        if self.database:
            self.database.update_update_time()
            return self.database.get_database_chat()
        return []

    def close(self):
        if self.thread:
            self.message_queue.put({"TERMINATE": True})
            self.stop_event.set()
            self.thread.join()
