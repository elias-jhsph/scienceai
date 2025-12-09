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
    def __init__(self, project_name=None, storage_path=None, n_workers=5, provider=None, validate_keys=True):
        """Initialize ScienceAI client.

        Args:
            project_name: Name of the project. If None, uses timestamp.
            storage_path: Where to store project data. Defaults to ~/Documents/ScienceAI.
            n_workers: Number of worker threads for processing.
            provider: LLM provider to use ('openai', 'anthropic', 'google'). Defaults to configured provider.
            validate_keys: Whether to validate API keys on startup. Default True.
        """
        import tempfile
        import warnings
        from datetime import datetime

        # Validate API keys if requested
        if validate_keys:
            self._ensure_valid_provider(provider)
        elif provider:
            self.set_provider(provider)

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

    def _ensure_valid_provider(self, preferred_provider=None):
        """Ensure a valid provider is configured."""
        from .llm_providers import (
            get_available_providers,
            get_current_provider_name,
            switch_provider,
            validate_api_key,
        )

        available = get_available_providers()

        # If no keys at all, raise an error
        if not any(available.values()):
            raise RuntimeError(
                "No API keys configured. Please set up API keys first:\n"
                "  - Run 'scienceai --setup-keys' in terminal, or\n"
                "  - Use ScienceAI.set_api_key('openai', 'your-key')"
            )

        # If a specific provider was requested, validate and switch to it
        if preferred_provider:
            if not available.get(preferred_provider.lower(), False):
                raise RuntimeError(
                    f"API key for '{preferred_provider}' not configured. "
                    f"Available providers: {[k for k, v in available.items() if v]}"
                )
            is_valid, msg = validate_api_key(preferred_provider)
            if not is_valid:
                raise RuntimeError(f"API key for '{preferred_provider}' is invalid: {msg}")
            switch_provider(preferred_provider)
            return

        # Validate current provider
        current = get_current_provider_name()
        if available.get(current, False):
            is_valid, msg = validate_api_key(current)
            if is_valid:
                return  # Current provider is fine

        # Find a valid provider
        for provider_name, is_available in available.items():
            if is_available:
                is_valid, msg = validate_api_key(provider_name)
                if is_valid:
                    switch_provider(provider_name)
                    return

        # No valid providers found
        raise RuntimeError(
            "No valid API keys found. Please check your API keys:\n"
            "  - Run 'scienceai --validate-keys' in terminal, or\n"
            "  - Use ScienceAI.validate_keys()"
        )

    @staticmethod
    def get_provider():
        """Get the current LLM provider name.

        Returns:
            str: Current provider name ('openai', 'anthropic', or 'google')
        """
        from .llm_providers import get_current_provider_name

        return get_current_provider_name()

    @staticmethod
    def set_provider(provider_name):
        """Set the LLM provider.

        Args:
            provider_name: Provider to use ('openai', 'anthropic', 'google')

        Returns:
            bool: True if successful

        Raises:
            RuntimeError: If provider is invalid or has no API key
        """
        from .llm_providers import get_available_providers, switch_provider

        provider_name = provider_name.lower()
        if provider_name not in ["openai", "anthropic", "google"]:
            raise RuntimeError(f"Invalid provider: {provider_name}. Must be 'openai', 'anthropic', or 'google'")

        available = get_available_providers()
        if not available.get(provider_name, False):
            raise RuntimeError(f"No API key configured for '{provider_name}'")

        if not switch_provider(provider_name):
            raise RuntimeError(f"Failed to switch to '{provider_name}'")

        return True

    @staticmethod
    def get_available_providers():
        """Get dictionary of available providers and their status.

        Returns:
            dict: Mapping of provider names to availability (True/False)
        """
        from .llm_providers import get_available_providers

        return get_available_providers()

    @staticmethod
    def validate_keys():
        """Validate all configured API keys.

        Returns:
            dict: Mapping of provider names to (is_valid, message) tuples
        """
        from .llm_providers import validate_all_configured_keys

        return validate_all_configured_keys()

    @staticmethod
    def set_api_key(provider_name, api_key):
        """Set an API key for a provider.

        Args:
            provider_name: Provider name ('openai', 'anthropic', 'google')
            api_key: The API key string

        Returns:
            bool: True if successful
        """
        from .llm_providers import save_api_key

        provider_name = provider_name.lower()
        if provider_name not in ["openai", "anthropic", "google"]:
            raise RuntimeError(f"Invalid provider: {provider_name}. Must be 'openai', 'anthropic', or 'google'")

        return save_api_key(provider_name, api_key)

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
