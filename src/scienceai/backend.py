import traceback


def run_backend(folder, project_path, storage_path, message_queue, stop_event):
    try:
        from .llm import update_stop_event
        update_stop_event(stop_event)
        from .process_paper import process_paper
        from .database_manager import DatabaseManager
        from .principle_investigator import PrincipalInvestigator
        import subprocess
        import sys
        import time
        start = time.time()
        dm = DatabaseManager(folder, process_paper, project_path, storage_path=storage_path)
        pi = PrincipalInvestigator(dm)
        if time.time() - start > 5*60:
            dm.save_database()
            if sys.platform == "darwin":
                subprocess.Popen(["say", "ScienceAI is ready"])
        while True:
            if message_queue.empty():
                time.sleep(1)
            else:
                message = message_queue.get()
                if message.get("TERMINATE"):
                    print("Terminating backend")
                    break
                elif stop_event.is_set():
                    print("Stop event set. Terminating backend")
                    break
                start = time.time()
                pi.process_message(**message)
                end = time.time()
                if end - start > 10 and sys.platform == "darwin":
                    subprocess.Popen(["say", "New message from ScienceAI"])
                dm.save_database()
    except Exception as e:
        print("Backend error")
        traceback.print_exc()
        raise e

