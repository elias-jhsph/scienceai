import time
import sys
import subprocess
from .process_paper import process_paper
from .database_manager import DatabaseManager
from .principle_investigator import PrincipalInvestigator
from datetime import datetime


if __name__ == "__main__":
    folder = sys.argv[1]
    project_path = sys.argv[2]
    storage_path = sys.argv[3]
    dm = DatabaseManager(folder, process_paper, project_path, storage_path=storage_path)
    print("Assistant: Hello, I am ScienceAI. I first need to make sure all your papers are loaded into the "
          "system before I can help you. I will let you know when I am ready to answer your questions. ")
    messages = dm.get_database_chat()
    offset = len(messages)
    if offset == 0:
        print("Assistant: All papers have been loaded into the system. How can I help you today?")
    pi = PrincipalInvestigator(dm)
    for message in messages:
        if message["role"] == "user":
            print("User: " + message["content"])
        else:
            print("Assistant: " + message["content"])
    while True:
        messages = dm.get_database_chat()
        for message in messages[offset:]:
            if message["role"] == "user":
                print("User: " + message["content"])
            else:
                print("Assistant: " + message["content"])
        input_message = input("User: ")
        if input_message == "exit":
            break
        offset += 1
        input_dict = {"content": input_message, "role": "user", "status": "Pending",
                      "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')}
        start = time.time()
        pi.process_message(**input_dict)
        end = time.time()
        if end - start > 10:
            subprocess.Popen(["say", "New message from ScienceAI"])

